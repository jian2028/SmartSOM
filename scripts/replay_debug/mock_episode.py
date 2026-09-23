"""Seeded mock episode generator for the SO-MARL grid factory.

This is NOT the research simulator. It exists so the replay viewer, schema and
metrics can be built and tested before the real grid-factory simulator emits
``smartsom.somarl.replay.v1`` traces. Two controllers:

- ``rule``: reserves destination capacity and cells, so it never collides
  (the slide-4 rule-based controller has 0 conflicts).
- ``marl``: stand-in for an untrained SO-MARL policy. Random moves (epsilon),
  no reservations, a bias towards FAST mode. Its action probabilities are
  synthetic and flagged as such in the manifest.

Rewards follow the slide 18-33 definitions; see ``REWARD_CONFIG``.
"""

from __future__ import annotations

import random
from typing import Any

from .layout import MODES, Grid, default_layout
from .schema import new_manifest

Cell = tuple[int, int]

NOMINAL_TIME = {"O1": 10, "O2": 8, "O3": 12, "O4": 9}
INSPECTION_TIME = 12
SERVICE_TIME = 2
MOVES: dict[str, Cell] = {
    "UP": (0, -1),
    "DOWN": (0, 1),
    "LEFT": (-1, 0),
    "RIGHT": (1, 0),
    "WAIT": (0, 0),
}

REWARD_CONFIG: dict[str, Any] = {
    "machine_operation_done": 0.2,
    "machine_avoidable_idle": -0.001,
    "buffer_nominated_pickup": 0.1,
    "quality_batch_cost": -0.1,
    "quality_resolved": {1: 0.2, 2: 0.5, 3: 0.8, 4: 1.0},
    "dispatcher_correct_handoff": 0.1,
    "dispatcher_failed_service": -0.05,
    "mover_progress_per_cell": 0.1,
    "mover_extra_move": -0.5,
    "mover_env_conflict": -20.0,
    "mover_agv_conflict": -10.0,
    "shared_delivery": {"pass": 10.0, "unknown": 5.0, "fail": -10.0},
    "shared_scrapped": 1.0,
    "shared_on_time": 3.0,
    # Late penalty as a fraction of the positive delivery reward. The slide
    # leaves 50-100 ticks unspecified; this mock uses the 100-200 bracket.
    "shared_late_brackets": [[50, 0.50], [200, 0.75], [None, 1.25]],
}


class MockFactory:
    def __init__(
        self,
        controller: str = "rule",
        seed: int = 0,
        horizon: int = 600,
        num_jobs: int = 60,
        release_every: int = 10,
        num_agvs: int = 4,
    ) -> None:
        if controller not in ("rule", "marl"):
            raise ValueError(f"controller must be 'rule' or 'marl', got {controller!r}")
        self.controller = controller
        self.marl = controller == "marl"
        self.seed = seed
        self.rng = random.Random(seed)
        self.horizon = horizon
        self.num_jobs = num_jobs
        self.release_every = release_every
        self.tick = 0

        self.layout = default_layout()
        self.grid = Grid(self.layout)
        self.stations = {s["id"]: s for s in self.layout["stations"]}
        self.interaction_points: set[Cell] = {
            tuple(p)
            for s in self.layout["stations"]
            for p in s["interaction_points"]  # type: ignore[misc]
        }

        self.buffers: dict[str, list[str | None]] = {
            sid: [None] * s["capacity"]
            for sid, s in self.stations.items()
            if s["type"] in ("pre_buffer", "post_buffer", "input")
        }
        self.nominated: dict[str, str | None] = {sid: None for sid in self.buffers}
        self.reserved_slots: dict[str, int] = {sid: 0 for sid in self.stations}
        self.machines = {
            sid: {
                "status": "idle",
                "job": None,
                "mode": None,
                "remaining": 0,
                "total": 0,
                "op": s["operation"],
            }
            for sid, s in self.stations.items()
            if s["type"] == "machine"
        }
        self.inspection = {
            sid: {
                "status": "idle",
                "slots": [None] * s["capacity"],
                "remaining": 0,
                "first_wait": None,
            }
            for sid, s in self.stations.items()
            if s["type"] == "inspection"
        }
        starts = [
            (3, 2),
            (8, 2),
            (3, 4),
            (8, 4),
            (2, 1),
            (9, 1),
            (2, 5),
            (9, 5),
            (5, 3),
            (6, 3),
        ]
        self.agvs = [
            {
                "id": f"A{i + 1}",
                "pos": starts[i],
                "carrying": None,
                "task": None,
                "status": "idle",
                "service_left": 0,
                "wait_count": 0,
            }
            for i in range(num_agvs)
        ]
        self.jobs: dict[str, dict[str, Any]] = {}
        self.job_specs: list[dict[str, Any]] = []
        self.released = 0
        self.sinks = {"OUT": 0, "D1": 0, "D2": 0}

        # Per-tick scratch.
        self.events: list[dict[str, Any]] = []
        self.decisions: list[dict[str, Any]] = []
        self.local: dict[str, float] = {}
        self.shared = 0.0
        self.just_serviced: set[str] = set()

    # ------------------------------------------------------------------ utils
    def _reward(self, agent: str, value: float) -> None:
        self.local[agent] = self.local.get(agent, 0.0) + value

    def _probs(self, chosen: str, options: list[str]) -> list[list[Any]] | None:
        """Synthetic top-k action probabilities for the mock MARL controller."""
        if not self.marl:
            return None
        p = self.rng.uniform(0.35, 0.9)
        others = [o for o in options if o != chosen][:3]
        rest = [self.rng.random() for _ in others]
        total = sum(rest) or 1.0
        out = [[chosen, round(p, 3)]] + [
            [o, round((1 - p) * r / total, 3)] for o, r in zip(others, rest)
        ]
        return sorted(out, key=lambda kv: -kv[1])

    def _decide(
        self, agent: str, agent_type: str, action: str, options: list[str]
    ) -> None:
        self.decisions.append(
            {
                "agent": agent,
                "agent_type": agent_type,
                "action": action,
                "mask": options,
                "probs": self._probs(action, options),
            }
        )

    def _free_slots(self, sid: str) -> int:
        stype = self.stations[sid]["type"]
        if stype in ("output", "disposal"):
            return 10**6
        slots = (
            self.inspection[sid]["slots"]
            if stype == "inspection"
            else self.buffers[sid]
        )
        return sum(1 for s in slots if s is None) - self.reserved_slots[sid]

    def _nearest_ip(self, sid: str, frm: Cell) -> Cell:
        ips = [tuple(p) for p in self.stations[sid]["interaction_points"]]
        return min(ips, key=lambda p: self.grid.distance(frm, p) or 0)  # type: ignore[return-value]

    def _job_location(self, jid: str) -> str:
        return self.jobs[jid]["location"]

    # --------------------------------------------------------------- phases
    def _release(self) -> None:
        if self.released >= self.num_jobs or self.tick % self.release_every:
            return
        if self._free_slots("IN") <= 0:
            return
        jid = f"J{self.released + 1}"
        rush = self.rng.random() < 0.2
        due = self.tick + (110 if rush else 180) + self.rng.randint(0, 40)
        self.job_specs.append(
            {"id": jid, "release_tick": self.tick, "due_tick": due, "rush": rush}
        )
        self.jobs[jid] = {
            "next_op": 0,
            "quality": "unknown",
            "p_defect": 0.0,
            "defective": False,
            "due_tick": due,
            "rush": rush,
            "location": "IN",
            "reserved_by": None,
        }
        slot = self.buffers["IN"].index(None)
        self.buffers["IN"][slot] = jid
        self.released += 1
        self.events.append({"type": "job_released", "job": jid})

    def _place_in_post(self, mid: str) -> bool:
        post = self.stations[mid]["post_buffer"]
        slots = self.buffers[post]
        if None not in slots:
            return False
        jid = self.machines[mid]["job"]
        slots[slots.index(None)] = jid
        self.jobs[jid]["location"] = post
        return True

    def _step_machines(self) -> None:
        ops = self.layout["operations"]
        for mid, m in self.machines.items():
            if m["status"] == "processing":
                m["remaining"] -= 1
                if m["remaining"] > 0:
                    continue
                job = self.jobs[m["job"]]
                q = MODES[m["mode"]]["defect_rate"]
                introduced = self.rng.random() < q
                job["defective"] = job["defective"] or introduced
                job["p_defect"] = round(1 - (1 - job["p_defect"]) * (1 - q), 4)
                job["next_op"] += 1
                self.events.append(
                    {
                        "type": "op_done",
                        "machine": mid,
                        "job": m["job"],
                        "defect_introduced": introduced,
                    }
                )
                self._reward(mid, REWARD_CONFIG["machine_operation_done"])
                if self._place_in_post(mid):
                    m.update(status="idle", job=None, mode=None, remaining=0, total=0)
                else:
                    m["status"] = "blocked"
                    self.events.append(
                        {"type": "machine_blocked", "machine": mid, "job": m["job"]}
                    )
                continue
            if m["status"] == "blocked":
                if self._place_in_post(mid):
                    m.update(status="idle", job=None, mode=None, remaining=0, total=0)
                continue

            pre = self.stations[mid]["pre_buffer"]
            post = self.stations[mid]["post_buffer"]
            waiting = [j for j in self.buffers[pre] if j is not None]
            if not waiting:
                continue
            options = [f"START({j},{mode})" for j in waiting for mode in MODES] + [
                "WAIT"
            ]
            if None not in self.buffers[post]:
                self._decide(mid, "machine", "WAIT", ["WAIT"])
                continue
            if self.marl:
                if self.rng.random() < 0.1:
                    self._decide(mid, "machine", "WAIT", options)
                    self._reward(mid, REWARD_CONFIG["machine_avoidable_idle"])
                    continue
                jid = self.rng.choice(waiting)
                mode = self.rng.choices(
                    ["FAST", "NORMAL", "SLOW"], weights=[0.5, 0.3, 0.2]
                )[0]
            else:
                jid = min(waiting, key=lambda j: self.jobs[j]["due_tick"])
                job = self.jobs[jid]
                remaining_nominal = sum(NOMINAL_TIME[o] for o in ops[job["next_op"] :])
                slack = job["due_tick"] - self.tick
                mode = (
                    "FAST"
                    if job["rush"] or slack < 1.5 * remaining_nominal + 40
                    else "NORMAL"
                )
            self._decide(mid, "machine", f"START({jid},{mode})", options)
            slots = self.buffers[pre]
            slots[slots.index(jid)] = None
            dur = max(
                1,
                round(NOMINAL_TIME[m["op"]] * MODES[mode]["time_scale"])
                + self.rng.choice([-1, 0, 1]),
            )
            m.update(status="processing", job=jid, mode=mode, remaining=dur, total=dur)
            self.jobs[jid]["location"] = mid
            self.events.append(
                {"type": "op_start", "machine": mid, "job": jid, "mode": mode}
            )

    def _step_inspection(self) -> None:
        for qid, q in self.inspection.items():
            if q["status"] == "inspecting":
                q["remaining"] -= 1
                if q["remaining"] > 0:
                    continue
                results = {}
                for slot in q["slots"]:
                    if slot is not None and slot["state"] == "inspecting":
                        job = self.jobs[slot["job"]]
                        job["quality"] = "fail" if job["defective"] else "pass"
                        job["p_defect"] = 1.0 if job["defective"] else 0.0
                        slot["state"] = "inspected"
                        results[slot["job"]] = job["quality"]
                q.update(status="idle", remaining=0)
                self.events.append(
                    {"type": "inspection_done", "station": qid, "results": results}
                )
                self._reward(
                    qid, REWARD_CONFIG["quality_resolved"].get(len(results), 0.0)
                )
                continue

            pending = [
                s for s in q["slots"] if s is not None and s["state"] == "not_inspected"
            ]
            if not pending:
                q["first_wait"] = None
                continue
            if q["first_wait"] is None:
                q["first_wait"] = self.tick
            full = None not in q["slots"]
            if self.marl:
                start = self.rng.random() < 0.25
            else:
                start = full or self.tick - q["first_wait"] >= 15
            self._decide(
                qid, "quality", "START" if start else "WAIT", ["START", "WAIT"]
            )
            if start:
                for s in pending:
                    s["state"] = "inspecting"
                q.update(
                    status="inspecting", remaining=INSPECTION_TIME, first_wait=None
                )
                self.events.append(
                    {
                        "type": "inspection_start",
                        "station": qid,
                        "jobs": [s["job"] for s in pending],
                    }
                )
                self._reward(qid, REWARD_CONFIG["quality_batch_cost"])

    def _step_nominations(self) -> None:
        for sid, slots in self.buffers.items():
            if self.stations[sid]["type"] == "pre_buffer":
                continue
            jobs = [j for j in slots if j is not None]
            free = [j for j in jobs if self.jobs[j]["reserved_by"] is None]
            current = self.nominated[sid]
            if not jobs:
                self.nominated[sid] = None
                continue
            if self.marl:
                keep = current in jobs and self.rng.random() < 0.7
                choice = current if keep else self.rng.choice(jobs)
            else:
                choice = (
                    min(free, key=lambda j: self.jobs[j]["due_tick"])
                    if free
                    else current
                )
                if choice not in jobs:
                    choice = None
            if choice != current:
                self._decide(
                    sid,
                    "buffer",
                    f"SELECT({choice})" if choice else "WAIT",
                    [f"SELECT({j})" for j in jobs] + ["WAIT"],
                )
            self.nominated[sid] = choice

    # ------------------------------------------------------------- dispatch
    def _destination_for(self, jid: str, frm: Cell) -> str | None:
        job = self.jobs[jid]
        ops = self.layout["operations"]
        if job["quality"] == "pass":
            return "OUT"
        if job["quality"] == "fail":
            if self.marl and self.rng.random() < 0.15:
                return "OUT"
            return min(
                ("D1", "D2"),
                key=lambda d: self.grid.distance(frm, self._nearest_ip(d, frm)) or 0,
            )
        if job["next_op"] < len(ops):
            op = ops[job["next_op"]]
            pres = [
                s["pre_buffer"]
                for s in self.stations.values()
                if s["type"] == "machine" and s["operation"] == op
            ]
            if self.marl:
                return self.rng.choice(pres)
            open_pres = [p for p in pres if self._free_slots(p) > 0]
            if not open_pres:
                return None
            return min(
                open_pres,
                key=lambda p: (
                    -self._free_slots(p),
                    self.grid.distance(frm, self._nearest_ip(p, frm)) or 0,
                ),
            )
        # All operations done, quality unknown: inspect or deliver directly.
        if self.marl:
            return self.rng.choice(["Q1", "Q2"]) if self.rng.random() < 0.4 else "OUT"
        if job["p_defect"] > 0.4:
            open_q = [q for q in ("Q1", "Q2") if self._free_slots(q) > 0]
            if open_q:
                return min(
                    open_q,
                    key=lambda q: (
                        self.grid.distance(frm, self._nearest_ip(q, frm)) or 0
                    ),
                )
        return "OUT"

    def _pickup_candidates(self) -> list[tuple[str, str]]:
        out: list[tuple[str, str]] = []
        for sid, slots in self.buffers.items():
            if self.stations[sid]["type"] == "pre_buffer":
                continue
            nom = self.nominated[sid]
            if nom is not None and nom in slots:
                out.append((nom, sid))
        for qid, q in self.inspection.items():
            for s in q["slots"]:
                if s is not None and s["state"] == "inspected":
                    out.append((s["job"], qid))
        return out

    def _dispatch(self, agv: dict[str, Any]) -> None:
        aid = f"{agv['id']}.dispatch"
        candidates = []
        for jid, sid in self._pickup_candidates():
            if self.jobs[jid]["reserved_by"] is not None and not (
                self.marl and self.rng.random() < 0.1
            ):
                continue
            ip = self._nearest_ip(sid, agv["pos"])
            candidates.append((self.grid.distance(agv["pos"], ip) or 0, jid, sid, ip))
        options = [f"LOAD@{sid}" for _, _, sid, _ in candidates] + ["WAIT"]
        if not candidates:
            return
        candidates.sort()
        pick = (
            candidates[0]
            if not self.marl or self.rng.random() < 0.6
            else self.rng.choice(candidates)
        )
        _, jid, sid, ip = pick
        dest = self._destination_for(jid, ip)
        if dest is None:
            self._decide(aid, "dispatcher", "WAIT", options)
            return
        self._decide(aid, "dispatcher", f"LOAD@{sid}", options)
        self.jobs[jid]["reserved_by"] = agv["id"]
        if not self.marl:
            self.reserved_slots[dest] += 1
        agv["task"] = {
            "job": jid,
            "pickup": sid,
            "dest": dest,
            "phase": "to_pickup",
            "goal": ip,
        }
        agv["status"] = "moving"
        self.events.append(
            {
                "type": "dispatch",
                "agv": agv["id"],
                "job": jid,
                "kind": "load",
                "station": sid,
            }
        )

    # ------------------------------------------------------------- movement
    def _step_agvs(self) -> None:
        for agv in self.agvs:
            if agv["task"] is None and agv["status"] in ("idle", "waiting"):
                self._dispatch(agv)

        order = list(self.agvs)
        if self.marl:
            self.rng.shuffle(order)
        start_pos = {a["id"]: a["pos"] for a in self.agvs}
        final_pos: dict[str, Cell] = {}
        goals = {tuple(a["task"]["goal"]) for a in self.agvs if a["task"]}

        for agv in order:
            aid = agv["id"]
            if aid in self.just_serviced:
                # Pickup/drop-off consumes the tick: no move, so event cell == frame pos.
                final_pos[aid] = agv["pos"]
                continue
            occupied = {
                start_pos[o["id"]]
                for o in self.agvs
                if o["id"] != aid and o["id"] not in final_pos
            }
            occupied |= {p for o, p in final_pos.items() if o != aid}
            moved_into_me = {
                o
                for o, p in final_pos.items()
                if p == agv["pos"] and start_pos[o] != agv["pos"]
            }

            if agv["status"] == "servicing":
                final_pos[aid] = agv["pos"]
                continue
            task = agv["task"]
            if task is None:
                # Idle AGVs step off interaction points / other AGVs' goals.
                if agv["pos"] in self.interaction_points or agv["pos"] in goals:
                    free = [
                        c for c in self.grid.neighbors(agv["pos"]) if c not in occupied
                    ]
                    calm = [
                        c
                        for c in free
                        if c not in self.interaction_points and c not in goals
                    ]
                    if calm or free:
                        agv["pos"] = self.rng.choice(calm or free)
                final_pos[aid] = agv["pos"]
                continue

            goal = tuple(task["goal"])
            if agv["pos"] == goal:
                agv["status"] = "servicing"
                agv["service_left"] = SERVICE_TIME
                final_pos[aid] = agv["pos"]
                continue

            dist = self.grid.distances_from(goal)
            d_old = dist.get(agv["pos"], 0)
            valid = [
                name
                for name, (dx, dy) in MOVES.items()
                if name == "WAIT"
                or self.grid.is_road((agv["pos"][0] + dx, agv["pos"][1] + dy))
            ]
            greedy = [
                n
                for n in valid
                if n != "WAIT"
                and dist.get(
                    (agv["pos"][0] + MOVES[n][0], agv["pos"][1] + MOVES[n][1]), 999
                )
                < d_old
            ]

            if self.marl:
                action = (
                    self.rng.choice(list(MOVES))
                    if self.rng.random() < 0.12
                    else (greedy[0] if greedy else "WAIT")
                )
            else:
                safe = [
                    n
                    for n in greedy
                    if self._target(agv, n) not in occupied
                    and not (
                        moved_into_me
                        and self._target(agv, n)
                        in {start_pos[o] for o in moved_into_me}
                    )
                ]
                if safe:
                    action = safe[0]
                    agv["wait_count"] = 0
                elif agv["wait_count"] >= 3:
                    side = [
                        n
                        for n in valid
                        if n != "WAIT" and self._target(agv, n) not in occupied
                    ]
                    action = self.rng.choice(side) if side else "WAIT"
                    agv["wait_count"] = 0
                else:
                    action = "WAIT"
                    agv["wait_count"] += 1

            mover = f"{aid}.move"
            target = self._target(agv, action)
            if action != "WAIT" and not self.grid.is_road(target):
                self.events.append(
                    {
                        "type": "movement_conflict",
                        "agv": aid,
                        "kind": "env",
                        "cell": list(target),
                        "other": None,
                    }
                )
                self._reward(mover, REWARD_CONFIG["mover_env_conflict"])
            elif action != "WAIT" and (
                target in occupied
                or (moved_into_me and target in {start_pos[o] for o in moved_into_me})
            ):
                other = next(
                    (
                        o["id"]
                        for o in self.agvs
                        if o["id"] != aid
                        and (final_pos.get(o["id"], start_pos[o["id"]]) == target)
                    ),
                    None,
                )
                self.events.append(
                    {
                        "type": "movement_conflict",
                        "agv": aid,
                        "kind": "agv",
                        "cell": list(target),
                        "other": other,
                    }
                )
                self._reward(mover, REWARD_CONFIG["mover_agv_conflict"])
            elif action != "WAIT":
                agv["pos"] = target
                d_new = dist.get(target, d_old)
                if d_new < d_old:
                    self._reward(
                        mover,
                        REWARD_CONFIG["mover_progress_per_cell"] * (d_old - d_new),
                    )
                else:
                    self._reward(mover, REWARD_CONFIG["mover_extra_move"])
            self._decide(mover, "mover", action, list(MOVES))
            agv["status"] = "moving"
            final_pos[aid] = agv["pos"]

    @staticmethod
    def _target(agv: dict[str, Any], action: str) -> Cell:
        dx, dy = MOVES[action]
        return (agv["pos"][0] + dx, agv["pos"][1] + dy)

    # -------------------------------------------------------------- service
    def _remove_from_station(self, jid: str, sid: str) -> bool:
        stype = self.stations[sid]["type"]
        if stype == "inspection":
            for i, s in enumerate(self.inspection[sid]["slots"]):
                if s is not None and s["job"] == jid:
                    self.inspection[sid]["slots"][i] = None
                    return True
            return False
        slots = self.buffers[sid]
        if jid in slots:
            slots[slots.index(jid)] = None
            if self.nominated.get(sid) == jid:
                self.nominated[sid] = None
            return True
        return False

    def _deliver(self, jid: str, sid: str) -> None:
        job = self.jobs.pop(jid)
        self.sinks[sid] += 1
        if self.stations[sid]["type"] == "disposal":
            self.events.append({"type": "scrapped", "job": jid, "station": sid})
            self.shared += REWARD_CONFIG["shared_scrapped"]
            return
        lateness = max(0, self.tick - job["due_tick"])
        on_time = lateness == 0
        base = REWARD_CONFIG["shared_delivery"][job["quality"]]
        if on_time:
            base += REWARD_CONFIG["shared_on_time"]
        elif base > 0:
            for limit, frac in REWARD_CONFIG["shared_late_brackets"]:
                if limit is None or lateness <= limit:
                    base -= base * frac
                    break
        self.shared += base
        self.events.append(
            {
                "type": "delivered",
                "job": jid,
                "observed_quality": job["quality"],
                "defective": job["defective"],
                "due_tick": job["due_tick"],
                "lateness": lateness,
                "on_time": on_time,
            }
        )

    def _step_service(self) -> None:
        for agv in self.agvs:
            if agv["status"] != "servicing":
                continue
            agv["service_left"] -= 1
            if agv["service_left"] > 0:
                continue
            self.just_serviced.add(agv["id"])
            task = agv["task"]
            disp = f"{agv['id']}.dispatch"
            jid = task["job"]
            if task["phase"] == "to_pickup":
                was_nominated = self.nominated.get(task["pickup"]) == jid
                if (
                    jid in self.jobs
                    and self._job_location(jid) == task["pickup"]
                    and self._remove_from_station(jid, task["pickup"])
                ):
                    agv["carrying"] = jid
                    self.jobs[jid]["location"] = agv["id"]
                    self.events.append(
                        {
                            "type": "pickup",
                            "agv": agv["id"],
                            "job": jid,
                            "station": task["pickup"],
                            "cell": list(agv["pos"]),
                        }
                    )
                    if was_nominated:
                        self._reward(
                            task["pickup"], REWARD_CONFIG["buffer_nominated_pickup"]
                        )
                    dest = task["dest"]
                    task.update(
                        phase="to_dropoff", goal=self._nearest_ip(dest, agv["pos"])
                    )
                    self._decide(
                        disp, "dispatcher", f"UNLOAD@{dest}", [f"UNLOAD@{dest}", "WAIT"]
                    )
                    self.events.append(
                        {
                            "type": "dispatch",
                            "agv": agv["id"],
                            "job": jid,
                            "kind": "unload",
                            "station": dest,
                        }
                    )
                    agv["status"] = "moving"
                else:
                    self.events.append(
                        {
                            "type": "service_conflict",
                            "agv": agv["id"],
                            "station": task["pickup"],
                            "job": jid,
                            "reason": "job_not_available",
                        }
                    )
                    self._reward(disp, REWARD_CONFIG["dispatcher_failed_service"])
                    if jid in self.jobs and self.jobs[jid]["reserved_by"] == agv["id"]:
                        self.jobs[jid]["reserved_by"] = None
                    if not self.marl:
                        self.reserved_slots[task["dest"]] -= 1
                    agv.update(task=None, status="idle")
                continue

            dest = task["dest"]
            if self._free_slots(dest) + (0 if self.marl else 1) <= 0:
                self.events.append(
                    {
                        "type": "service_conflict",
                        "agv": agv["id"],
                        "station": dest,
                        "job": jid,
                        "reason": "destination_full",
                    }
                )
                self._reward(disp, REWARD_CONFIG["dispatcher_failed_service"])
                new_dest = self._destination_for(jid, agv["pos"])
                if new_dest and new_dest != dest:
                    task.update(
                        dest=new_dest, goal=self._nearest_ip(new_dest, agv["pos"])
                    )
                    agv["status"] = "moving"
                else:
                    agv["service_left"] = SERVICE_TIME + 2  # retry in place
                continue

            if not self.marl:
                self.reserved_slots[dest] -= 1
            stype = self.stations[dest]["type"]
            agv["carrying"] = None
            self.events.append(
                {
                    "type": "dropoff",
                    "agv": agv["id"],
                    "job": jid,
                    "station": dest,
                    "cell": list(agv["pos"]),
                }
            )
            self.jobs[jid]["reserved_by"] = None
            if stype in ("output", "disposal"):
                self._deliver(jid, dest)
            elif stype == "inspection":
                slots = self.inspection[dest]["slots"]
                slots[slots.index(None)] = {"job": jid, "state": "not_inspected"}
                self.jobs[jid]["location"] = dest
            else:
                slots = self.buffers[dest]
                slots[slots.index(None)] = jid
                self.jobs[jid]["location"] = dest
                if stype == "pre_buffer":
                    self._reward(disp, REWARD_CONFIG["dispatcher_correct_handoff"])
            agv.update(task=None, status="idle")

    # ---------------------------------------------------------------- frame
    def _frame(self) -> dict[str, Any]:
        ops = self.layout["operations"]
        return {
            "tick": self.tick,
            "agvs": [
                {
                    "id": a["id"],
                    "pos": list(a["pos"]),
                    "carrying": a["carrying"],
                    "status": a["status"],
                    "goal": list(a["task"]["goal"]) if a["task"] else None,
                    "task": (
                        {
                            "job": a["task"]["job"],
                            "pickup": a["task"]["pickup"],
                            "dest": a["task"]["dest"],
                            "phase": a["task"]["phase"],
                        }
                        if a["task"]
                        else None
                    ),
                    "battery": None,
                }
                for a in self.agvs
            ],
            "machines": [
                {
                    "id": mid,
                    "status": m["status"],
                    "job": m["job"],
                    "mode": m["mode"],
                    "remaining": m["remaining"],
                    "total": m["total"],
                    "operation": m["op"],
                }
                for mid, m in self.machines.items()
            ],
            "buffers": [
                {"id": sid, "slots": list(slots), "nominated": self.nominated.get(sid)}
                for sid, slots in self.buffers.items()
            ],
            "inspection": [
                {
                    "id": qid,
                    "status": q["status"],
                    "remaining": q["remaining"],
                    "slots": [dict(s) if s else None for s in q["slots"]],
                }
                for qid, q in self.inspection.items()
            ],
            "jobs": {
                jid: {
                    "location": j["location"],
                    "next_task": ops[j["next_op"]]
                    if j["next_op"] < len(ops)
                    else "DELIVER",
                    "done_ops": j["next_op"],
                    "quality": j["quality"],
                    "p_defect": j["p_defect"],
                    "defective": j["defective"],
                    "due_tick": j["due_tick"],
                    "rush": j["rush"],
                }
                for jid, j in self.jobs.items()
            },
            "sinks": dict(self.sinks),
            "decisions": self.decisions,
            "rewards": {
                "shared": round(self.shared, 4),
                "local": {k: round(v, 4) for k, v in self.local.items()},
            },
            "events": self.events,
        }

    def step(self) -> dict[str, Any]:
        self.events, self.decisions, self.local, self.shared = [], [], {}, 0.0
        self.just_serviced = set()
        self._release()
        self._step_machines()
        self._step_inspection()
        self._step_nominations()
        self._step_service()
        self._step_agvs()
        frame = self._frame()
        self.tick += 1
        return frame

    def agents(self) -> list[dict[str, Any]]:
        out = [{"id": mid, "type": "machine", "entity": mid} for mid in self.machines]
        out += [
            {"id": sid, "type": "buffer", "entity": sid}
            for sid in self.buffers
            if self.stations[sid]["type"] != "pre_buffer"
        ]
        out += [
            {"id": qid, "type": "quality", "entity": qid} for qid in self.inspection
        ]
        for a in self.agvs:
            out += [
                {"id": f"{a['id']}.dispatch", "type": "dispatcher", "entity": a["id"]},
                {"id": f"{a['id']}.move", "type": "mover", "entity": a["id"]},
            ]
        return out


def generate_episode(
    controller: str = "rule",
    seed: int = 0,
    horizon: int = 600,
    num_jobs: int = 60,
    release_every: int = 10,
    num_agvs: int = 4,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    factory = MockFactory(controller, seed, horizon, num_jobs, release_every, num_agvs)
    frames = [factory.step() for _ in range(horizon)]
    manifest = new_manifest(
        trace_id=f"mock_{controller}_seed{seed}",
        episode_id=0,
        controller=controller,
        seed=seed,
        horizon=horizon,
        source="scripts/replay_debug/mock_episode.py",
        synthetic=True,
        decision_probs="synthetic" if controller == "marl" else None,
        layout=factory.layout,
        agents=factory.agents(),
        jobs=factory.job_specs,
        reward_config={
            k: v for k, v in REWARD_CONFIG.items() if k != "quality_resolved"
        }
        | {
            "quality_resolved": {
                str(k): v for k, v in REWARD_CONFIG["quality_resolved"].items()
            }
        },
    )
    return manifest, frames
