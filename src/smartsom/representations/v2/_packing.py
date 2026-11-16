"""One V2 packing layout for numeric environment observations and Torch callers."""

import numpy as np

from smartsom.representations.v2.observation import FEATURES
from smartsom.representations.v2.rules import CATALOG


class NumpyArrays:
    """The two allocation operations used by the shared packing layout."""

    float32 = np.float32
    bool = np.bool_
    int64 = np.int64

    @staticmethod
    def tensor(values, *, dtype=None):
        inferred = np.asarray(values)
        if inferred.dtype.kind not in "biuf" or (
            dtype is None
            and (
                inferred.dtype.kind != "f"
                or np.max(np.abs(inferred), initial=0).item() >= 2**63
            )
        ):
            # Delegate unusual custom inputs to the existing conversion rules:
            # NumPy accepts strings/None and has different integer inference
            # and rounding. Do not maintain a second Torch dtype inferrer.
            import torch

            torch_dtype = (
                None if dtype is None else getattr(torch, np.dtype(dtype).name)
            )
            return torch.tensor(values, dtype=torch_dtype).numpy()
        # Convert the original values: going through inferred float64 first
        # can change rounding for a mixed batch of large integers and floats.
        return np.asarray(values, dtype=np.float32 if dtype is None else dtype)

    @staticmethod
    def zeros(*shape, dtype=None):
        return np.zeros(shape, dtype=np.float32 if dtype is None else dtype)


def peers_array(relations, width, *, arrays, field="peers"):
    b, c = len(relations), len(relations[0])
    p = max(1, max(len(row[field]) for rows in relations for row in rows))
    data, mask = arrays.zeros(b, c, p, width), arrays.zeros(b, c, p, dtype=arrays.bool)
    for i, rows in enumerate(relations):
        for j, relation in enumerate(rows):
            values = relation[field]
            if values:
                data[i, j, : len(values)] = arrays.tensor(values)
                mask[i, j, : len(values)] = True
    return data, mask


def pack(requests, *, critics, options, arrays):
    width = max(len(r.candidates) for r in requests)
    result = {"size": len(requests), "width": width, "stages": {}}
    if critics is not None:
        result["critic"] = arrays.tensor(critics, dtype=arrays.float32)
    for stage in dict.fromkeys(r.stage for r in requests):
        indices = [i for i, r in enumerate(requests) if r.stage == stage]
        rows = [requests[i] for i in indices]
        group = rows[0].group
        point = group + "." + stage
        n = len(rows)
        size = FEATURES[point] + len(CATALOG[point])
        data = {
            "indices": arrays.tensor(indices, dtype=arrays.int64),
            "context": arrays.tensor(
                [r.inputs["context"] for r in rows], dtype=arrays.float32
            ),
            "candidates": arrays.zeros(n, width, size),
            "modes": arrays.zeros(n, width, 3),
            "mask": arrays.zeros(n, width, dtype=arrays.bool),
        }
        for i, row in enumerate(rows):
            count = len(row.candidates)
            data["candidates"][i, :count] = arrays.tensor(row.inputs["candidates"])
            data["mask"][i, :count] = True
            if group == "machine":
                data["modes"][i, :count] = arrays.tensor(row.inputs["modes"])
        if rows[0].inputs["social"]:
            relations = [
                [*r.inputs["social"], *({} for _ in range(width - len(r.candidates)))]
                for r in rows
            ]
            if group == "dispatcher":
                peers = [
                    [{"peers": r.get("peers", ())} for r in row] for row in relations
                ]
                data["peers"], data["peer_mask"] = peers_array(peers, 3, arrays=arrays)
            elif group in ("post", "input") and stage == "accept":
                jobs = max(
                    1, max(len(r.get("jobs", ())) for row in relations for r in row)
                )
                peers = max(
                    1,
                    max(
                        (
                            len(j["downstream"]["peers"])
                            for row in relations
                            for r in row
                            for j in r.get("jobs", ())
                        ),
                        default=0,
                    ),
                )
                data.update(
                    jobs=arrays.zeros(n, width, jobs, 4 if group == "post" else 3),
                    job_mask=arrays.zeros(n, width, jobs, dtype=arrays.bool),
                    job_peers=arrays.zeros(n, width, jobs, peers, 9),
                    job_peer_mask=arrays.zeros(
                        n, width, jobs, peers, dtype=arrays.bool
                    ),
                    job_meta=arrays.zeros(n, width, jobs, 2),
                    job_count=arrays.zeros(n, width, 1),
                )
                for i, row in enumerate(relations):
                    for k, relation in enumerate(row):
                        supply = relation.get("jobs", ())
                        data["job_count"][i, k, 0] = len(supply) / options.count_scale
                        for j, job in enumerate(supply):
                            station = job["downstream"]
                            count = len(station["peers"])
                            data["jobs"][i, k, j] = arrays.tensor(job["condition"])
                            data["job_mask"][i, k, j] = True
                            data["job_meta"][i, k, j] = arrays.tensor(
                                [count / options.count_scale, station["terminal"]]
                            )
                            if count:
                                data["job_peers"][i, k, j, :count] = arrays.tensor(
                                    station["peers"]
                                )
                                data["job_peer_mask"][i, k, j, :count] = True
            else:
                for name in ("downstream", "alternatives"):
                    if name == "alternatives" and not (
                        group == "pre" and stage == "accept"
                    ):
                        continue
                    peers = [
                        [r.get(name, {"peers": (), "terminal": 0.0}) for r in row]
                        for row in relations
                    ]
                    data[name], data[name + "_mask"] = peers_array(
                        peers, 9, arrays=arrays
                    )
                    count = [
                        [[len(r["peers"]) / options.count_scale] for r in row]
                        for row in peers
                    ]
                    data[name + "_count"] = arrays.tensor(count)
                    if name == "downstream":
                        data[name + "_meta"] = arrays.tensor(
                            [
                                [
                                    [
                                        len(r["peers"]) / options.count_scale,
                                        r["terminal"],
                                    ]
                                    for r in row
                                ]
                                for row in peers
                            ]
                        )
        result["stages"][stage] = data
    return result
