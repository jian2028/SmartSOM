"""Interactive terminal supervising the existing frozen-plan driver."""

import json
import os
import time
from pathlib import Path
from uuid import uuid4

from smartsom.experiments import control

PROMPT_ENV = "SMARTSOM_FOREGROUND_PROMPTS"


def offline_choice(error):
    """Ask the attached terminal; workers never compete for terminal input."""
    endpoint = os.environ.get(PROMPT_ENV)
    if not endpoint:
        return False
    directory = Path(endpoint)
    if not (directory / "session.json").is_file():
        return False
    session = json.loads((directory / "session.json").read_text())
    identity = uuid4().hex
    path = directory / f"{identity}.request.json"
    control.write_json(
        path,
        {"id": identity, "error": str(error), "session": session["id"]},
    )
    answer = directory / f"{identity}.answer.json"
    try:
        while control.alive(session["owner"], control.processes()):
            if not (directory / "session.json").exists():
                break
            if control.requested(directory.parent.parent, fresh=True):
                break
            if answer.is_file():
                receipt = json.loads(answer.read_text())
                return (
                    receipt.get("session") == session["id"]
                    and receipt.get("offline") is True
                )
            time.sleep(0.1)
        return False
    finally:
        path.unlink(missing_ok=True)
        answer.unlink(missing_ok=True)


def pending_prompt(root):
    directory = Path(root) / "control/foreground-prompts"
    session_path = directory / "session.json"
    if not session_path.is_file():
        return None
    session = json.loads(session_path.read_text())
    for path in sorted(directory.glob("*.request.json")):
        request = json.loads(path.read_text())
        if (
            request["session"] == session["id"]
            and not path.with_name(request["id"] + ".answer.json").exists()
        ):
            return request
    return None


def answer_prompt(root, request, *, offline):
    control.write_json(
        Path(root) / "control/foreground-prompts" / (request["id"] + ".answer.json"),
        {"session": request["session"], "offline": offline},
    )


def run(root, *, resume=False, retry_failed=False, display_options=None):
    from smartsom.experiments.background import _check_owner, launch
    from smartsom.telemetry.monitor import monitor
    from smartsom.telemetry.runtime import DisplayOptions, RuntimeDisplay

    root = Path(root).resolve()
    _check_owner(root)
    options = DisplayOptions.from_value(display_options)
    # Publish a readable initial frame before the driver startup handshake.
    display = RuntimeDisplay(options, kind="run", quiet=True)
    display.bind(root)
    display.phase("starting")
    display.close()
    endpoint = root / "control/foreground-prompts"
    endpoint.mkdir(parents=True, exist_ok=True)
    session_path = endpoint / "session.json"
    owner = control.processes()[os.getpid()]
    session = {"id": uuid4().hex, "owner": owner}
    control.write_json(session_path, session)
    try:
        launched = launch(
            root,
            resume=resume,
            retry_failed=retry_failed,
            display_options=display_options,
            prompt_endpoint=endpoint,
        )
        print(f"前台控制：q / Ctrl+C 退出菜单 · d 仅离开界面\n输出：{root}", flush=True)
        code = monitor(root, controlling=True, options=options)
        if not root.exists():
            return {"directory": str(root), "status": "deleted", "deleted": True}
        owner = control.read(root)
        detached = not code and owner.get("status") in control.ACTIVE
        return {
            **launched,
            "background": detached,
            "detached": detached,
            "status": "failed" if code else owner.get("status", "running"),
        }
    finally:
        # Detaching never grants consent to go offline in a later SDK failure.
        session_path.unlink(missing_ok=True)
