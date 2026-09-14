"""Local template records and recovery snapshots; factory files remain v2 YAML."""

import hashlib
import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path

from smartsom.config.factory_design import (
    load_factory_design,
    load_factory_design_file,
    save_factory_design_file,
)
from smartsom.studio.templates import load_template_file


def file_digest(path):
    path = Path(path)
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else None


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=path.parent, delete=False
        ) as stream:
            temporary = Path(stream.name)
            json.dump(value, stream, ensure_ascii=False, indent=2)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


@dataclass(frozen=True)
class TemplateOrigin:
    name: str
    target_path: Path
    target_digest: str | None
    builtin: int | None = None


class TemplateCatalog:
    def __init__(self, directory):
        self.directory = Path(directory) / "templates"
        self.records_path = self.directory / "catalog.json"

    def records(self):
        if not self.records_path.exists():
            return {"local": [], "files": []}
        value = json.loads(self.records_path.read_text())
        if (
            not isinstance(value, dict)
            or not isinstance(value.get("files"), list)
            or not isinstance(value.get("local"), list)
        ):
            raise ValueError("Invalid local template catalog")
        return value

    def builtin_path(self, number):
        return self.directory / f"template_{number:03d}.yaml"

    def load_builtin(self, number):
        envelope, origin = self.load_builtin_file(number)
        return envelope.factory, origin

    def load_builtin_file(self, number):
        if number not in (1, 2):
            raise ValueError(f"Unknown built-in template: {number}")
        target = self.builtin_path(number)
        if number in self.records()["local"]:
            envelope, digest = load_factory_design_file(target)
        else:
            envelope = load_template_file(number)
            digest = file_digest(target)
        return envelope, TemplateOrigin(f"Template {number}", target, digest, number)

    def use_local(self, number):
        records = self.records()
        records["local"] = sorted(set(records["local"]) | {number})
        write_json(self.records_path, records)

    def restore_original(self, number):
        records = self.records()
        records["local"] = [n for n in records["local"] if n != number]
        write_json(self.records_path, records)

    def register(self, path):
        path = Path(path).expanduser().resolve()
        load_factory_design(path)
        records = self.records()
        records["files"] = list(dict.fromkeys([*records["files"], str(path)]))
        write_json(self.records_path, records)

    def forget(self, path):
        records = self.records()
        records["files"] = [
            p for p in records["files"] if p != str(Path(path).resolve())
        ]
        write_json(self.records_path, records)


class RecoveryStore:
    def __init__(self, directory):
        self.directory = Path(directory) / "recovery"

    def snapshot(self, document):
        self.directory.mkdir(parents=True, exist_ok=True)
        path = self.directory / f"{document.recovery_id}.yaml"
        save_factory_design_file(path, document.file, expected_digest=file_digest(path))
        write_json(
            path.with_suffix(".json"),
            {"title": document.title, "source": str(document.source_path or "")},
        )
        return path

    def entries(self):
        return sorted(
            self.directory.glob("*.yaml"), key=lambda p: p.stat().st_mtime, reverse=True
        )

    def remove(self, identifier):
        # Only application-generated UUID names can address recovery files.
        if len(identifier) != 32 or any(
            c not in "0123456789abcdef" for c in identifier
        ):
            raise ValueError("Invalid recovery identity")
        for suffix in (".yaml", ".json"):
            (self.directory / (identifier + suffix)).unlink(missing_ok=True)
