"""Document identity and save provenance, separate from factory truth."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from uuid import uuid4

from smartsom.config.factory_design import FactoryAuthoring, FactoryDesignFile
from smartsom.domain.factory_design import (
    FactoryDesign,
    GridDesign,
    validate_factory_design,
)


@dataclass
class FactoryDocument:
    design: FactoryDesign
    source_path: Path | None = None
    source_digest: str | None = None
    title: str = "Untitled Factory"
    selected_id: str | None = None
    scene: Any = None
    view: Any = None
    footer: Any = None
    selected_ids: tuple[str, ...] = ()
    edit_mode: bool = False
    undo_stack: Any = None
    interaction: Any = None
    origin: Any = None
    saved_as_template: bool = False
    recovery_id: str = field(default_factory=lambda: uuid4().hex)
    authoring: FactoryAuthoring = field(default_factory=FactoryAuthoring)

    @property
    def file(self):
        return FactoryDesignFile(
            schema="smartsom.factory/v2", factory=self.design, authoring=self.authoring
        )

    @property
    def modified(self):
        return self.undo_stack is not None and not self.undo_stack.isClean()

    @property
    def issues(self):
        return validate_factory_design(self.design)


def blank_design(number: int = 1) -> FactoryDesign:
    return FactoryDesign(
        factory_id=f"factory_{number:03d}",
        name="Untitled Factory",
        grid=GridDesign(width=20, height=15),
    )
