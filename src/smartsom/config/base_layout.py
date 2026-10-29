"""Reproducible layouts for the two approved Base diagnostic scales."""

from smartsom.domain.factory_design import (
    AGVDesign,
    BufferDesign,
    BufferTarget,
    Cell,
    FactoryDesign,
    Footprint,
    GridDesign,
    MachineDesign,
    PoolStorage,
    PortBinding,
    PortDesign,
)
from smartsom.domain.quality import QualityMode


def base_factory(scale):
    # Keep stable factory/entity identities when renaming the former target scale.
    columns, rows, width, height, factory_id = {
        "small": (4, 2, 24, 11, "base_small"),
        "medium": (5, 4, 29, 19, "base_target"),
    }[scale]
    modes = (
        QualityMode("slow", "1.2", ".01"),
        QualityMode("normal", "1", ".018"),
        QualityMode("fast", ".8", ".03"),
    )
    buffers = [
        BufferDesign(
            "input",
            "Input",
            Footprint(0, 0, 1, height),
            "system_input",
            PoolStorage(None),
        ),
        BufferDesign(
            "output",
            "Output",
            Footprint(width - 1, 0, 1, height),
            "system_output",
            PoolStorage(None),
        ),
    ]
    machines, vehicles, ports = [], [], []

    def port(owner, x, y, operation, suffix):
        ports.append(
            PortDesign(
                f"{owner}_{suffix}",
                f"{owner} {suffix}",
                Cell(x, y),
                bindings=(PortBinding(BufferTarget(owner), (operation,)),),
            )
        )

    for r in range(rows):
        port("input", 1, 3 + 4 * r, "pickup", str(r))
        port("output", width - 2, 3 + 4 * r, "drop_off", str(r))
        for c in range(columns):
            index, x, y = r * columns + c, 4 + 5 * c, 3 + 4 * r
            m = f"machine_{index:02d}"
            machines.append(
                MachineDesign(
                    m,
                    m,
                    Footprint(x, y, 1, 1),
                    modes,
                    operation_types=("ABCD"[(c + r) % 4],),
                )
            )
            vehicles.append(
                AGVDesign(f"agv_{index:02d}", f"AGV {index}", Cell(x, y - 2))
            )
            for side, dx, op in (("pre", -1, "drop_off"), ("post", 1, "pickup")):
                owner = f"{m}_{side}"
                buffers.append(
                    BufferDesign(
                        owner,
                        owner,
                        Footprint(x + dx, y, 1, 1),
                        f"machine_{side}",
                        PoolStorage(4),
                        m,
                    )
                )
                port(owner, x + dx, y - 1, op, "up")
                port(owner, x + dx, y + 1, op, "down")
                port(owner, x + 2 * dx, y, op, "side")
    return FactoryDesign(
        factory_id,
        f"Base {scale}",
        GridDesign(width, height),
        tuple(machines),
        tuple(buffers),
        ports=tuple(ports),
        agvs=tuple(vehicles),
        operation_types=tuple("ABCD"),
    )
