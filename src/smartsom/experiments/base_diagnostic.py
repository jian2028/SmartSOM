"""Compatibility entry point for frozen Base diagnostic launch commands.

New experiments use smartsom check/run; both routes execute the same runtime.
"""

from smartsom.experiments.reservation_runtime import (
    CONDITIONS as CONDITIONS,
)
from smartsom.experiments.reservation_runtime import (
    calibrate as calibrate,
)
from smartsom.experiments.reservation_runtime import (
    episode as episode,
)
from smartsom.experiments.reservation_runtime import (
    evaluate as evaluate,
)
from smartsom.experiments.reservation_runtime import (
    freeze as freeze,
)
from smartsom.experiments.reservation_runtime import (
    frozen_drain as frozen_drain,
)
from smartsom.experiments.reservation_runtime import (
    identity as identity,
)
from smartsom.experiments.reservation_runtime import (
    main as main,
)
from smartsom.experiments.reservation_runtime import (
    metrics as metrics,
)
from smartsom.experiments.reservation_runtime import (
    source_identity as source_identity,
)
from smartsom.experiments.reservation_runtime import (
    train as train,
)
from smartsom.experiments.reservation_runtime import (
    write_json as write_json,
)

if __name__ == "__main__":
    main()
