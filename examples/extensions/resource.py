"""Resource RLlib PPO with different machine/AGV branches and role rewards."""

import sys
from pathlib import Path

# This script must not shadow Python's standard-library resource module.
sys.path[0] = str(Path(__file__).resolve().parents[1])
from extensions.common import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main("rllib.resource_ppo"))
