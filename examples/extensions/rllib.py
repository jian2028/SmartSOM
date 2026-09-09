"""Central RLlib PPO using the same custom public-input encoder."""

import sys
from pathlib import Path

# Import the example by package name, also making it importable by worker processes.
sys.path[0] = str(Path(__file__).resolve().parents[1])
from extensions.common import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main("rllib.ppo"))
