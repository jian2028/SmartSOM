"""SB3 MaskablePPO with a real custom encoder and separate actor/critic."""

import sys
from pathlib import Path

# Keep resource.py from shadowing Python's standard-library resource module.
sys.path[0] = str(Path(__file__).resolve().parents[1])
from extensions.common import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main("sb3.maskable_ppo"))
