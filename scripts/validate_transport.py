"""Verify explicit grid transport and recorded execution on current configured cases.

Historical whole-trip matrix makespans remain tied to their original source.
"""

from validation.grid_cases import main

if __name__ == "__main__":
    raise SystemExit(
        main(
            ("transport_hand", "transport_reroute", "transport_combined"),
            description=__doc__,
        )
    )
