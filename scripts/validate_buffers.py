"""Verify explicit buffer occupancy and recorded execution on current grid cases.

Absent PRE/POST facilities do not create implicit unlimited storage. Detailed
arbitration, capacity and composed-facility arithmetic live in the unit regressions.
"""

from validation.grid_cases import main

if __name__ == "__main__":
    raise SystemExit(
        main(
            (
                "buffers_direct_zero",
                "buffers_vehicle_zero",
                "buffers_post_one",
                "buffers_combined",
                "holding_hand",
            ),
            description=__doc__,
        )
    )
