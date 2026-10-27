"""Integer-time reconstruction from actual sparse Base movement records."""

from smartsom.domain.production import MOVES


def positions_at(factory, records, tick):
    if type(tick) is not int or tick < 0:
        raise ValueError("tick must be a nonnegative integer")
    positions = {a.agv_id: (a.initial_cell.x, a.initial_cell.y) for a in factory.agvs}
    current = 0
    if tick == 0:
        return positions
    for record in records:
        start, end = record["start"], record["end"]
        if start != current or type(end) is not int or end <= start:
            raise ValueError("noncontiguous or invalid sparse execution interval")
        if set(record["moves"]) != set(positions):
            raise ValueError("sparse trace must cover every vehicle")
        if end - start > 1 and any(a != "WAIT" for a in record["moves"].values()):
            raise ValueError("only waiting intervals may span several ticks")
        if tick < end:
            return positions
        for v, action in record["moves"].items():
            if action not in (*MOVES, "WAIT"):
                raise ValueError("unknown recorded movement")
            dx, dy = MOVES.get(action, (0, 0))
            x, y = positions[v]
            positions[v] = (x + dx, y + dy)
        if len(set(positions.values())) != len(positions):
            raise ValueError("recorded vertex conflict")
        current = end
        if tick == current:
            return positions
    raise ValueError("requested tick was not recorded")
