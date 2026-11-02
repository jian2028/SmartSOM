"""Load the canonical Base designs; geometry lives only in the map resources."""

from smartsom.config.map_catalog import load_map


def base_factory(scale):
    return load_map(f"{scale}-factory").factory
