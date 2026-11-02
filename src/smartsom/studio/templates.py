"""Load the four bundled factories as new documents."""

from smartsom.config.map_catalog import MAP_NAMES, load_map

BUILTIN_TEMPLATES = MAP_NAMES


def load_template_file(name):
    return load_map(name)
