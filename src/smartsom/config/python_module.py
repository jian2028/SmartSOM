"""Explicit, hashed Python module loading outside simulation decision loops."""

import hashlib
import importlib.util
import sys
from pathlib import Path


def load_module(spec):
    path = Path(spec.path)
    raw = path.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    if digest != spec.sha256:
        raise ValueError(f"extension source mismatch: {path}")
    name = "smartsom_extension_" + digest
    if name not in sys.modules:
        module_spec = importlib.util.spec_from_file_location(name, path)
        if module_spec is None or module_spec.loader is None:
            raise ValueError(f"cannot load extension: {path}")
        module = importlib.util.module_from_spec(module_spec)
        sys.modules[name] = module
        try:
            module_spec.loader.exec_module(module)
        except BaseException:
            sys.modules.pop(name, None)
            raise
    return sys.modules[name]
