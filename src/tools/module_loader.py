#!/usr/bin/python3
#‑∗‑ coding: utf‑8 ‑∗‑

import os
import importlib

def load_module(package: str, module_name: str):
    # Build module path
    mod_path = package + "." + module_name

    # Import module
    module = importlib.import_module(mod_path)

    return module

def list_modules(package_name: str) -> list:
    # Use importlib to find agents module directory
    spec = importlib.util.find_spec(package_name)
    path = spec.submodule_search_locations[0]

    # List all mods
    mods = os.listdir(path)

    # Remove non-pertinent content
    for e in ['agent_interface.py', '__pycache__', '__init__.py', 'tools']:
        try:
            mods.remove(e)
        except:
            # In this case, nothing need to be remove
            pass

    # Remove extensions
    mods = map(lambda a: os.path.splitext(a)[0], mods)

    return mods
