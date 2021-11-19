#!/usr/bin/python3
#‑∗‑ coding: utf‑8 ‑∗‑

import yaml

def load_config(path):
    data_loaded = None

    with open(path, 'r') as stream:
        data_loaded = yaml.safe_load(stream)

    return data_loaded

def pretty_print_list(iterable):
    for e in iterable:
        print("  -", e)
