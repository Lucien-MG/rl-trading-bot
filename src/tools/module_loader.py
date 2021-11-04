#!/usr/bin/python3
#‑∗‑ coding: utf‑8 ‑∗‑

import os

def load_agent(name):
    agent_lib = importlib.import_module("agent." + name)
    agent_class = agent_lib.Agent

    return agent_class

def list_modules(path):
    content = os.listdir(path)
    print(content)

    return content
