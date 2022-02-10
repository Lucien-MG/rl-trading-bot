#!/usr/bin/python3
# ‑∗‑ coding: utf‑8 ‑∗‑

from pathlib import Path

ROOT_DIRECTORY = Path(__file__).resolve().parent.parent

ROOT_CONFIG = ROOT_DIRECTORY / "config"
ROOT_CONFIG_AGENT = ROOT_CONFIG / "agents"
ROOT_CONFIG_ENVIRONMENT = ROOT_CONFIG / "environments"

USER_HOME = Path().home()
