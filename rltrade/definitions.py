#!/usr/bin/python3
# ‑∗‑ coding: utf‑8 ‑∗‑

from pathlib import Path

# Root path
ROOT_DIRECTORY = Path(__file__).resolve().parent.parent

ROOT_CONFIG = ROOT_DIRECTORY / "config"
ROOT_CONFIG_AGENT = ROOT_CONFIG / "agents"
ROOT_CONFIG_ENVIRONMENT = ROOT_CONFIG / "environments"

# Src path
SRC_DIRECTORY = Path(__file__).resolve().parent

# App Directory
APP_DIRECTORY = SRC_DIRECTORY / "app"
TEMPLATE_DIRECTORY = APP_DIRECTORY / "templates"
STATIC_DIRECTORY = APP_DIRECTORY / "static"


# Home
USER_HOME = Path().home()
