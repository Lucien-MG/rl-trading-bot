#!/usr/bin/python3
# ‑∗‑ coding: utf‑8 ‑∗‑

import sys
import app

from core import core



def cmd_application(**kwargs) -> None:
    # Launch application:
    app.launch(kwargs['rltrade_config'])


def cmd_list_agent(**kwargs) -> None:
    print("Available agents:")
    agents = core.list_agent()

    for agent in agents:
        print("  -", agent)


def cmd_list_env(**kwargs) -> None:
    print("Available environments:")
    envs = core.list_env()

    for env in envs:
        print("  -", env)


def cmd_train(**kwargs) -> None:
    # Launch a training on the environement:
    core.train(kwargs['rltrade_config'])


def cmd_run(**kwargs) -> None:
    # Launch a run of the environement:
    core.run(kwargs['rltrade_config'])


def run_cmd(arguments, **kwargs):
    command_module = sys.modules[__name__]
    command_module_functions = dir(command_module)

    for arg in arguments.__dict__:
        func_name = "cmd_" + arg
        if arguments.__dict__[arg] is True and func_name in command_module_functions:
            cmd_func = getattr(command_module, func_name)
            cmd_func(**kwargs)
