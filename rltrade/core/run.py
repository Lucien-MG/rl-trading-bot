#!/usr/bin/python3
#‑∗‑ coding: utf‑8 ‑∗‑

import sys

from core import core
from core.utils.utils import pretty_print_list

def cmd_list_agent(**kwargs) -> None:
    print("Available agents:")
    agents = core.list_agent()

    pretty_print_list(agents)


def cmd_list_env(**kwargs) -> None:
    print("Available environments:")
    envs = core.list_env()

    pretty_print_list(envs)


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
