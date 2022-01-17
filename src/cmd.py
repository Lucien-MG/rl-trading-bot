#!/usr/bin/python3
#‑∗‑ coding: utf‑8 ‑∗‑

import sys
import reinforcement_api

from tools.utils import pretty_print_list
from config.environment_config import EnvironmentConfig


def cmd_list_agent(args) -> None:
    print("Available agents:")
    agents = reinforcement_api.list_agent()

    pretty_print_list(agents)


def cmd_list_env(args) -> None:
    print("Available environments:")
    envs = reinforcement_api.list_env()

    pretty_print_list(envs)


def cmd_train(args) -> None:
    print("Launch training on", args.env)
    reinforcement_api.train(args.env, args.agent, args.config, args.episode, None)


def cmd_run(args) -> None:
    print("Launch run on", args.env)

    # Create a default config:
    environment_config = EnvironmentConfig("/Users/lulu/Worskpace/rl-trading-bot/environement_config.yaml")

    # Get agent class compatible with gym:
    agent_class = args.agent

    # Get agent config:
    agent_config = args.config

    # Launch on run of the encironement:
    reinforcement_api.run(environment_config, agent_class, agent_config)


def run_cmd(args):
    command_module = sys.modules[__name__]
    command_module_functions = dir(command_module)

    for arg in args.__dict__:
        func_name = "cmd_" + arg
        if args.__dict__[arg] is True and func_name in command_module_functions:
            cmd_func = getattr(command_module, func_name)
            cmd_func(args)
