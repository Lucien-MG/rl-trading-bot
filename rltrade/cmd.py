#!/usr/bin/python3
#‑∗‑ coding: utf‑8 ‑∗‑

import sys
import reinforcement_api

from tools.utils import pretty_print_list

from config.agent_config import AgentConfig
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

    # Get a default agent config
    agent_config = AgentConfig("/Users/lulu/Worskpace/rl-trading-bot/agent_config.yaml")

    # Get a default environement config
    environment_config = EnvironmentConfig("/Users/lulu/Worskpace/rl-trading-bot/environement_config.yaml")

    reinforcement_api.train(environment_config, agent_config, args.episode, None)


def cmd_run(args) -> None:
    print("Launch run on", args.env)

    # Get a default agent config
    agent_config = AgentConfig("/Users/lulu/Worskpace/rl-trading-bot/agent_config.yaml")

    # Get a default environement config
    environment_config = EnvironmentConfig("/Users/lulu/Worskpace/rl-trading-bot/environement_config.yaml")

    # Launch on run of the environement:
    reinforcement_api.run(environment_config, agent_config)


def run_cmd(args):
    command_module = sys.modules[__name__]
    command_module_functions = dir(command_module)

    for arg in args.__dict__:
        func_name = "cmd_" + arg
        if args.__dict__[arg] is True and func_name in command_module_functions:
            cmd_func = getattr(command_module, func_name)
            cmd_func(args)
