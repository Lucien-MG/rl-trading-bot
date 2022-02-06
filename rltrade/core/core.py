#!/usr/bin/python3
#‑∗‑ coding: utf‑8 ‑∗‑

import os
import gym

from core.utils import module

from core.cmd.parser import argument_parser

from core.genv.run import RunEnvironment
from core.genv.train import TrainEnvironment

from core.config.rltrade_config import RLtradeConfig
from config.agent_config import AgentConfig
from config.environment_config import EnvironmentConfig

def list_agent() -> list:
    return module.list_modules("agents")


def list_env() -> list:
    return module.list_modules("stock_exchange_engine.envs")


def list_data() -> list:
    datas = [data.split('.')[0] for data in os.listdir("data/")]
    return datas


def load_agent(agent_name: str):
    return module.load_module("agents", agent_name).Agent


def load_logger(logger_name: str):
    return module.load_module("core.log", logger_name).Logger


def load_environment(rltrade_config):
    environment_config = [EnvironmentConfig()]

    if os.path.isdir(rltrade_config.environment_config_path):
        environment_config = [EnvironmentConfig(os.path.join(rltrade_config.environment_config_path, config_path)) for config_path in os.listdir(rltrade_config.environment_config_path)]
    else:
        environment_config = [EnvironmentConfig(rltrade_config.environment_config_path)]

    environment = [gym.make(env_config.environment_id, config=env_config) for env_config in environment_config]

    return environment


def train(rltrade_config):
    # Get a default agent config
    agent_config = AgentConfig(rltrade_config.agent_config_path)

    # Load environments:
    environment = load_environment(rltrade_config)

    # Load and build agent
    agent = load_agent(agent_config.name)(agent_config)

    # Create a logger instance
    logger = load_logger(rltrade_config.logger)()

    # Train the agent with the environment
    renv = TrainEnvironment(environment, agent, logger=logger, logging_variables=rltrade_config.logging_variables)

    res = renv.train(episodes=rltrade_config.train_episodes)

    return res


def run(rtrade_config):
    # Get a default agent config
    agent_config = AgentConfig(rltrade_confg.agent_config_path)

    environment = load_environment(rltrade_config)[0]

    # Load and build agent
    agent = load_agent(agent_config.name)(agent_config)

    # Run the environment with the agent
    renv = RunEnvironment(environment, agent)

    # Run the environment with the agent
    res = renv.episode()

    return res


