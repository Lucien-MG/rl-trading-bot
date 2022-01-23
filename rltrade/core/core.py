#!/usr/bin/python3
#‑∗‑ coding: utf‑8 ‑∗‑

import os
import gym

from core.utils import module

from core.cmd.cmd import run_cmd
from core.cmd.parser import argument_parser

from core.genv.run import RunEnvironment
from core.genv.train import TrainEnvironment
from core.genv.scenario import ScenarioEnvironment

from config.rltrade_config import RLtradeConfig
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


def train():
    # Get a default agent config
    agent_config = AgentConfig(RLTRADE_CONFIG.agent_config_path)

    # Get a default environement config
    if os.path.isdir(RLTRADE_CONFIG.environment_config_path):
        environment_config = [EnvironmentConfig(os.path.join(RLTRADE_CONFIG.environment_config_path, config_path)) for config_path in os.listdir(RLTRADE_CONFIG.environment_config_path)]
        # Build environment id and create an environement
        environment = [gym.make(env_config.environment_id, config=env_config) for env_config in environment_config]
    else:
        environment_config = EnvironmentConfig(RLTRADE_CONFIG.environment_config_path)
        # Build environment id and create an environement
        environment = gym.make(environment_config.environment_id, config=environment_config)

    # Load and build agent
    agent = load_agent(agent_config.name)(agent_config)

    # Create a logger instance
    logger = load_logger(RLTRADE_CONFIG.logger)()

    # Train the agent with the environment
    if os.path.isdir(RLTRADE_CONFIG.environment_config_path):
        renv = ScenarioEnvironment(environment, agent, logger=logger, logging_variables=RLTRADE_CONFIG.logging_variables)
    else:
        renv = TrainEnvironment(environment, agent, logger=logger, logging_variables=RLTRADE_CONFIG.logging_variables)

    res = renv.train(episodes=RLTRADE_CONFIG.train_episodes)

    return res


def run():
    # Get a default agent config
    agent_config = AgentConfig(RLTRADE_CONFIG.agent_config_path)

    # Get a default environement config
    environment_config = EnvironmentConfig(RLTRADE_CONFIG.environment_config_path)

    # Build environment id and create an environement
    environment = gym.make(environment_config.environment_id, config=environment_config)

    # Load and build agent
    agent = load_agent(agent_config.name)(agent_config)

    # Run the environment with the agent
    renv = RunEnvironment(environment, agent, render=environment_config.render)

    # Run the environment with the agent
    res = renv.episode()

    return res

def main():
    global RLTRADE_CONFIG

    # Parse command line arguments
    arguments = argument_parser()

    # Load configuration
    RLTRADE_CONFIG = RLtradeConfig(arguments.config) if arguments.config else RLtradeConfig()

    # Run the command
    cmd_code = run_cmd(arguments)

    return cmd_code
