#!/usr/bin/python3
#‑∗‑ coding: utf‑8 ‑∗‑

import os
import gym

import core.utils.module as module_loader

from core.genv.run_env import RunEnv
from core.genv.train_env import TrainEnv


def list_agent() -> list:
    return module_loader.list_modules("agents")


def list_env() -> list:
    return module_loader.list_modules("stock_exchange_engine.envs")

def list_data() -> list:
    datas = [data.split('.')[0] for data in os.listdir("data/")]
    return datas


def load_agent(agent_name: str):
    return module_loader.load_module("agents", agent_name).Agent


def train(environment_config: dict, agent_config: str, nb_episode: int, log_path: str):
    # Build environment id and create an environement
    environment = gym.make(environment_config.environment_id, config=environment_config)

    # Load and build agent
    Agent = load_agent(agent_config.name)
    agent = Agent(agent_config)

    renv = TrainEnv(environment, agent, nb_episode=nb_episode, log_path=log_path, render=None)

    res = renv.train()

    return res


def run(environment_config: dict, agent_config: dict):
    # Build environment id and create an environement
    environment = gym.make(environment_config.environment_id, config=environment_config)

    # Load and build agent
    Agent = load_agent(agent_config.name)
    agent = Agent(agent_config)

    # Build the environment with the agent
    renv = RunEnv(environment, agent, render=environment_config.render)

    # Run the environment with the agent
    res = renv.episode()

    return res
