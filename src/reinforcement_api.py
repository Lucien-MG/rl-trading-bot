#!/usr/bin/python3
#‑∗‑ coding: utf‑8 ‑∗‑

import os

import gym

import tools.module_loader as module_loader
import tools.utils as utils

from gymenv.run_env import RunEnv
from gymenv.train_env import TrainEnv


def list_agent() -> list:
    return module_loader.list_modules("agents")


def list_env() -> list:
    return module_loader.list_modules("gym_stock_exchange.envs")

def list_data() -> list:
    datas = [data.split('.')[0] for data in os.listdir("data/")]
    return datas


def load_agent(agent_name: str):
    return module_loader.load_module("agents", agent_name).Agent


def train(env_name: str, agent_name: str, agent_config: str, nb_episode: int, log_path: str):
    env = gym.make("gym_stock_exchange:" + env_name + "-v0", stock_exchange_data="data/cac40.csv")

    Agent = load_agent(agent_name)
    config = utils.load_config(agent_config)

    utils.pretty_print_dic(config, 1)

    agent = Agent(config)
    print(nb_episode)
    renv = TrainEnv(env, agent, nb_episode=nb_episode, log_path=log_path, render=None)

    res = renv.train()

    return


def run(environment_config: dict, agent_name: str, agent_config: dict):

    # Build environment id and create an environement:
    environment_id = "gym_stock_exchange:" + environment_config.name + "-v0"
    environment = gym.make(environment_id, config=environment_config)

    Agent = load_agent(agent_name)
    config = utils.load_config(agent_config)

    agent = Agent(config)

    renv = RunEnv(environment, agent, render=environment_config.render)

    res = renv.episode()

    return res
