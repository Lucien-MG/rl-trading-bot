#!/usr/bin/python3
#‑∗‑ coding: utf‑8 ‑∗‑

import gym
import gym_stock_exchange

import tools.module_loader as module_loader
import tools.utils as utils

from env.run_env import RunEnv
from env.train_env import TrainEnv


def list_agent() -> list:
    return module_loader.list_modules("agents")


def list_env() -> list:
    return module_loader.list_modules("gym_stock_exchange.envs")


def load_agent(agent_name):
    return module_loader.load_module("agents", agent_name).Agent


def train(env_name, agent_name, agent_config, nb_episode):
    env = gym.make("gym_stock_exchange:" + env_name + "-v0", stock_exchange_data_dir="data/")

    Agent = load_agent(agent_name)
    config = utils.load_config(agent_config)

    agent = Agent(config)

    renv = TrainEnv(env, agent, nb_episode=nb_episode, render=None)

    res = renv.train()

    return


def run(env_name, agent_name, agent_config):
    env = gym.make("gym_stock_exchange:" + env_name + "-v0", stock_exchange_data_dir="data/")

    Agent = load_agent(agent_name)
    config = utils.load_config(agent_config)

    agent = Agent(config)

    renv = RunEnv(env, agent, render=None)

    res = renv.episode()

    return
