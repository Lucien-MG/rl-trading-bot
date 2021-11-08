import gym
import gym_stock_exchange

from env.runenv import RunEnv
from agents.dqn_v1 import Agent

dqn_config = {
    "alpha": 1e-2,
    "gamma": 0.99,
    "epsilon": 1.0,
    "min_epsilon": 1e-2,
    "epsilon_decay_factor": 0.999,
    "memory_size": 50000,
    "batch_size": 128,
    "update_step": 8,
    "tau": 1e-3, 
    "device": "cpu",
    "tensorboard_log": True
}

def run():
    env = gym.make("gym_stock_exchange:gym_simulation_relative_stock_exchange-v0", stock_exchange_data_dir="data/")
    agent = Agent(10 * 4 + 1, env.action_space.n, dqn_config)

    renv = RunEnv(env, agent, render=None)

    res = renv.episode()
