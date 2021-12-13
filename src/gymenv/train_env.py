#!/usr/bin/python3
# ‑∗‑ coding: utf‑8 ‑∗‑

import tqdm

from collections import deque
from .run_env import RunEnv

class TrainEnv:
    """ Initialize the RunEnv class.
    Args:
        env: instance of OpenAI Gym's environment
        agent: agent that will interact with the environment.
    Attributes:
        env: instance of OpenAI Gym's environment
        agent: agent that will interact with the environment.
    """
    def __init__(self, env, agent, nb_episode=10, log_path=None, limit_step=None, reward_solved=None, render=None):
        self.env = env
        self.agent = agent

        self.nb_episode = nb_episode
        self.reward_solved = reward_solved

        self.log_path = log_path

        self.limit_step = limit_step
        self.render = render

        self.rewards = deque(maxlen=100)

    def train(self):
        for i in tqdm.tqdm(range(self.nb_episode)):
            self.rewards.append(sum(RunEnv(self.env, self.agent, self.log_path, limit_step=self.limit_step, render=self.render).episode()))

            mean_reward = sum(self.rewards) / len(self.rewards)

            if self.reward_solved and mean_reward > self.reward_solved:
                print("Environment Solved.")
                break
