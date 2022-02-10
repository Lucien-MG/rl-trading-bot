#!/usr/bin/python3
# ‑∗‑ coding: utf‑8 ‑∗‑

import gym
import random
from tqdm import tqdm
from collections import deque

from agents.agent_interface import AgentInterface
from core.genv.run import RunEnvironment
from core.log.log_interface import LogInterface

class TrainEnvironment:
    """ Initialize the RunEnvironment class.
    Args:
        env: instance of OpenAI Gym's environment
        agent: agent that will interact with the environment.
    Attributes:
        env: instance of OpenAI Gym's environment
        agent: agent that will interact with the environment.
    """
    def __init__(self, env, agent, solved=None, logger=None, logging_variables=None):
        self.envs = env
        self.agent = agent
        self.solved = solved

        self.logger = logger if logger else LogInterface()
        self.logging_variables = logging_variables

        self.step = 0

        self.environment = None
        self.rewards = deque(maxlen=100)

        assert isinstance(self.envs, list)
        assert isinstance(self.agent, AgentInterface)
        assert isinstance(self.logger, LogInterface)

    def __mean__(self, list):
        return sum(list) / len(list)

    def __log__(self):
        self.logger.log("reward", self.__mean__(self.rewards), self.step)

        for element in self.logging_variables:
            for attributes in self.logging_variables[element]:
                if isinstance(self.__dict__[element], RunEnvironment):
                    self.logger.log(attributes, self.__dict__[element].env.env.__dict__[attributes], self.step)
                else:
                    self.logger.log(attributes, self.__dict__[element].__dict__[attributes], self.step)

    def train(self, episodes=100):
        environment_runners = [RunEnvironment(env, self.agent, render=False) for env in self.envs]

        self.logger.start()

        for _ in tqdm(range(episodes)):
            self.environment = random.choice(environment_runners)

            episode_rewards = self.environment.episode()

            self.step += len(episode_rewards)

            episode_reward_indicator = sum(episode_rewards) / len(episode_rewards)

            self.rewards.append(episode_reward_indicator)

            self.__log__()

            if self.solved and self.__mean__(self.rewards) > self.solved:
                print("Environment Solved.")
                break

        self.logger.stop()

        return list(self.rewards)
