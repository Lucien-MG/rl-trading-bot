#!/usr/bin/python3
#‑∗‑ coding: utf‑8 ‑∗‑

from .agent_interface import AgentInterface

import random

class Agent(AgentInterface):

    def __init__(self, config):
        """Init the randim agent class
        Args:
            input_shape (file): the set up config file
        """
        self._set_config(config)

    def _set_config(self, configuration):
         """Read the config and put it in var
        Args:
            configuration (obj): the set up config file
        """
        self.__dict__ = { k:v for (k,v) in configuration.__dict__.items() }

    def action(self, state):
        """Decide an action
        Args:
            state (obj): current state of the agent
        """
        return random.randint(0, int(self.action_space) - 1)

    def step(self, state, action, reward, next_state, done):
        """Step of current state
        Args:
            state (obj): current state of the agent
            action (int): the choosen action
            reward (int): current reward
            next_state (obj): next state env
            done (bool): is training finished
        """
        pass
