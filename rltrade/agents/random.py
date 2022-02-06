#!/usr/bin/python3
#‑∗‑ coding: utf‑8 ‑∗‑

from .agent_interface import AgentInterface

import random

class Agent(AgentInterface):

    def __init__(self, config):
        self._set_config(config)

    def _set_config(self, configuration):
        self.__dict__ = { k:v for (k,v) in configuration.__dict__.items() }

    def action(self, state):
        return random.randint(0, int(self.action_space) - 1)

    def step(self, state, action, reward, next_state, done):
        pass
