#!/usr/bin/python3
#‑∗‑ coding: utf‑8 ‑∗‑

from .agent_interface import AgentInterface

import random

class AgentRandom(AgentInterface):

    def __init__(self, action_space):
        self.action_space = action_space
    
    def action(self, state):
        return random.randint(0, self.action_space - 1)

    def step(self):
        pass
