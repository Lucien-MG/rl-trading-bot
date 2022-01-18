#!/usr/bin/python3
#‑∗‑ coding: utf‑8 ‑∗‑

from abc import ABC, abstractmethod

class AgentInterface(ABC):
    
    @abstractmethod
    def action(self, state):
        pass

    @abstractmethod
    def step(self, state, action, reward, next_state, done):
        pass
