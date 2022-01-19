#!/usr/bin/python3
#‑∗‑ coding: utf‑8 ‑∗‑

from abc import ABC, abstractmethod

class StockExchangeStateInterface(ABC):
    
    @abstractmethod
    def step(self, action):
        pass

    @abstractmethod
    def reset(self):
        pass