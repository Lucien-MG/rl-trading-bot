#!/usr/bin/python3
#‑∗‑ coding: utf‑8 ‑∗‑

from abc import ABC

class LogInterface(ABC):

    def start(self):
        pass

    def stop(self):
        pass
    
    def log(self, name, value, step):
        pass
