#!/usr/bin/python3
#‑∗‑ coding: utf‑8 ‑∗‑

import enum

class Actions(enum.Enum):
    """ Possible action for stock exchange bot. """
    Skip = 0
    Buy = 1
    Close = 2
