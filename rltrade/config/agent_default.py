#!/usr/bin/python3
#‑∗‑ coding: utf‑8 ‑∗‑

# Path to the environment config:
AGENT_CONFIG_PATH: str = None

# Choose the agent to use:
NAME: str = "dqn_v1"

# Define the number of action:
ACTION_SPACE: int = 3

# Agent parameters
ALPHA: float = 0.00001
GAMMA: float = 0.99
EPSILON: float = 1.0
MIN_EPSILON: float = 0.01
EPSILON_DECAY_FACTOR: float = 0.999
MEMORY_SIZE: int = 100000
BATCH_SIZE: int = 32
UPDATE_STEP: int = 1000
TAU: float = 0.001

# Device use for calculation
DEVICE: str = "cpu"
