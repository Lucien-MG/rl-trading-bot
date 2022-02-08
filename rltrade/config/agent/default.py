#!/usr/bin/python3
#‑∗‑ coding: utf‑8 ‑∗‑

# Path to the environment config:
AGENT_CONFIG_PATH: str = None

# Choose the agent to use:
NAME: str = "dqn"

# Define the number of action:
ACTION_SPACE: int = 3

# Agent parameters

# Alpha, equivalent of learning rate
ALPHA: float = 0.0001

# Propagation rate of the reward
GAMMA: float = 0.99

# Epsilon start value
EPSILON: float = 1.0

# Minimal value for epsilon
MIN_EPSILON: float = 0.1

# Epsilon decay factor
EPSILON_DECAY_FACTOR: float = 0.9995

# Epsilon update step
EPSILON_UPDATE_STEP: int = 250

# Number of steps keep in memory
MEMORY_SIZE: int = 150000

# Batch size
BATCH_SIZE: int = 32

# Network update step
NETWORK_UPDATE_STEP: int = 2

# Double target network
TARGET_UPDATE_STEP: int = 1000
TAU: float = 0.001

# Device use for calculation
DEVICE: str = "cpu"
