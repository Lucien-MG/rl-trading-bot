from dataclasses import field

from definitions import *

# Path to the rltrade config:
RLTRADE_CONFIG_PATH: str = ROOT_CONFIG / "rltrade_config.yaml"

# Data folder:
RLTRADE_DATA_PATH: str = USER_HOME / ".rltrade"

# Mode:
COMMAND: str = "train"

# Agent configuration path:
AGENT_CONFIG_PATH: str = ROOT_CONFIG_AGENT / "agent_config.yaml"

# Agent configuration path:
ENVIRONMENT_CONFIG_PATH: str = ROOT_CONFIG_ENVIRONMENT / "environment_config.yaml"

# Choose the logger to use:
LOGGER: str = "tensorboard_logger"

# Variables to log while training
LOGGING_VARIABLES: dict = field(default_factory=dict)

# Number of training episodes
TRAIN_EPISODES: int = 1000
