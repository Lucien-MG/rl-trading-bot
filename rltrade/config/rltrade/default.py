from pathlib import Path
from dataclasses import field

from definitions import ROOT_DIRECTORY

# The root of the framework
FRAMEWORK_ROOT = Path(ROOT_DIRECTORY)

# Path to the rltrade config:
RLTRADE_CONFIG_PATH: str = FRAMEWORK_ROOT / "config/rltrade_config.yaml"

# Data folder:
RLTRADE_DATA_PATH: str = Path().home() / ".rltrade"

# Mode:
COMMAND: str = "train"

# Agent configuration path:
AGENT_CONFIG_PATH: str = FRAMEWORK_ROOT / "config/agents/agent_config.yaml"

# Agent configuration path:
ENVIRONMENT_CONFIG_PATH: str = FRAMEWORK_ROOT / "config/environments/environment_config.yaml"

# Choose the logger to use:
LOGGER: str = "tensorboard_logger"

# Variables to log while training
LOGGING_VARIABLES: dict = field(default_factory=dict)

# Number of training episodes
TRAIN_EPISODES: int = 1000
