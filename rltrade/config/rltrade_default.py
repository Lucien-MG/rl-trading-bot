from pathlib import Path
from dataclasses import field

# Path to the rltrade config:
RLTRADE_CONFIG_PATH: str = Path(__file__).absolute().parent / "../../rltrade_config.yaml"

# Data folder:
RLTRADE_DATA_PATH: str = (Path("~") / ".rltrade").expanduser()

# Agent configuration path:
AGENT_CONFIG_PATH: str = Path(__file__).absolute().parent / "../../agent_config.yaml"

# Agent configuration path:
ENVIRONMENT_CONFIG_PATH: str = Path(__file__).absolute().parent / "../../environment_config.yaml"

# Choose the logger to use:
LOGGER: str = "tensorboard_logger"

# Variables to log while training
LOGGING_VARIABLES: dict = field(default_factory=dict)

# Number of training episodes
TRAIN_EPISODES: int = 1000
