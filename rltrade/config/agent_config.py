#!/usr/bin/python3
#‑∗‑ coding: utf‑8 ‑∗‑

import yaml

from dataclasses import dataclass
from config.agent_default import *

@dataclass
class AgentConfig:
    """ Describe the data that we want for the agent. """

    # Path to the agent config:
    agent_config_path: str = AGENT_CONFIG_PATH

    # Agent name:
    name: str = NAME

    # Define the number of action:
    action_space: int = ACTION_SPACE

    # Agent parameters
    alpha: float = ALPHA
    gamma: float = GAMMA
    epsilon: float = EPSILON
    min_epsilon: float = MIN_EPSILON
    epsilon_decay_factor: float = EPSILON_DECAY_FACTOR
    epsilon_update_step: int = EPSILON_UPDATE_STEP
    memory_size: int = MEMORY_SIZE
    batch_size: int = BATCH_SIZE
    network_update_step: int = NETWORK_UPDATE_STEP
    target_update_step: int = TARGET_UPDATE_STEP
    tau: float = TAU

    # Device use for calculation
    device: str = DEVICE

    def __post_init__(self):
        self.__load_config__(self.agent_config_path)

    def __load_config__(self, config_path):
        """ Load the configuration variables from the cli or the config file.
            If no configuration is found, use the default configuration.

        Attributes:
            config_path (Path): The path of the config file.
        """
        # Load config file if there is one:
        if not config_path:
            return

        try:
            with open(config_path, "r") as yaml_file:
                loaded_config = yaml.safe_load(yaml_file)
                loaded_config = {} if loaded_config is None else loaded_config
        except IOError:
            print("Agent config: Specified file not found, using default or cli values")
            self.config_path = None
            loaded_config = {}

        # Overwrite old arguments and add new ones:
        for key in loaded_config:
            setattr(self, key, loaded_config[key])
