#!/usr/bin/python3
#‑∗‑ coding: utf‑8 ‑∗‑

import yaml

from dataclasses import dataclass
from core.config.rltrade_default import *

@dataclass
class RLtradeConfig:
    """ Describe the data that we want for our environment. """

    # # The root of the framework
    framework_root: str = FRAMEWORK_ROOT

    # Path to the rltrade config:
    rltrade_config_path: str = RLTRADE_CONFIG_PATH

    # Data folder:
    rltrade_data_path: str = RLTRADE_DATA_PATH

    # Command to execute:
    command: str = COMMAND

    # Agent configuration path:
    agent_config_path: str = AGENT_CONFIG_PATH

    # Agent configuration path:
    environment_config_path: str = ENVIRONMENT_CONFIG_PATH

    # Choose the logger to use:
    logger: str = LOGGER

    # Variables to log while training
    logging_variables: dict = LOGGING_VARIABLES

    # Number of training episodes
    train_episodes: int = TRAIN_EPISODES

    def __post_init__(self):
        self.__load_config__()

    def __load_config__(self):
        """ Load the configuration variables from the cli or the config file.
            If no configuration is found, use the default configuration.

        Attributes:
            config_path (Path): The path of the config file.
        """
        # Load config file if there is one:
        try:
            if not self.rltrade_config_path:
                print("RLtrade config: value is None, using default or cli values\n")
            else:
                with open(self.rltrade_config_path, "r") as yaml_file:
                    loaded_config = yaml.safe_load(yaml_file)
                    loaded_config = {} if loaded_config is None else loaded_config

                # Overwrite old arguments and add new ones:
                for key in loaded_config:
                    setattr(self, key, loaded_config[key])
        except IOError:
            print("RLtrade config: Specified file not found, using default or cli values\n")

    def infos(self):
        return (
            f"Root path: {self.framework_root}\n"
            f"Config path: {self.rltrade_config_path}\n"
            f"Data folder path: {self.rltrade_data_path}\n"
            f"\n"
            f"Mode: {self.command}\n"
            f"Logger: {self.logger}\n"
            f"\n"
            f"Agent config path: {self.agent_config_path}\n"
            f"Environment config path: {self.environment_config_path}"
        )

