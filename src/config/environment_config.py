#!/usr/bin/python3
#‑∗‑ coding: utf‑8 ‑∗‑

import yaml

from dataclasses import dataclass
from config.environment_default import *

@dataclass
class EnvironmentConfig:
    """ Describe the data that we want for our environment. """

    # Path to the environment config:
    environment_config_path: str = ENVIRONMENT_CONFIG_PATH

    # Choose the environement to use:
    name: str = NAME
    simulation: bool = SIMULATION

    # Choose the source where data come from ands the index to use:
    source: str = SOURCE
    index: str = INDEX
    data_path: str = DATA_PATH

    # Choose the time that will be use in the environment
    start_date: str = START_DATE
    end_date: str = END_DATE

    # Data encoding
    data_encoding: str = DATA_ENCODING

    # Choose the time that will be use in the environment
    render: bool = RENDER

    def __post_init__(self):
        self.__load_config__(self.environment_config_path)

    def __load_config__(self, config_path):
        """ Load the configuration variables from the cli or the config file.
            If no configuration is found, use the default configuration.

        Attributes:
            config_path (Path): The path of the config file.
        """
        # Load config file if there is one:
        try:
            if config_path:
                with open(config_path, "r") as yaml_file:
                    loaded_config = yaml.safe_load(yaml_file)
                    loaded_config = {} if loaded_config is None else loaded_config
        except IOError:
            print("Config: Specified file not found, using default or cli values\n")
            self.config_path = None
            loaded_config = {}

        # Overwrite old arguments and add new ones:
        for key in loaded_config:
            setattr(self, key, loaded_config[key])
