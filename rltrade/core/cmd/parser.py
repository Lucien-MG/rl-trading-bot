#!/usr/bin/python3
#‑∗‑ coding: utf‑8 ‑∗‑

import argparse

DEFAULT_CMD = False
DEFAULT_CONFIG = None

def commands(parser):
    parser.add_argument('-t', '--train', action='store_true', default=DEFAULT_CMD,
                    help='launch a training session')

    parser.add_argument('-r', '--run', action='store_true', default=DEFAULT_CMD,
                    help='launch a run session')

    parser.add_argument('-la', '--list-agent', action='store_true', default=DEFAULT_CMD,
                    help='list all available agents')

    parser.add_argument('-le', '--list-env', action='store_true', default=DEFAULT_CMD,
                    help='list all available environment.')

    parser.add_argument('-i', '--interactive', action='store_true', default=DEFAULT_CMD,
                    help='lauch a interactive session') 

def configuration(parser):
    parser.add_argument('-c', '--config', type=str, default=DEFAULT_CONFIG,
                    help='choose the config to use.')

    parser.add_argument('-e', '--env', type=str, default=DEFAULT_CONFIG,
                    help='select the environment to use')

    parser.add_argument('-a', '--agent', type=str, default=DEFAULT_CONFIG,
                    help='choose the agent to use')

    parser.add_argument('-l', '--load', type=str, default=DEFAULT_CONFIG,
                    help='load weights for the agent')

    parser.add_argument('-ep', '--episodes', type=int, default=DEFAULT_CONFIG,
                    help='choose the number of episodes for training')

def argument_parser():
    parser = argparse.ArgumentParser(description='Command line interface for rltrade.')

    commands(parser)

    configuration(parser)

    args = parser.parse_args()

    return args
