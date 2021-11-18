#!/usr/bin/python3
#‑∗‑ coding: utf‑8 ‑∗‑

import argparse

def argument_parser():
    parser = argparse.ArgumentParser(description='Command line controler for rltrade.')

    parser.add_argument('-i', '--interactive', action='store_true', default=False,
                    help='Activate the interactive mode in your browser. All other arguments are ignored.')

    parser.add_argument('-t', '--train', action='store_true', default=False,
                    help='Launch training.')

    parser.add_argument('-r', '--run', action='store_true', default=False,
                    help='Launch a run.')

    parser.add_argument('-la', '--list-agent', action='store_true', default=False,
                    help='If run in cmd mod, list all available agents.')

    parser.add_argument('-le', '--list-env', action='store_true', default=False,
                    help='If run in cmd mod, list all available environment.')

    parser.add_argument('-e', '--env', type=str, default="gym_simulation_relative_stock_exchange",
                    help='If run in cmd mod, choose the environment to use.')

    parser.add_argument('-a', '--agent', type=str, default="random",
                    help='If run in cmd mod, choose the agent to use.')

    parser.add_argument('-c', '--config', type=str, default="./config.yaml",
                    help='If run in cmd mod, choose the config to use.')

    parser.add_argument('-l', '--load', type=str, default="./weights.pt",
                    help='If run in cmd mod, load weights for the agent.')

    parser.add_argument('-ep', '--episode', type=int, default=100,
                    help='If run in cmd mod, choose the number of episodes for training.')

    args = parser.parse_args()

    return args
