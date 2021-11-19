#!/usr/bin/python3
#‑∗‑ coding: utf‑8 ‑∗‑

import reinforcement_api

from app import run_app
from tools.cmd import argument_parser
from tools.utils import pretty_print_list

def interactive_mod(args):
    run_app()


def cmd_mod(args):
    if args.list_agent:
        print("Available agents:")
        pretty_print_list(reinforcement_api.list_agent())
    elif args.list_env:
        print("Available envs:")
        pretty_print_list(reinforcement_api.list_env())
    elif args.train:
        print("Launch training:")
        reinforcement_api.train(args.env, args.agent, args.config, args.episode)
    elif args.run:
        print("Launch run:")
        reinforcement_api.run(args.env, args.agent, args.config)


if __name__ == '__main__':
    # Parse command line arguments
    args = argument_parser()

    # Choose program mod
    if args.interactive:
        interactive_mod(args)
    else:
        cmd_mod(args)
