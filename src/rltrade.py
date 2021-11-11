#!/usr/bin/python3
#‑∗‑ coding: utf‑8 ‑∗‑

import sys
import reinforcement_api

from app import run_app
from tools.cmd import argument_parser


def interactive_mod(args):
    run_app()


def cmd_mod(args):
    if args.list_agent:
        print("Available agents:")
        for a in reinforcement_api.list_agent():
            print("  -", a)
    elif args.list_env:
        print("Available envs:")
        for e in reinforcement_api.list_env():
            print("  -", e)
    elif args.train:
        print("Launch training:")
        reinforcement_api.train(args.env, args.agent, args.config, args.episode)
    else:
        reinforcement_api.run(args.env, args.agent, args.config)


if __name__ == '__main__':
    # Parse command line arguments
    args = argument_parser()

    # Choose program 
    if args.interactive:
        interactive_mod(args)
    else:
        cmd_mod(args)
