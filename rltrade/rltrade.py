#!/usr/bin/python3
#‑∗‑ coding: utf‑8 ‑∗‑

from core.cmd.parser import argument_parser
from core.cmd.cmd import run_cmd

def interactive_mod(args):
    from app import run_app
    run_app()


def main() -> int:
    # Parse command line arguments
    args = argument_parser()
 
    # Choose and run program mod
    if args.interactive:
        interactive_mod(args)
    else:
        run_cmd(args)

    return 0


if __name__ == '__main__':
    # Run the main program and return execution code
    exit(main())