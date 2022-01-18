#!/usr/bin/python3
#‑∗‑ coding: utf‑8 ‑∗‑

from tools.cmd import argument_parser

def interactive_mod(args):
    from app import run_app
    run_app()


def cmd_mod(args):
    from cmd import run_cmd
    run_cmd(args)


def main() -> int:
    # Parse command line arguments
    args = argument_parser()
 
    # Choose and run program mod
    if args.interactive:
        interactive_mod(args)
    else:
        cmd_mod(args)

    return 0


if __name__ == '__main__':
    # Run the main program and return execution code
    exit(main())
