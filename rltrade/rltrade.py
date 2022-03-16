#!/usr/bin/python3
# ‑∗‑ coding: utf‑8 ‑∗‑

from run import run_cmd
from cmd import argument_parser

from config.rltrade.rltrade_config import RLtradeConfig


def main() -> int:
    # Parse command line arguments
    arguments = argument_parser()

    # Load configuration
    rltrade_config = RLtradeConfig()

    if arguments.config:
        rltrade_config = RLtradeConfig(arguments.config)

    # Print information
    print(rltrade_config.infos())

    # Run the command
    cmd_code = run_cmd(arguments, rltrade_config=rltrade_config)

    return cmd_code


if __name__ == '__main__':
    # Run the main program and return execution code
    exit(main())
