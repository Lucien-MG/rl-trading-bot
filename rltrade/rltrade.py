#!/usr/bin/python3
#‑∗‑ coding: utf‑8 ‑∗‑

from core.cmd.cmd import run_cmd
from core.cmd.parser import argument_parser

from core.config.rltrade_config import RLtradeConfig

def main() -> int:
    print("########## RLtrade ##########\n")

    # Parse command line arguments
    arguments = argument_parser()

    # Load configuration
    rltrade_config = RLtradeConfig(arguments.config) if arguments.config else RLtradeConfig()

    # Print information
    print(rltrade_config.infos())

    # Run the command
    cmd_code = run_cmd(arguments)
    return cmd_code


if __name__ == '__main__':
    # Run the main program and return execution code
    exit(main())
