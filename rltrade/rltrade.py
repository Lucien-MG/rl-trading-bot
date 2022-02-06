#!/usr/bin/python3
#‑∗‑ coding: utf‑8 ‑∗‑

from core import core

def main() -> int:
    # Run the core main
    exit_code = core.main()
    return exit_code


if __name__ == '__main__':
    # Run the main program and return execution code
    exit(main())
