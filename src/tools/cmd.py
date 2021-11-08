import argparse

def argument_parser():
    parser = argparse.ArgumentParser(description='Command line controler for rltrade.')

    parser.add_argument('-i', '--interactive', action='store_true', default=False,
                    help='Activate the interactive mode in your browser.')

    args = parser.parse_args()

    return args
