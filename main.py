import argparse
import os
import sys

from dataset import load_data

def argument_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('train_data')

    return parser.parse_args()


def main(args):
    load_data(args.train_data)

    pass


if __name__ == '__main__':
    if len(sys.argv) < 2:
        sys.argv = [
            __file__,
            os.path.expanduser('~/data/kaggle/titanic/train.csv')
        ]

    args = argument_parser()

    main(args)