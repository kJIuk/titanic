import sys

from src.train import main as train
from src.demo import main as demo


class Main:
    train = train
    demo = demo


if __name__ == '__main__':
    if len(sys.argv) < 2:
        sys.argv = [
            __file__,
            'train'
        ]

    getattr(Main, sys.argv[1])()
