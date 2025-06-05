import sys
if sys.version_info < (3, 11) or sys.version_info >= (3, 12):
    raise RuntimeError("This project requires Python 3.11")

from config import CONFIG
from datasets import load_datasets
from training import train


def main():
    datasets_list = load_datasets(CONFIG.data_dir)
    train(datasets_list, CONFIG)


if __name__ == "__main__":
    main()
