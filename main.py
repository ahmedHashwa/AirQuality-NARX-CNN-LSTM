import os
import sys

# Suppress verbose TensorFlow logging and disable GPU usage before the
# library is imported anywhere else.
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

if sys.version_info < (3, 11) or sys.version_info >= (3, 12):
    raise RuntimeError("This project requires Python 3.11")

from airquality.config import CONFIG
from airquality.datasets import load_datasets
from airquality.training import train


def main():
    datasets_list = load_datasets(CONFIG.data_dir)
    train(datasets_list, CONFIG)


if __name__ == "__main__":
    main()
