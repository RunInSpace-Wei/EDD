import os

from utils import get_data


if __name__ == "__main__":
    f = open('../datasets/data/processed/MSL_train.pkl', "rb")
    (train_data, _), (test_data, test_label) = get_data("MSL", normalize=True)
