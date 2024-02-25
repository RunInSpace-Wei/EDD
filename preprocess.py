from ast import literal_eval
from csv import reader
from os import listdir, makedirs, path
from pickle import dump
import numpy as np
import pandas as pd

from args import get_parser

def load_and_save(category, filename, dataset, dataset_folder, output_folder):
    temp = np.genfromtxt(
        path.join(dataset_folder, category, filename),
        dtype=np.float32,
        delimiter=",",
    )
    print(dataset, category, filename, temp.shape)
    with open(path.join(output_folder, dataset + "_" + category + ".pkl"), "wb") as file:
        dump(temp, file)


def load_data(dataset):
    """ Method from OmniAnomaly (https://github.com/NetManAIOps/OmniAnomaly) """

    if dataset == "SMAP" or dataset == "MSL":
        dataset_folder = "datasets/data"
        output_folder = "datasets/data/processed"
        makedirs(output_folder, exist_ok=True)
        with open(path.join(dataset_folder, "labeled_anomalies.csv"), "r") as file:
            csv_reader = reader(file, delimiter=",")
            res = [row for row in csv_reader][1:]
        res = sorted(res, key=lambda k: k[0])
        data_info = [row for row in res if row[1] == dataset and row[0] != "P-2"]
        labels = []
        for row in data_info:
            anomalies = literal_eval(row[2])
            length = int(row[-1])
            label = np.zeros([length], dtype=np.bool_)
            for anomaly in anomalies:
                label[anomaly[0] : anomaly[1] + 1] = True
            labels.extend(label)

        labels = np.asarray(labels)
        print(dataset, "test_label", labels.shape)

        with open(path.join(output_folder, dataset + "_" + "test_label" + ".pkl"), "wb") as file:
            dump(labels, file)

        def concatenate_and_save(category):
            data = []
            for row in data_info:
                filename = row[0]
                temp = np.load(path.join(dataset_folder, category, filename + ".npy"))
                data.extend(temp)
            data = np.asarray(data)
            print(dataset, category, data.shape)
            with open(path.join(output_folder, dataset + "_" + category + ".pkl"), "wb") as file:
                dump(data, file)

        for c in ["train", "test"]:
            concatenate_and_save(c)

    elif dataset.startswith("ETL"):
        dataset_folder = f"datasets/{dataset}"
        output_folder = f"datasets/{dataset}/processed"
        makedirs(output_folder, exist_ok=True)
        train = pd.read_csv(path.join(dataset_folder, 'train_data.csv'), index_col='time')
        test = pd.read_csv(path.join(dataset_folder, 'test_data_with_label.csv'), index_col='time')
        test_label = test['label']
        test.drop('label', axis=1, inplace=True)
        train, test, test_label = train.values.astype(float), test.values.astype(float), test_label.values.astype(int).reshape(-1, 1)
        for filename in ['train', 'test', 'test_label']:
            print(dataset, dataset + '_' + filename, eval(filename).shape)
            with open(path.join(output_folder, dataset + "_" + filename + ".pkl"), "wb") as file:
                dump(eval(filename), file)


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    args.__setattr__("dataset", "SMAP")
    ds = args.dataset
    load_data(ds)
