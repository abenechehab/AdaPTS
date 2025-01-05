import os
import glob
from pathlib import Path

import pandas as pd
import numpy as np
import torch

from scipy import ndimage

from sklearn.preprocessing import LabelEncoder, StandardScaler

from dicl.utils import loading
from dicl.utils import preprocessing


def resize(image, size):
    zoom_factors = (size[0] / image.shape[0], size[1] / image.shape[1])
    return ndimage.zoom(image, zoom_factors)


def load_rl_data(env_name: str, data_label: str) -> np.ndarray:
    """
    Load the expert data for a given environment and data label.

    Args:
        env_name: The name of the environment.
        data_label: The label of the data to load.

    Returns:
        The expert data for the given environment and data label.
    """
    if env_name == "HalfCheetah":
        n_actions = 6  # number of actions in the HalfCheetah system
        n_observations = 17  # number of observations in the HalfCheetah system
    else:
        raise ValueError(f"Unknown environment: {env_name}")

    root_path = Path("/mnt/vdb/abenechehab/dicl-adapters/src/dicl/data/")
    data_path = root_path / f"D4RL_{env_name}_{data_label}.csv"
    X = pd.read_csv(data_path, index_col=0)
    X = X.values.astype("float")

    # find episodes beginnings. the restart column is equal to 1 at the start of
    # an episode, 0 otherwise.
    restart_index = n_observations + n_actions + 1
    restarts = X[:, restart_index]
    episode_starts = np.where(restarts)[0]

    # sample an episode and extract time series
    episode = np.random.choice(np.arange(len(episode_starts)))
    return X[
        episode_starts[episode] : episode_starts[episode]
        + episode_starts[episode + 1]
        - 1
    ]


class DataReader:
    def __init__(
        self,
        data_path="/data/bucket7893/dataset/",
        transform_ts_size=512,
        resize_func=resize,
        univariate=True,
    ):
        self.data_path = data_path
        self.transform_ts_size = transform_ts_size
        self.resize_func = resize_func
        self.univariate = univariate
        self._get_base_paths()
        self._get_dataset_lists()

    def _get_base_paths(
        self,
    ):
        self.base_path_ucr = self.data_path + "UCRArchive_2018/"
        os.makedirs(self.base_path_ucr, exist_ok=True)
        self.base_path_uea = self.data_path + "UEA/"
        os.makedirs(self.base_path_uea, exist_ok=True)
        self.base_path_others = self.data_path + "Others/"
        os.makedirs(self.base_path_others, exist_ok=True)
        self.base_path_forecasting = self.data_path + "forecasting/"
        os.makedirs(self.base_path_forecasting, exist_ok=True)

    def _get_dataset_lists(
        self,
    ):
        self.dataset_list_ucr = os.listdir(self.base_path_ucr)
        self.dataset_list_uea = os.listdir(self.base_path_uea)
        # self.dataset_list_uea = [dataset for dataset in self.dataset_list_uea if
        # dataset not in ["EigenWorms", "InsectWingbeat"]]
        self.dataset_list_others = os.listdir(self.base_path_others)
        self.dataset_list_forecasting = [
            "ETTh1",
            "ETTh2",
            "ETTm1",
            "ETTm2",
            "Electricity",
            "ExchangeRate",
            "Illness",
            "Traffic",
            "Weather",
        ]
        self.dataset_list_forecasting_additional = [
            "ForecastBaseline",
            "SPO-PPG",
            "EEG",
        ]

    def read_dataset(self, dataset_name, training_set=True, setting="train"):
        """
        dataset_name: if it's of the form "name:idx", then it will select only channel
        idx
        """
        if ":" in dataset_name:
            dataset_name, channel_idx = dataset_name.split(":")
            channel_idx = int(channel_idx)
        else:
            channel_idx = None

        if dataset_name in self.dataset_list_ucr:
            data = self._read_ucr_dataset(dataset_name, training_set=training_set)
        elif dataset_name in self.dataset_list_uea:
            if dataset_name == "InsectWingbeatSubset":
                data = self._read_insect_wingbeat_subset(
                    training_set=training_set, channel_idx=channel_idx
                )
            else:
                data = self._read_uea_dataset(
                    dataset_name, training_set=training_set, channel_idx=channel_idx
                )
        elif dataset_name in self.dataset_list_others:
            data = self._read_other_dataset(
                dataset_name, training_set=training_set, channel_idx=channel_idx
            )
        elif np.any(
            [
                dataset_name.startswith(ds_name)
                for ds_name in self.dataset_list_forecasting
            ]
        ):
            dataset_name_, pred_len, feature_idx, target_idx = (
                loading.process_forecasting_dataset_name(dataset_name)
            )
            data = self._read_forecasting_dataset(
                dataset_name_,
                setting=setting,
                seq_len=self.transform_ts_size,
                pred_len=pred_len,
                time_increment=1,
                feature_idx=feature_idx,
                target_idx=target_idx,
            )
        elif dataset_name == "ForecastBaseline":
            data = self._read_forecast_baseline_dataset()
        elif dataset_name == "SPO-PPG":
            data = self._read_spo_ppg_dataset()
        elif dataset_name == "EEG":
            data = self._read_eeg_dataset()
        elif dataset_name == "TCMNIST":
            data = self._read_tcmnist(
                training_set=training_set, channel_idx=channel_idx
            )
        else:
            raise KeyError("Unknown dataset name.")
        # encode labels to 0...K-1 if classification dataset
        x, y = data
        if (
            dataset_name
            in self.dataset_list_ucr + self.dataset_list_uea + self.dataset_list_others
        ):
            lab_encoder = LabelEncoder()
            y = lab_encoder.fit_transform(y)
        return x, y

    def _read_ucr_dataset(self, dataset_name, training_set=True):
        filename_suffix = "_TRAIN.tsv" if training_set else "_TEST.tsv"
        file_name = (
            self.base_path_ucr + dataset_name + "/" + dataset_name + filename_suffix
        )
        data = pd.read_csv(file_name, sep="\t", header=None).to_numpy()
        X, y = torch.tensor(data[:, 1:], dtype=torch.float), data[:, 0]
        X = X.unsqueeze(-2)
        X = self.resize_func(size=(1, self.transform_ts_size))(X)
        return X, y

    def _read_insect_wingbeat_subset(self, training_set=True, channel_idx=None):
        filename_suffix = "_train.npy" if training_set else "_test.npy"
        base_path = self.base_path_uea + "/InsectWingbeatSubset/"
        X, y = [np.load(base_path + s + filename_suffix) for s in ["x", "y"]]
        if self.univariate:
            if channel_idx is None:
                # so far, everything is univariate. So each row is (instance-channel)
                # entry from original 3d matrix
                # each entry y is repeated number of channels times
                y = np.repeat(y, X.shape[1])
                # flat the variable dimension
                X = torch.tensor(X.reshape(-1, X.shape[-1]), dtype=torch.float)
            else:
                X = torch.tensor(X[:, channel_idx, :], dtype=torch.float)
            # insert back a variable dimension for convention
            X = X.unsqueeze(-2)
            # interpolate time-series to self.transform_ts_size
            X = self.resize_func(size=(1, self.transform_ts_size))(X)
        else:
            X = torch.tensor(X, dtype=torch.float)
            # interpolate time-series to self.transform_ts_size
            X = self.resize_func(size=(X.shape[1], self.transform_ts_size))(X)
        return X, y

    def _read_tcmnist(self, training_set=True, channel_idx=None):
        filename_suffix = "_train.npy" if training_set else "_test.npy"
        base_path = self.data_path + "woods/TCMNIST/"
        X, y = [np.load(base_path + s + filename_suffix) for s in ["x", "y"]]
        # original shape of the matrix is (n_samples, seq_len, n_rows, n_cols)
        X = X.reshape(X.shape[0], X.shape[1], -1)
        X = np.swapaxes(X, 1, 2)
        if self.univariate:
            if channel_idx is None:
                # so far, everything is univariate. So each row is (instance-channel)
                # entry from original 3d matrix
                # each entry y is repeated number of channels times
                y = np.repeat(y, X.shape[1])
                # flat the variable dimension
                X = torch.tensor(X.reshape(-1, X.shape[-1]), dtype=torch.float)
            else:
                X = torch.tensor(X[:, channel_idx, :], dtype=torch.float)
            # insert back a variable dimension for convention
            X = X.unsqueeze(-2)
            # interpolate time-series to self.transform_ts_size
            X = self.resize_func(size=(1, self.transform_ts_size))(X)
        else:
            X = torch.tensor(X, dtype=torch.float)
            # interpolate time-series to self.transform_ts_size
            X = self.resize_func(size=(X.shape[1], self.transform_ts_size))(X)
        return X, y

    def _read_uea_dataset(self, dataset_name, training_set=True, channel_idx=None):
        """
        Reads UEA dataset. So far, it flattens the variable dimension considering each
        channel as a separate observation
        """
        filename_suffix = "_TRAIN.ts" if training_set else "_TEST.ts"
        file_name = (
            self.base_path_uea
            + "/"
            + dataset_name
            + "/"
            + dataset_name
            + filename_suffix
        )
        label_dict = loading.get_label_dict(file_name)
        X, y = loading.get_data_and_label_from_ts_file(file_name, label_dict)
        X = preprocessing.set_nan_to_zero(X)
        if self.univariate:
            if channel_idx is None:
                # so far, everything is univariate. So each row is (instance-channel)
                # entry from original 3d matrix
                # each entry y is repeated number of channels times
                y = np.repeat(y, X.shape[1])
                # flat the variable dimension
                X = torch.tensor(X.reshape(-1, X.shape[-1]), dtype=torch.float)
            else:
                X = torch.tensor(X[:, channel_idx, :], dtype=torch.float)
            # insert back a variable dimension for convention
            X = X.unsqueeze(-2)
            # interpolate time-series to self.transform_ts_size
            X = self.resize_func(size=(1, self.transform_ts_size))(X)
        else:
            X = torch.tensor(X, dtype=torch.float)
            # interpolate time-series to self.transform_ts_size
            X = self.resize_func(size=(X.shape[1], self.transform_ts_size))(X)
        return X, y

    def _read_other_dataset(self, dataset_name, training_set=True, channel_idx=None):
        filename_suffix = "/train.pt" if training_set else "/test.pt"
        filename = self.base_path_others + dataset_name + filename_suffix
        data = torch.load(filename)
        X, y = data["samples"], data["labels"]
        if type(X) is not np.ndarray:
            X = X.numpy()
        if self.univariate:
            if channel_idx is None:
                y = np.repeat(y, X.shape[1])
                X = torch.tensor(X.reshape(-1, X.shape[-1]), dtype=torch.float)
            else:
                X = torch.tensor(X[:, channel_idx, :], dtype=torch.float)
            X = X.unsqueeze(-2)
            # interpolate time-series to self.transform_ts_size
            X = self.resize_func(size=(1, self.transform_ts_size))(X)
        else:
            X = torch.tensor(X, dtype=torch.float)
            # interpolate time-series to self.transform_ts_size
            X = self.resize_func(size=(X.shape[1], self.transform_ts_size))(X)
        return X, y

    def _read_forecasting_dataset(
        self,
        dataset_name,
        setting,
        seq_len,
        pred_len,
        time_increment=1,
        feature_idx=None,
        target_idx=None,
    ):
        """
        NOW: val set is ignored, using old framework with training_set argument
        TODO:
        setting: "train", "test" or "test_with_val"
        if setting=="train", (pre-)training regime, so returns x_train, y_train
        if setting=="test", fine_tuning regime, so returns returns x_train, y_train,
            x_test, y_test
        if setting=="test_with_val", additionally provides validation set returns
            x_train, y_train, x_val, y_val, x_test, y_test
        """
        file_name = (
            self.base_path_forecasting + dataset_name + "/" + dataset_name + ".csv"
        )
        df_raw = pd.read_csv(file_name, index_col=0)
        # split train / valid / test
        n = len(df_raw)
        if "ETTm" in dataset_name:
            train_end = 12 * 30 * 24 * 4
            val_end = train_end + 4 * 30 * 24 * 4
            test_end = val_end + 4 * 30 * 24 * 4
        elif "ETTh" in dataset_name:
            train_end = 12 * 30 * 24
            val_end = train_end + 4 * 30 * 24
            test_end = val_end + 4 * 30 * 24
        else:
            train_end = int(n * 0.7)
            val_end = n - int(n * 0.2)
            test_end = n
        train_df = df_raw[:train_end]
        val_df = df_raw[train_end - seq_len : val_end]
        test_df = df_raw[val_end - seq_len : test_end]

        # standardize by training set
        # according to Romain, TSMixer does not scale back,
        # so MSE is reported on this scaled data
        scaler = StandardScaler()
        scaler.fit(train_df.values)

        train_df, val_df, test_df = [
            scaler.transform(df.values) for df in [train_df, val_df, test_df]
        ]

        if "train" in setting:
            x, y = loading.construct_sliding_window_data(
                train_df, seq_len, pred_len, time_increment, feature_idx, target_idx
            )
        elif "test" in setting:
            x, y = loading.construct_sliding_window_data(
                test_df, seq_len, pred_len, time_increment, feature_idx, target_idx
            )
        elif "val" in setting:
            x, y = loading.construct_sliding_window_data(
                val_df, seq_len, pred_len, time_increment, feature_idx, target_idx
            )
        else:
            raise ValueError(
                "Unknown setting. must contain either 'train', 'test' or 'val'"
            )

        # TODO: y now is not 3D, it's flatten
        return x, y  # .reshape((y.shape[0], y.shape[1] * y.shape[2]))

        # x_train, y_train = construct_sliding_window_data(train_df, seq_len, pred_len,
        #   time_increment, target_idx)
        # if setting == 'train':
        #     return x_train, y_train
        # elif 'test' in setting:
        #     x_test, y_test = construct_sliding_window_data(test_df, seq_len, pred_len,
        #   time_increment, target_idx)
        #     if setting == 'test_with_val':
        #         x_val, y_val = construct_sliding_window_data(val_df, seq_len,
        #   pred_len, time_increment, target_idx)
        #         return x_train, y_train, x_val, y_val, x_test, y_test
        #     else:
        #         return x_train, y_train, x_test, y_test

    def _read_forecast_baseline_dataset(self):
        dataset_list = glob.glob(
            self.data_path + "/forecast_baseline_dataset/ALL_data/*"
        )
        data = torch.tensor([], dtype=torch.float)
        for dataset in dataset_list[:]:
            if "txt" in dataset:
                continue
            data_file = glob.glob(dataset + "/*.csv")
            for file in data_file:
                ori_data = pd.read_csv(file)
                seg_length = ori_data.shape[0]
                while True:
                    if seg_length < 64:
                        break
                    tmp = (
                        ori_data.iloc[
                            : int(ori_data.shape[0] / seg_length) * seg_length, 1:
                        ]
                        .to_numpy()
                        .T
                    )
                    tmp = tmp.reshape(-1, seg_length)
                    tmp = self.resize_func(
                        size=(1, self.transform_ts_size), antialias=True
                    )(torch.tensor(tmp, dtype=torch.float).unsqueeze(-2))
                    data = torch.cat((data, tmp), dim=0)
                    seg_length = int(seg_length / 2)
        return data

    def _read_spo_ppg_dataset(self):
        dataset_list = glob.glob(self.data_path + "/ppg/osahs/*")
        data = torch.tensor([], dtype=torch.float)
        for dataset in dataset_list[:]:
            if "good" in dataset:
                continue
            tmp = torch.load(dataset).unsqueeze(dim=1)
            tmp = self.resize_func(size=(1, self.transform_ts_size), antialias=True)(
                torch.tensor(tmp.clone().detach(), dtype=torch.float)
            )
            data = torch.cat((data, tmp), dim=0)

        return data

    def _read_eeg_dataset(self):
        file_list = [
            "test_data_onemodelTrue_onlyiiFalse_interTrue.npy",
            "train_data_onemodelTrue_onlyiiFalse_interTrue.npy",
            "train_data_icen.npy",
        ]
        data = torch.tensor([], dtype=torch.float)
        for file in file_list[:]:
            tmp = self.resize_func(size=(1, self.transform_ts_size), antialias=True)(
                torch.tensor(
                    np.load(self.data_path + "/ECG/" + file, allow_pickle=True),
                    dtype=torch.float,
                ).unsqueeze(-2)
            )
            data = torch.cat((data, tmp), dim=0)

        return data

    def read_multiple_clf_datasets(self, dataset_subset=None, n_random_datasets=10):
        if dataset_subset is None:
            dataset_list = np.concatenate(
                [
                    self.dataset_list_ucr
                    + self.dataset_list_uea
                    + self.dataset_list_others
                ]
            )
            dataset_subset_ = np.random.choice(
                dataset_list, n_random_datasets, replace=False
            )
        else:
            dataset_subset_ = dataset_subset
        datasets = [
            self.read_dataset(ds_name, training_set=True) for ds_name in dataset_subset_
        ]
        x = torch.cat([dataset[0] for dataset in datasets], dim=0)
        y = torch.cat([dataset[1] for dataset in datasets], dim=0)
        return x, y

    def read_fm_dataset_1(self):
        """
        Train data inlcudes Others + 2 UEA datasets (Face-detection and PEMS-SF)
        """
        dataset_list = np.concatenate(
            [self.dataset_list_others, ["FaceDetection", "PEMS-SF"]]
        )
        datasets = [
            self.read_dataset(ds_name, training_set=True) for ds_name in dataset_list
        ]
        x = torch.cat([dataset[0] for dataset in datasets], dim=0)
        y = torch.cat([dataset[1] for dataset in datasets], dim=0)
        return x, y

    def read_multiple_collections(self, collection_list):
        """
        For ablation study of Songkang
        """
        dataset_list = []
        for collection in collection_list:
            if collection == "ucr":
                dataset_list.append(self.dataset_list_ucr)
            elif collection == "uea":
                dataset_list.append(self.dataset_list_uea)
            elif collection == "others":
                dataset_list.append(self.dataset_list_others)
            elif collection == "spo_ppg":
                dataset_list.append("SPO-PPG")
            elif collection == "eeg":
                dataset_list.append("EEG")
            elif collection == "forecast":
                dataset_list.append("ForecastBaseline")

        datasets = [
            self.read_dataset(ds_name, training_set=True) for ds_name in dataset_list
        ]
        x = torch.cat([dataset[0] for dataset in datasets], dim=0)
        y = torch.cat([dataset[1] for dataset in datasets], dim=0)
        return x, y
