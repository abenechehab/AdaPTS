import numpy as np

from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.decomposition import PCA

import torch
import torch.nn as nn


def set_nan_to_zero(a):
    where_are_NaNs = np.isnan(a)
    a[where_are_NaNs] = 0
    return a


def fill_out_with_Nan(data, max_length):
    """
    TODO: pad_length cannot be negative? maybe better to call enlarge_dim_by_padding
    """
    # via this it can works on more dimensional array
    pad_length = max_length - data.shape[-1]
    if pad_length == 0:
        return data
    else:
        pad_shape = list(data.shape[:-1])
        pad_shape.append(pad_length)
        Nan_pad = np.empty(pad_shape) * np.nan
        return np.concatenate((data, Nan_pad), axis=-1)


class AxisPCA(TransformerMixin, BaseEstimator):
    def __init__(self, n_components=None, axis=1):
        self.n_components = n_components
        self.axis = axis
        self.pca = PCA(n_components=n_components)

    def fit(self, X, y=None):
        # Reshape to 2D for scaling
        shape = X.shape
        assert len(shape) == 3 and self.axis == 1, "Only support 3D input with axis=1"

        X_2d = X.transpose(0, 2, 1).reshape(-1, shape[1])

        self.pca.fit(X_2d)
        return self

    def transform(self, X):
        shape = X.shape
        assert len(shape) == 3 and self.axis == 1, "Only support 3D input with axis=1"

        X_2d = X.transpose(0, 2, 1).reshape(-1, shape[1])
        X_scaled = self.pca.transform(X_2d)
        return X_scaled.reshape(shape[0], shape[2], shape[1]).transpose(0, 2, 1)

    def inverse_transform(self, X):
        shape = X.shape
        assert len(shape) == 3 and self.axis == 1, "Only support 3D input with axis=1"

        X_2d = X.transpose(0, 2, 1).reshape(-1, shape[1])
        X_scaled = self.pca.inverse_transform(X_2d)
        return X_scaled.reshape(shape[0], shape[2], shape[1]).transpose(0, 2, 1)


class AxisScaler(TransformerMixin, BaseEstimator):
    def __init__(self, scaler, axis=1):
        self.scaler = scaler
        self.axis = axis

    def fit(self, X, y=None):
        # Reshape to 2D for scaling
        shape = X.shape
        assert len(shape) == 3 and self.axis == 1, "Only support 3D input with axis=1"

        X_2d = X.transpose(0, 2, 1).reshape(-1, shape[1])

        self.scaler.fit(X_2d)
        return self

    def transform(self, X):
        shape = X.shape
        assert len(shape) == 3 and self.axis == 1, "Only support 3D input with axis=1"

        X_2d = X.transpose(0, 2, 1).reshape(-1, shape[1])
        X_scaled = self.scaler.transform(X_2d)
        return X_scaled.reshape(shape[0], shape[2], shape[1]).transpose(0, 2, 1)

    def inverse_transform(self, X):
        shape = X.shape
        assert len(shape) == 3 and self.axis == 1, "Only support 3D input with axis=1"

        X_2d = X.transpose(0, 2, 1).reshape(-1, shape[1])
        X_scaled = self.scaler.inverse_transform(X_2d)
        return X_scaled.reshape(shape[0], shape[2], shape[1]).transpose(0, 2, 1)


def get_gpu_memory_stats():
    """
    Get GPU memory statistics:
    - Allocated: Memory actually used by tensors
    - Reserved: Memory managed by caching allocator
    - Total: Total GPU memory
    - Percentages of usage
    """
    if not torch.cuda.is_available():
        return {}

    stats = {}
    for i in range(torch.cuda.device_count()):
        # Get memory in MB
        allocated = torch.cuda.memory_allocated(i) / 1024**2
        reserved = torch.cuda.memory_reserved(i) / 1024**2
        total = torch.cuda.get_device_properties(i).total_memory / 1024**2

        # Calculate percentages
        allocated_percent = (allocated / total) * 100
        reserved_percent = (reserved / total) * 100

        stats.update(
            {
                f"gpu_{i}_allocated(%)": allocated_percent,
                f"gpu_{i}_reserved(%)": reserved_percent,
            }
        )
    return stats


class RevIN(nn.Module):
    def __init__(self, num_features: int, eps=1e-5, affine=True):
        """
        :param num_features: the number of features or channels
        :param eps: a value added for numerical stability
        :param affine: if True, RevIN has learnable affine parameters
        """
        super(RevIN, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        if self.affine:
            self._init_params()

    def forward(self, x, mode: str):
        if mode == "norm":
            self._get_statistics(x)
            x = self._normalize(x)
        elif mode == "denorm":
            x = self._denormalize(x)
        else:
            raise NotImplementedError
        return x

    def _init_params(self):
        # initialize RevIN params: (C,)
        self.affine_weight = nn.Parameter(torch.ones(self.num_features))
        self.affine_bias = nn.Parameter(torch.zeros(self.num_features))

    def _get_statistics(self, x):
        dim2reduce = tuple(range(1, x.ndim - 1))
        self.mean = torch.mean(x, dim=dim2reduce, keepdim=True).detach()
        self.stdev = torch.sqrt(
            torch.var(x, dim=dim2reduce, keepdim=True, unbiased=False) + self.eps
        ).detach()

    def _normalize(self, x):
        x = x - self.mean
        x = x / self.stdev
        if self.affine:
            x = x * self.affine_weight
            x = x + self.affine_bias
        return x

    def _denormalize(self, x):
        if self.affine:
            x = x - self.affine_bias
            x = x / (self.affine_weight + self.eps * self.eps)
        x = x * self.stdev
        x = x + self.mean
        return x


# Rest of the code remains same, just using the enhanced get_gpu_memory_stats()
