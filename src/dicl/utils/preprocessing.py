import numpy as np

from sklearn.base import TransformerMixin, BaseEstimator


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


class AxisScaler(TransformerMixin, BaseEstimator):
    def __init__(self, scaler, axis=1):
        self.scaler = scaler
        self.axis = axis

    def fit(self, X, y=None):
        # Reshape to 2D for scaling
        shape = X.shape
        if len(shape) == 3:
            # Reshape depending on axis
            if self.axis == 0:
                X_2d = X.reshape(-1, shape[1] * shape[2])
            elif self.axis == 1:
                X_2d = X.transpose(0, 2, 1).reshape(-1, shape[1])
            else:  # axis == 2
                X_2d = X.reshape(-1, shape[2])
        else:
            X_2d = X

        self.scaler.fit(X_2d)
        return self

    def transform(self, X):
        shape = X.shape
        if len(shape) == 3:
            if self.axis == 0:
                X_2d = X.reshape(-1, shape[1] * shape[2])
                X_scaled = self.scaler.transform(X_2d)
                return X_scaled.reshape(shape)
            elif self.axis == 1:
                X_2d = X.transpose(0, 2, 1).reshape(-1, shape[1])
                X_scaled = self.scaler.transform(X_2d)
                return X_scaled.reshape(shape[0], shape[2], shape[1]).transpose(0, 2, 1)
            else:  # axis == 2
                X_2d = X.reshape(-1, shape[2])
                X_scaled = self.scaler.transform(X_2d)
                return X_scaled.reshape(shape)
        return self.scaler.transform(X)

    def inverse_transform(self, X):
        shape = X.shape
        if len(shape) == 3:
            if self.axis == 0:
                X_2d = X.reshape(-1, shape[1] * shape[2])
                X_scaled = self.scaler.inverse_transform(X_2d)
                return X_scaled.reshape(shape)
            elif self.axis == 1:
                X_2d = X.transpose(0, 2, 1).reshape(-1, shape[1])
                X_scaled = self.scaler.inverse_transform(X_2d)
                return X_scaled.reshape(shape[0], shape[2], shape[1]).transpose(0, 2, 1)
            else:  # axis == 2
                X_2d = X.reshape(-1, shape[2])
                X_scaled = self.scaler.inverse_transform(X_2d)
                return X_scaled.reshape(shape)
        return self.scaler.inverse_transform(X)
