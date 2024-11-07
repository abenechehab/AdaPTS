from typing import Optional, Union, Any
import numpy as np
from numpy.typing import NDArray

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.random_projection import SparseRandomProjection


class IdentityTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        """
        A custom scikit-learn transformer that performs no operation on the data.
        This is useful for the vICL variant of DICL.

        Methods:
            fit(
                input_array: NDArray, y: Optional[NDArray] = None
            ) -> IdentityTransformer:
                Fits the transformer (no-op in this case), returning the instance
                itself.

            transform(input_array: NDArray, y: Optional[NDArray] = None) -> NDArray:
                Returns the input data without modification.

            inverse_transform(
                input_array: NDArray, y: Optional[NDArray] = None) -> NDArray:
                    Returns the input data without modification.
        """
        pass

    def fit(self, input_array: NDArray, y: Optional[NDArray] = None):
        return self

    def transform(self, input_array: NDArray, y: Optional[NDArray] = None) -> NDArray:
        return input_array * 1

    def inverse_transform(
        self, input_array: NDArray, y: Optional[NDArray] = None
    ) -> NDArray:
        return input_array * 1


class MultichannelProjector:
    """
    A class used to project multichannel data using various dimensionality reduction
    techniques.
    Attributes
    ----------
    new_num_channels : int
        The number of channels after projection.
    patch_window_size : int, optional
        The size of the patch window. If None, defaults to 1.
    patch_window_size_ : int
        The effective patch window size used internally.
    base_projector : str or object, optional
        The base projector to use. Can be 'pca', 'svd', 'rand', or a custom projector
        object.
    base_projector_ : object
        The instantiated base projector used for dimensionality reduction.
    Methods
    -------
    fit(X)
        Fits the base projector to the input data X.
    transform(X)
        Transforms the input data X using the fitted base projector.
    """

    def __init__(
        self,
        num_channels: int,
        new_num_channels: int,
        patch_window_size: Optional[int] = None,
        base_projector: Optional[Union[str, Any]] = None,
    ):
        # init dimensions
        self.num_channels = num_channels
        self.new_num_channels = new_num_channels
        self.patch_window_size = patch_window_size
        self.patch_window_size_ = 1 if patch_window_size is None else patch_window_size
        # init base projector
        self.base_projector = base_projector
        n_components = self.patch_window_size_ * new_num_channels
        if base_projector is None:
            self.base_projector_ = IdentityTransformer()
        elif base_projector == "pca":
            self.base_projector_ = PCA(n_components=n_components)
        elif base_projector == "svd":
            self.base_projector_ = TruncatedSVD(n_components=n_components)
        elif base_projector == "rand":
            self.base_projector_ = SparseRandomProjection(n_components=n_components)
        # you can give your own base_projector with fit() and transform() methods, and
        # it should have the argument `n_components`.
        else:
            self.base_projector_ = base_projector(n_components=n_components)

    def fit(self, X, y: Optional[NDArray] = None):
        X_transposed = np.swapaxes(X, 1, 2)

        num_samples, seq_len, num_channels = X_transposed.shape
        assert num_channels == self.num_channels, "Number of channels must match."
        num_patches = seq_len // self.patch_window_size_
        assert num_patches * self.patch_window_size_ == seq_len

        X_2d = X_transposed.reshape(
            num_samples * num_patches, self.patch_window_size_ * num_channels
        )
        return self.base_projector_.fit(X_2d)

    def transform(self, X, y: Optional[NDArray] = None):
        """
        Apply the PCA transform on the input data.
        """
        X_transposed = np.swapaxes(X, 1, 2)

        num_samples, seq_len, num_channels = X_transposed.shape
        assert num_channels == self.num_channels, "Number of channels must match."
        num_patches = seq_len // self.patch_window_size_
        assert num_patches * self.patch_window_size_ == seq_len

        X_2d = X_transposed.reshape(
            num_samples * num_patches, self.patch_window_size_ * num_channels
        )

        X_transformed = self.base_projector_.transform(X_2d)
        X_transformed = X_transformed.reshape(
            [num_samples, seq_len, self.new_num_channels]
        )
        return np.swapaxes(X_transformed, 1, 2)

    def inverse_transform(self, X, y: Optional[NDArray] = None):
        """
        Apply the base_projector_ inverse transform on the input data.
        """
        X_transposed = np.swapaxes(X, 1, 2)

        num_samples, seq_len, num_channels = X_transposed.shape
        assert num_channels == self.new_num_channels, "Number of channels must match."
        num_patches = seq_len // self.patch_window_size_
        assert num_patches * self.patch_window_size_ == seq_len

        X_2d = X_transposed.reshape(
            num_samples * num_patches, self.patch_window_size_ * num_channels
        )

        X_inverse_transformed = self.base_projector_.inverse_transform(X_2d)
        X_inverse_transformed = X_inverse_transformed.reshape(
            [num_samples, seq_len, self.num_channels]
        )
        return np.swapaxes(X_inverse_transformed, 1, 2)
