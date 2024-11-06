"""
The functions and classes in this file are either copied or modified from the
repository of the paper: "LLMs learn governing principles of dynamical systems,
revealing an in-context neural scaling law" (https://arxiv.org/abs/2402.00795)
link of the github repo: https://github.com/AntonioLiu97/llmICL
By:
- Toni J.B. Liu, jl3499@cornell.edu
- Nicolas Boullé, nb690@cam.ac.uk
- Raphaël Sarfati, raphael.sarfati@cornell.edu
- Christopher J. Earls, earls@cornell.edu
"""


from functools import partial
import numpy as np
from dataclasses import dataclass
import matplotlib.pyplot as plt
import torch

# colors = {1: 'dodgerblue', 0.1: 'violet', 0.01: 'hotpink'}
colors = {1: "lightseagreen", 0.1: "dodgerblue", 0.01: "blue"}


# colors = {1: 'dodgerblue', 0.1: 'dodgerblue', 0.01: 'dodgerblue'}
def closest_color(width, colors):
    return colors[min(colors.keys(), key=lambda k: abs(k - width))]


class MultiResolutionPDF:
    """
    A class for managing and visualizing probability density functions (PDFs)
    in a multi-resolution format.

    This class allows for adding data in the form of bins, normalizing the bins,
    computing statistical properties (mean, mode, and standard deviation), plotting
    the PDF, and evaluating the PDF at a given point.

    Attributes:
        bin_center_arr (numpy.array): Stores the centers of the bins.
        bin_width_arr (numpy.array): Stores the widths of the bins.
        bin_height_arr (numpy.array): Stores the heights of the bins.
        mode (float): The mode of the PDF, computed in `compute_stats`.
        mean (float): The mean of the PDF, computed in `compute_stats`.
        sigma (float): The standard deviation of the PDF, computed in `compute_stats`.
    """

    def __init__(self):
        """
        Constructor for the MultiResolutionPDF class.

        Initializes arrays for bin centers, widths, and heights. Statistical properties
        (mode, mean, sigma) are initialized to None.
        """
        self.bin_center_arr = np.array([])
        self.bin_width_arr = np.array([])
        self.bin_height_arr = np.array([])
        self.mode = None
        self.mean = None
        self.sigma = None

    def add_bin(self, center_arr, width_arr, height_arr, idx=None):
        """
        Adds bins to the PDF.
        Do not normalize because PDF may need multiple add_bin operations

        Args:
            center_arr (array_like): Array or list of bin centers.
            width_arr (array_like): Array or list of bin widths.
            height_arr (array_like): Array or list of bin heights.

        Raises:
            AssertionError: If the lengths of center_arr, width_arr, and height_arr are
            not equal.
        """
        # print(len(center_arr))
        # print(len(width_arr))
        # print(len(height_arr))
        assert (
            len(center_arr) == len(width_arr) == len(height_arr)
        ), "center_arr, width_arr, height_arr must have the same length"
        if idx is None:  # insert to index position
            self.bin_center_arr = np.append(self.bin_center_arr, center_arr)
            self.bin_width_arr = np.append(self.bin_width_arr, width_arr)
            self.bin_height_arr = np.append(self.bin_height_arr, height_arr)
        else:  # append
            self.bin_center_arr = np.insert(self.bin_center_arr, idx, center_arr)
            self.bin_width_arr = np.insert(self.bin_width_arr, idx, width_arr)
            self.bin_height_arr = np.insert(self.bin_height_arr, idx, height_arr)
        # self.normalize()

    def sort_by_center(self):
        """
        Sorts the bins by their centers.
        """
        if not np.all(
            np.diff(self.bin_center_arr) >= 0
        ):  # check if bin_center_arr is already sorted
            sort_indices = np.argsort(self.bin_center_arr)  # sort by bin_center_arr
            self.bin_center_arr = self.bin_center_arr[sort_indices]
            self.bin_width_arr = self.bin_width_arr[sort_indices]
            self.bin_height_arr = self.bin_height_arr[sort_indices]

    def delete_by_idx(self, idx):
        """
        Deletes bins from the PDF by their indices.

        Args:
            idx (int or array_like): Index or list of indices of the bins to delete.
        """
        self.bin_center_arr = np.delete(self.bin_center_arr, idx)
        self.bin_width_arr = np.delete(self.bin_width_arr, idx)
        self.bin_height_arr = np.delete(self.bin_height_arr, idx)

    def refine(self, Multi_PDF):
        """
        Refines the PDF by merging it with another MultiResolutionPDF.
        Reduce to add_bin if self empty

        Args:
            Multi_PDF (MultiResolutionPDF): Another MultiResolutionPDF to merge with.
        """
        if len(self.bin_center_arr) == 0:
            self.add_bin(
                Multi_PDF.bin_center_arr,
                Multi_PDF.bin_width_arr,
                Multi_PDF.bin_height_arr,
            )
        else:
            Multi_PDF.normalize()
            assert isinstance(
                Multi_PDF, MultiResolutionPDF
            ), "Input must be an instance of MultiResolutionPDF"

            self.sort_by_center()
            right_edges = self.bin_center_arr + self.bin_width_arr / 2
            insert_index = np.searchsorted(right_edges, Multi_PDF.bin_center_arr.min())
            insert_index_right = np.searchsorted(
                right_edges, Multi_PDF.bin_center_arr.max()
            )
            # print(right_edges)
            # print(Multi_PDF.bin_center_arr)
            assert (
                insert_index == insert_index_right
            ), "refinement cannot straddle coarse bins"
            prefactor = (
                self.bin_width_arr[insert_index] * self.bin_height_arr[insert_index]
            )  # probability of coase bin to replace

            Multi_PDF.bin_height_arr *= prefactor
            self.delete_by_idx(insert_index)
            ### add bin, but to specific index
            self.add_bin(
                Multi_PDF.bin_center_arr,
                Multi_PDF.bin_width_arr,
                Multi_PDF.bin_height_arr,
                insert_index,
            )

            # print(self.bin_center_arr)
            assert np.all(
                np.diff(self.bin_center_arr) >= 0
            ), "final array should be sorted"
            self.check_gap_n_overlap()
            self.normalize()

    def coarsen(self, coarse_bin_centers, coarse_bin_widths):
        """
        Replace fine bins using coarse ones. This is for plotting purposes only.

        Args:
            coarse_bin_centers (np.ndarray): The centers of the coarse bins.
            coarse_bin_widths (np.ndarray): The widths of the coarse bins.
        """
        for coarse_bin_center, coarse_bin_width in zip(
            coarse_bin_centers, coarse_bin_widths
        ):
            # Find the indices of the fine bins that fall within the coarse bin
            indices = np.where(
                (self.bin_center_arr >= coarse_bin_center - coarse_bin_width / 2)
                & (self.bin_center_arr <= coarse_bin_center + coarse_bin_width / 2)
            )[0]

            if len(indices) == 0:
                continue

            # Compute the total height of the fine bins
            total_height = np.sum(
                self.bin_height_arr[indices] * self.bin_width_arr[indices]
            )

            # Replace the fine bins with the coarse bin
            self.bin_center_arr = np.delete(self.bin_center_arr, indices)
            self.bin_width_arr = np.delete(self.bin_width_arr, indices)
            self.bin_height_arr = np.delete(self.bin_height_arr, indices)

            self.bin_center_arr = np.append(self.bin_center_arr, coarse_bin_center)
            self.bin_width_arr = np.append(self.bin_width_arr, coarse_bin_width)
            self.bin_height_arr = np.append(
                self.bin_height_arr, total_height / coarse_bin_width
            )

        # Sort the bins by their centers
        sort_indices = np.argsort(self.bin_center_arr)
        self.bin_center_arr = self.bin_center_arr[sort_indices]
        self.bin_width_arr = self.bin_width_arr[sort_indices]
        self.bin_height_arr = self.bin_height_arr[sort_indices]

    def load_from_num_prob(self, num_slice, prob_slice):
        """
        Loads the PDF from a given number slice and probability slice.

        Args:
            num_slice (array_like): The number slice to load from.
            prob_slice (array_like): The probability slice to load from.
        """
        assert len(num_slice) == len(
            prob_slice
        ), "number of digits must equal number of probs"
        preceding_digits = None
        for idx, probs in enumerate(prob_slice):
            single_digit_PDF = MultiResolutionPDF()
            single_digit_PDF.load_from_prec_digits_prob(preceding_digits, probs)
            self.refine(single_digit_PDF)
            preceding_digits = num_slice[: idx + 1]

    def load_from_prec_digits_prob(self, preceding_digits, probs):
        """
        Loads the PDF from a given preceding digits and probabilities of the last digit.

        Args:
            preceding_digits (array_like): The preceding digits,
                which imply left_edge and bin_width
            probs (array_like): Distribution of next digit
        """
        assert len(probs.shape) == 1, "probs must be 1D"
        if preceding_digits is None:
            prec_len = 0
            w = 1
            left_edge = 0
        else:
            prec_len = len(preceding_digits)
            w = 0.1**prec_len
            left_edge = int(preceding_digits) * 10 * w
        x_coords = (
            np.linspace(left_edge, left_edge + 10 * w, 10, endpoint=False) + 0.5 * w
        )

        self.add_bin(center_arr=x_coords, width_arr=np.ones(10) * w, height_arr=probs)
        self.normalize()

    def normalize(self, report=False):
        """
        Normalizes the PDF so that the total area under the bins equals 1.
        Prints the total area before and after normalization.
        """
        total_area = np.sum(self.bin_width_arr * self.bin_height_arr)
        if total_area == 1.0:
            if report:
                print("already normalized")
        else:
            if report:
                print("total area before normalization:", total_area)
            self.bin_height_arr = self.bin_height_arr / total_area

    def compute_stats(self):
        """
        Computes and updates the statistical properties of the PDF: mean, mode, and
        standard deviation (sigma).
        """
        self.mean = np.sum(
            self.bin_center_arr * self.bin_width_arr * self.bin_height_arr
        )
        self.mode = self.bin_center_arr[np.argmax(self.bin_height_arr)]
        variance = np.sum(
            (self.bin_center_arr - self.mean) ** 2
            * self.bin_height_arr
            * self.bin_width_arr
        )
        self.sigma = np.sqrt(variance)

    def compute_moment(self, n):
        """
        Computes the nth mean-centered moment of the PDF.

        Args:
            n (int): The order of the moment to compute.

        Returns:
            float: The nth moment of the PDF.
        """
        if self.mean is None:
            self.compute_stats()
        return np.sum(
            (self.bin_center_arr - self.mean) ** n
            * self.bin_height_arr
            * self.bin_width_arr
        )

    def rescale_temperature(self, alpha):
        """
        Rescale bins as if the original temperature
        of softmax is scaled from T to alpha T
        """
        self.bin_height_arr = self.bin_height_arr ** (1 / alpha)
        self.normalize()

    def check_gap_n_overlap(self):
        assert np.allclose(
            self.bin_center_arr[1:] - self.bin_width_arr[1:] / 2,
            self.bin_center_arr[:-1] + self.bin_width_arr[:-1] / 2,
        ), "bin overlap detected"

    def discretize(self, func, mode="pdf"):
        """
        Args:
            func: a function supported on self.bin_center_arr.
                  should be implmented using numpy operations for parallelization
            mode: 'pdf': approximate probability of bin using its center
                  'cdf': integrate over bin
        Populate bin height by dicretizng
        """
        if mode == "pdf":
            self.bin_height_arr = func(self.bin_center_arr)
        elif mode == "cdf":
            right_edge = self.bin_center_arr + self.bin_width_arr / 2
            left_edge = self.bin_center_arr - self.bin_width_arr / 2
            prob_arr = func(right_edge) - func(left_edge)
            self.bin_height_arr = prob_arr / self.bin_width_arr
        self.normalize()

    def BT_dist(self, Multi_PDF):
        """
        Calculate the Bhattacharyya distance with another Multi_PDF object
        """
        assert np.all(
            self.bin_center_arr == Multi_PDF.bin_center_arr
        ), "Only PDFs of the same discretization are comparable"
        weighted_PQ_arr = (
            np.sqrt(self.bin_height_arr * Multi_PDF.bin_height_arr) * self.bin_width_arr
        )
        return -np.log(np.sum(weighted_PQ_arr))

    def KL_div(self, Multi_PDF):
        """
        Calculate the KL divergence D_KL(self||Multi_PDF)
        Prone to numerical instabilities
        """
        assert np.all(
            self.bin_center_arr == Multi_PDF.bin_center_arr
        ), "Only PDFs of the same discretization are comparable"
        log_ratio = np.log(self.bin_height_arr) - np.log(Multi_PDF.bin_height_arr)
        weighted_log_ratio = log_ratio * self.bin_height_arr * self.bin_width_arr
        return np.sum(weighted_log_ratio)

    def plot(self, ax=None, log_scale=False, statistic=True):
        """
        Plots the PDF as a bar chart.

        Args:
            ax (matplotlib.axes.Axes, optional): Matplotlib axis object. If None, a new
            figure and axis are created.
            log_scale (bool, optional): If True, sets the y-axis to logarithmic scale.
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(16, 4), dpi=100)

        # Iterate over bins and plot with corresponding color
        for center, width, height in zip(
            self.bin_center_arr, self.bin_width_arr, self.bin_height_arr
        ):
            color = closest_color(width, colors)
            ax.bar(center, height, width=width, align="center", color=color, alpha=1)

        if statistic:
            ax.vlines(
                self.mean,
                0,
                np.max(self.bin_height_arr),
                color="blue",
                label="Mean",
                lw=2,
            )
            ax.vlines(
                self.mode,
                0,
                np.max(self.bin_height_arr),
                color="lightblue",
                label="Mode",
                lw=2,
            )
            # Visualize sigma as horizontal lines
            ax.hlines(
                y=np.max(self.bin_height_arr),
                xmin=self.mean - self.sigma,
                xmax=self.mean + self.sigma,
                color="g",
                label="Sigma",
                lw=2,
            )

        if log_scale:
            ax.set_yscale("log")

        ax.legend()

        # If ax was None, show the plot
        if ax is None:
            plt.show()


def vec_num2repr(val, base, prec, max_val):
    """
    Convert numbers to a representation in a specified base with precision.

    Parameters:
    - val (np.array): The numbers to represent.
    - base (int): The base of the representation.
    - prec (int): The precision after the 'decimal' point in the base representation.
    - max_val (float): The maximum absolute value of the number.

    Returns:
    - tuple: Sign and digits in the specified base representation.

    Examples:
        With base=10, prec=2:
            0.5   ->    50
            3.52  ->   352
            12.5  ->  1250
    """
    base = float(base)
    sign = 1 * (val >= 0) - 1 * (val < 0)
    val = np.abs(val)
    max_bit_pos = int(np.ceil(np.log(max_val) / np.log(base)).item())

    before_decimals = []
    for i in range(max_bit_pos):
        digit = (val / base ** (max_bit_pos - i - 1)).astype(int)
        before_decimals.append(digit)
        val -= digit * base ** (max_bit_pos - i - 1)

    before_decimals = np.stack(before_decimals, axis=-1)

    if prec > 0:
        after_decimals = []
        for i in range(prec):
            digit = (val / base ** (-i - 1)).astype(int)
            after_decimals.append(digit)
            val -= digit * base ** (-i - 1)

        after_decimals = np.stack(after_decimals, axis=-1)
        digits = np.concatenate([before_decimals, after_decimals], axis=-1)
    else:
        digits = before_decimals
    return sign, digits


@dataclass
class SerializerSettings:
    """
    Settings for serialization of numbers.

    Attributes:
    - base (int): The base for number representation.
    - prec (int): The precision after the 'decimal' point in the base representation.
    - signed (bool): If True, allows negative numbers. Default is False.
    - fixed_length (bool): If True, ensures fixed length of serialized string. Default
        is False.
    - max_val (float): Maximum absolute value of number for serialization.
    - time_sep (str): Separator for different time steps.
    - bit_sep (str): Separator for individual digits.
    - plus_sign (str): String representation for positive sign.
    - minus_sign (str): String representation for negative sign.
    - half_bin_correction (bool): If True, applies half bin correction during
        deserialization. Default is True.
    - decimal_point (str): String representation for the decimal point.
    """

    base: int = 10
    prec: int = 3
    signed: bool = True
    fixed_length: bool = False
    max_val: float = 1e7
    time_sep: str = " ,"
    bit_sep: str = " "
    plus_sign: str = ""
    minus_sign: str = " -"
    half_bin_correction: bool = True
    decimal_point: str = ""
    missing_str: str = " Nan"


def serialize_arr(arr, settings: SerializerSettings):
    """
    Serialize an array of numbers (a time series) into a string based on the provided
    settings.

    Parameters:
    - arr (np.array): Array of numbers to serialize.
    - settings (SerializerSettings): Settings for serialization.

    Returns:
    - str: String representation of the array.
    """
    # max_val is only for fixing the number of bits in nunm2repr so it can be vmapped
    assert np.all(
        np.abs(arr[~np.isnan(arr)]) <= settings.max_val
    ), f"abs(arr) must be <= max_val,\
         but abs(arr)={np.abs(arr)}, max_val={settings.max_val}"

    if not settings.signed:
        assert np.all(arr[~np.isnan(arr)] >= 0), "unsigned arr must be >= 0"
        plus_sign = minus_sign = ""
    else:
        plus_sign = settings.plus_sign
        minus_sign = settings.minus_sign

    vnum2repr = partial(
        vec_num2repr, base=settings.base, prec=settings.prec, max_val=settings.max_val
    )
    sign_arr, digits_arr = vnum2repr(np.where(np.isnan(arr), np.zeros_like(arr), arr))
    ismissing = np.isnan(arr)

    def tokenize(arr):
        return "".join([settings.bit_sep + str(b) for b in arr])

    bit_strs = []
    for sign, digits, missing in zip(sign_arr, digits_arr, ismissing):
        if not settings.fixed_length:
            # remove leading zeros
            nonzero_indices = np.where(digits != 0)[0]
            if len(nonzero_indices) == 0:
                digits = np.array([0])
            else:
                digits = digits[nonzero_indices[0] :]
            # add a decimal point
            prec = settings.prec
            if len(settings.decimal_point):
                digits = np.concatenate(
                    [digits[:-prec], np.array([settings.decimal_point]), digits[-prec:]]
                )
        digits = tokenize(digits)
        sign_sep = plus_sign if sign == 1 else minus_sign
        if missing:
            bit_strs.append(settings.missing_str)
        else:
            bit_strs.append(sign_sep + digits)
    bit_str = settings.time_sep.join(bit_strs)
    bit_str += (
        settings.time_sep
    )  # otherwise there is ambiguity in number of digits in the last time step
    return bit_str


def calculate_multiPDF_llama3(
    full_series,
    model,
    tokenizer,
    n_states=1000,
    temperature=1.0,
    number_of_tokens_original=None,
    use_cache=False,
    kv_cache_prev=None,
):
    """
    This function calculates the multi-resolution probability density function (PDF)
        for a given series.

    Parameters:
    full_series (str): The series for which the PDF is to be calculated.
    prec (int): The precision of the PDF.
    mode (str, optional): The mode of calculation. Defaults to 'neighbor'.
    refine_depth (int, optional): The depth of refinement for the PDF. Defaults to 1.
    llama_size (str, optional): The size of the llama model. Defaults to '13b'.

    Returns:
    list: A list of PDFs for the series.
    """
    assert (
        n_states <= 1000
    ), f"if n_states ({n_states}) is larger than 1000, there will be more than 1 token"
    "per value!"

    good_tokens_str = []
    for num in range(n_states):
        good_tokens_str.append(str(num))
    good_tokens = [tokenizer.convert_tokens_to_ids(token) for token in good_tokens_str]

    batch = tokenizer([full_series], return_tensors="pt", add_special_tokens=True)

    torch.cuda.empty_cache()
    with torch.no_grad():
        out = model(
            batch["input_ids"].cuda(),
            use_cache=use_cache,
            past_key_values=kv_cache_prev,
        )

    logit_mat = out["logits"]

    kv_cache_main = out["past_key_values"] if use_cache else None
    logit_mat_good = logit_mat[:, :, good_tokens].clone()

    if number_of_tokens_original:
        probs = torch.nn.functional.softmax(
            logit_mat_good[:, -(number_of_tokens_original - 1) :, :] / temperature,
            dim=-1,
        )
    else:
        probs = torch.nn.functional.softmax(
            logit_mat_good[:, 1:, :] / temperature, dim=-1
        )

    PDF_list = []

    # start_loop_from = 1 if use_instruct else 0
    for i in range(1, int(probs.shape[1]), 2):
        PDF = MultiResolutionPDF()
        PDF.bin_center_arr = np.arange(0, 1000) / 100
        PDF.bin_width_arr = np.array(1000 * [0.01])
        PDF.bin_height_arr = probs[0, i, :].cpu().numpy() * 100
        PDF_list.append(PDF)

    # release memory
    del logit_mat, kv_cache_prev  # , kv_cache_main
    return PDF_list, probs, kv_cache_main
