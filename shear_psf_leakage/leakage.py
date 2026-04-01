"""LEAKAGE.

:Name: leakage.py

:Description: This package contains methods to deal with
    leakage.

:Authors: Martin Kilbinger <martin.kilbinger@cea.fr>
        Clara Bonini
        Axel Guinot

"""

import os
import pickle
import re

import matplotlib.pylab as plt
import numpy as np
from astropy.io import fits
from cs_util import args as cs_args
from lmfit import Parameters, minimize
from uncertainties import ufloat

from .plot_style import *


# MKDEBUG TODO: to cs_util (and see sp_validation/io.py)
def open_stats_file(directory, file_name):
    """Open statistics file.

    Open output file for statistics

    Parameters
    ----------
    directory : string
        directory
    file_name : string
        file name

    """
    stats_file = open("{}/{}".format(directory, file_name), "w")

    return stats_file


def print_stats(msg, stats_file, verbose=False):
    """Print stats.

    Print message to stats file.

    Parameters
    ----------
    msg : string
        message
    stats_file : file handler
        statistics output file
    verbose : bool, optional, default=False
        print message to stdout if True
    """
    stats_file.write(msg)
    stats_file.write("\n")
    stats_file.flush()

    if verbose:
        print(msg)


def open_fits_or_npy(path, hdu_no=1, verbose=False):
    """Open FITS OR NPY.

    Open FITS or numpy binary file.

    Parameters
    ----------
    path : str
        path to input binary file
    hdu_no : int, optional
        HDU number, default is 1
    verbose : bool, optional
        verbose output if ``True``; default is ``False``

    Raises
    ------
    ValueError
        if file extension not valid, i.e. neither ``.fits`` nor ``.npy``

    Returns
    -------
    FITS.rec or numpy.ndarray
        data

    """
    filename, file_extension = os.path.splitext(path)
    if file_extension in [".fits", ".cat"]:
        hdu_list = fits.open(path)
        data = hdu_list[hdu_no].data
    elif file_extension == ".npy":
        data = np.load(path)
    else:
        raise ValueError(f"Invalid file extension '{file_extension}'")

    if verbose:
        print(f"{len(data)} objects found in {file_extension} file")

    return data


def cut_data(data, cut, verbose=False):
    """Cut Data.

    Cut data according to selection criteria list.

    Parameters
    ----------
    data : numpy,ndarray
        input data
    cut : str
        selection criteria expressions, white-space separated
    verbose : bool, optional
        verbose output if `True`, default is `False`

    Raises
    ------
    ValueError :
        if cut expression is not valid

    Returns
    -------
    numpy.ndarray
        data after cuts

    """
    if cut is None:
        if verbose:
            print("No cuts applied to input galaxy catalogue")

        return data

    cut_list = cut.split(" ")

    for cut in cut_list:
        res = re.match(r"(\w+)([<>=!]+)(\w+)", cut)
        if res is None:
            raise ValueError(f"cut '{cut}' has incorrect syntax")
        if len(res.groups()) != 3:
            raise ValueError(
                f"cut criterium '{cut}' does not match syntax "
                "'field rel val'"
            )
        field, rel, val = res.groups()

        cond = "data['{}']{}{}".format(field, rel, val)

        if verbose:
            print(f"Applying cut '{cond}' to input galaxy catalogue")

        data = data[np.where(eval(cond))]

    if verbose:
        print(f"Using {len(data)} galaxies after cuts.")

    return data


def func_bias_2d_full(params, x1, x2, order="lin", mix=False):
    """Func Bias 2D Full.

    Function of 2D bias model evaluated on full 2D grid.

    Parameters
    ----------
    params : lmfit.Parameters
        fit parameters
    x1 : list
        first component of x-values, float
    x2 : list
        second component of x-values, float
    order : str, optional
        order of fit, default is 'lin'
    mix : bool, optional
        mixing between components, default is `False`

    Returns
    -------
    np.array
        first component the 2D model y1(x1, x2) on the (x1, x2)-grid;
        2D array of float
    np.array
        second component the 2D model, y2(x1, x2) on the (x1, x2)-grid;
        2D array of float

    """
    len1 = len(x1)
    len2 = len(x2)

    # Initialise both components y1, y2 as 2D arrays
    y1 = np.zeros(shape=(len1, len2))
    y2 = np.zeros(shape=(len1, len2))

    # Create 2D mesh for input x1, x2 values
    v1, v2 = np.meshgrid(x1, x2, indexing="ij")

    # Compute both components y1, y2 over the meash
    y1, y2 = func_bias_2d(params, v1, v2, order=order, mix=mix)

    return y1, y2


def func_bias_2d(params, x1_data, x2_data, order="lin", mix=False):
    """Func Bias 2D.

    Function of 2D bias model.

    Parameters
    ----------
    params : lmfit.Parameters
        fit parameters
    x1_data : float or list of float
        first component of x-values of the data
    x2_data : float or list of float
        second component of x-values of the data
    order : str, optional
        order of fit, default is 'lin'
    mix : bool, optional
        mixing between components, default is `False`

    Returns
    -------
    list
        first component the 2D model, y1(x1, x2). Dimension
        is equal to x1_data and x2_data
    list
        second component the 2D model, y2(x1, x2). Dimension
        is equal to x1_data and x2_data

    """
    # Get affine parameters
    a11 = params["a11"].value
    a22 = params["a22"].value
    c1 = params["c1"].value
    c2 = params["c2"].value

    # Compute y-values for affine model
    y1_model = a11 * x1_data + c1
    y2_model = a22 * x2_data + c2

    if order == "quad":
        # Add quadratic part
        q111 = params["q111"].value
        q222 = params["q222"].value
        y1_model += q111 * x1_data**2
        y2_model += q222 * x2_data**2

    if mix:
        # Add linear mixing part
        a12 = params["a12"].value
        a21 = params["a21"].value
        y1_model += a12 * x2_data
        y2_model += a21 * x1_data

        if order == "quad":
            # Add quadratic mixing part
            q112 = params["q112"].value
            q122 = params["q122"].value
            q212 = params["q212"].value
            q211 = params["q211"].value
            y1_model += q112 * x1_data * x2_data + q122 * x2_data**2
            y2_model += q212 * x1_data * x2_data + q211 * x1_data**2

    return y1_model, y2_model


def weighted_mean_std(data, weights):
    """Weighted Mean and Standard Error.

    Fast computation of weighted mean and standard error of the mean.

    Parameters
    ----------
    data : array
        input sample
    weights : array
        weights

    Returns
    -------
    float
        weighted mean
    float
        weighted standard error of the mean

    """
    w_sum = np.sum(weights)
    if w_sum == 0:
        return np.nan, np.nan

    mean = np.average(data, weights=weights)

    # Weighted variance
    variance = np.average((data - mean) ** 2, weights=weights)

    # Standard error of the mean (using effective sample size)
    n_eff = w_sum ** 2 / np.sum(weights ** 2)
    std_err = np.sqrt(variance / n_eff)

    return mean, std_err


def jackknife_mean_std(data, weights):
    """Jackknife Mean Standard Deviation.

    Computes weighted mean and standard error using delete-1 jackknife.
    Uses vectorized O(n) computation - no loops or random sampling.

    Parameters
    ----------
    data : numpy.ndarray
        input sample
    weights : numpy.ndarray
        weights

    Returns
    -------
    float
        weighted mean
    float
        jackknife standard error

    """
    n = len(data)
    if n < 2:
        return np.nanmean(data), np.nan

    # Total weighted sum and total weights
    total_weighted = np.sum(data * weights)
    total_weights = np.sum(weights)

    # Leave-one-out weighted means: (total - x_i * w_i) / (total_w - w_i)
    leave_one_out_means = (total_weighted - data * weights) / (total_weights - weights)

    # Jackknife variance estimate: (n-1)/n * sum((theta_i - theta_bar)^2)
    mean_of_means = np.nanmean(leave_one_out_means)
    jackknife_var = (n - 1) / n * np.nansum((leave_one_out_means - mean_of_means) ** 2)

    return total_weighted / total_weights, np.sqrt(jackknife_var)


def jackknife_mean_std_random(
    data,
    weights,
    remove_size=0.1,
    n_realization=100,
):
    """Jackknife Mean Standard Deviation (Random Subsampling).

    Computes weighted mean and standard deviation from random subsampling.
    Uses vectorized operations for efficiency.

    Parameters
    ----------
    data : numpy.ndarray
        input sample
    weights : numpy.ndarray
        weights
    remove_size : float, optional
        fraction of input sample to remove for each resampling,
        default is ``0.1``
    n_realization : int, optional
        number of resamples, default is ``100``

    Returns
    -------
    float
        weighted mean
    float
        weighted standard deviation

    """
    samp_size = len(data)
    keep_size_pc = 1 - remove_size

    if keep_size_pc < 0:
        raise ValueError("remove size should be in [0, 1]")

    subsamp_size = int(samp_size * keep_size_pc)

    # Generate all random indices at once (n_realization x subsamp_size)
    all_indices = np.random.randint(0, samp_size, size=(n_realization, subsamp_size))

    # Extract subsampled data and weights (n_realization x subsamp_size)
    sub_data = data[all_indices]
    sub_weights = weights[all_indices]

    # Compute weighted means for all realizations at once
    # weighted_mean = sum(data * weights) / sum(weights)
    weighted_sums = np.sum(sub_data * sub_weights, axis=1)
    weight_sums = np.sum(sub_weights, axis=1)

    # Handle division by zero
    with np.errstate(divide='ignore', invalid='ignore'):
        all_est = np.where(weight_sums > 0, weighted_sums / weight_sums, np.nan)

    return np.nanmean(all_est), np.nanstd(all_est)


def func_bias_quad_1D(params, x_data):
    """Func Bias Quad 1D.

    Function for quadratic 1D bias model.

    Parameters
    ----------
    params : lmfit.Parameters
        fit parameters
    x_data : numpy.ndarray
        x-values of the data

    Returns
    -------
    numpy.ndarray
        y-values of the model

    """
    q = params["q"].value
    m = params["m"].value
    c = params["c"].value

    y_model = q * x_data**2 + m * x_data + c

    return y_model


def loss_bias_quad_1d(params, x_data, y_data, err):
    """Loss Bias quad 1D.

    Loss function for Quadratic 1D model

    Parameters
    ----------
    params : lmfit.Parameters
        fit parameters
    x_data : numpy.ndarray
        x-values of the data
    y_data : numpy.ndarray
        y-values of the data
    err : numpy.ndarray
        error values of the data

    Returns
    -------
    numpy.ndarray
        residuals

    """
    y_model = func_bias_quad_1D(params, x_data)
    residuals = (y_model - y_data) / err
    return residuals


def quad_corr_quant(
    x,
    y,
    xlabel,
    ylabel,
    qlabel=None,
    mlabel=None,
    clabel=None,
    weights=None,
    n_bin=30,
    out_path=None,
    title="",
    colors=None,
    stats_file=None,
    verbose=False,
    seed=None,
    rng=None,
    error_method="jackknife",
):
    """Quadratic Correlation Quantity.

    Computes and plots quadratic correlation of y(n) as function of x.

    Parameters
    ----------
    x: array(double)
        input x value
    y: array(m) of double
        input y arrays
    xlabel, ylabel : str
        x-and y-axis labels
    mlabel : str, optional, default=None
        label for slope in the plot legend
    clabel : str, optional, default=None
        label for offset in the plot legend
    weights : array of double, optional, default=None
        weights of x points
    n_bin : double, optional, default=30
        number of points onto which data are binned
    out_path : str, optional, default=None
        output file path, if not given, plot is not saved to file
    title : str, optional, default=''
        plot title
    colors : array(m) of str, optional, default=None
        line colors
    stats_file : filehandler, optional, default=None
        output file for statistics
    verbose : bool, optional, default=False
        verbose output if True
    seed: int
        Seed to initialize the randoms. [Default: None]
    rng: numpy.random.RandomState
        Random generator. [Default: None]
    error_method : str, optional, default="jackknife"
        method for computing binned error estimates:
        - "analytical": fast weighted mean/std (no resampling)
        - "jackknife": delete-1 jackknife (deterministic, fast)
        - "jackknife_random": random subsampling (original method)

    Returns
    -------
    list
        1rst order coeff of each e_gal vs quantities for recap plot
    list
        2nd order coeff of each e_gal vs quantities for recap plot
    list
        names of the quantities associated to each slopes
    list
        errors of 1rst order coeff
    list
        errors of 2nd order coeff

    """
    # Init randoms
    if isinstance(rng, np.random.RandomState):
        master_rng = rng
    else:
        master_rng = np.random.RandomState(seed)

    n_y = len(y)

    if qlabel is None:
        qlabel = np.full(n_y, "q")
    if mlabel is None:
        mlabel = np.full(n_y, "m")
    if clabel is None:
        clabel = np.full(n_y, "c")

    if weights is None:
        weights = np.ones_like(y[0])

    size_all = len(y[0])
    for idx in range(1, n_y):
        if len(y[idx]) != size_all:
            raise IndexError
            (
                f"Size {len(y[idx])} of input #{idx} is different from size "
                + f"{size_all} of input #0"
            )
    size_bin = int(size_all / n_bin)
    diff_size = size_all - size_bin

    # Prepare arrays for binned data
    x_arg_sort = np.argsort(x)
    x_bin = []
    y_bin = []
    err_bin = []

    for idx in range(len(y)):
        y_bin.append([])
        err_bin.append([])

    # Precompute bin indices (depends only on x, not y)
    bin_indices = []
    for idx in range(n_bin):
        if idx < diff_size:
            bin_size_tmp = size_bin + 1
            starter = 0
        else:
            bin_size_tmp = size_bin
            starter = diff_size
        ind = x_arg_sort[
            starter + idx * bin_size_tmp : starter + (idx + 1) * bin_size_tmp
        ]
        bin_indices.append(ind)
        x_bin.append(np.mean(x[ind]))

    # Bin y data using precomputed indices
    for ind in bin_indices:
        for j in range(len(y)):
            if error_method == "analytical":
                mean, std_err = weighted_mean_std(y[j][ind], weights[ind])
            elif error_method == "jackknife":
                mean, std_err = jackknife_mean_std(y[j][ind], weights[ind])
            elif error_method == "jackknife_random":
                mean, std_err = jackknife_mean_std_random(
                    y[j][ind],
                    weights[ind],
                    remove_size=0.2,
                    n_realization=50,
                )
            else:
                raise ValueError(
                    f"Unknown error_method '{error_method}'. "
                    "Use 'analytical', 'jackknife', or 'jackknife_random'."
                )
            y_bin[j].append(mean)
            err_bin[j].append(std_err)

    x_bin = np.array(x_bin)
    for jdx in range(len(y)):
        y_bin[jdx] = np.array(y_bin[jdx])
        err_bin[jdx] = np.array(err_bin[jdx])

    # Fit affine functions, plot function and data
    slope = []
    qslope = []
    ticks_names = []
    m_err = []
    q_err = []
    plt.figure(figsize=(10, 6))
    for jdx in range(len(y)):
        params = Parameters()
        params.add("q", value=0.01)
        params.add("m", value=0.01)
        params.add("c", value=0.01)

        # Optimize parameters
        res = minimize(
            loss_bias_quad_1d, params, args=(x, y[jdx], 1 / np.sqrt(weights))
        )

        qslope.append(res.params["q"].value)
        slope.append(res.params["m"].value)

        ticks_names.append(f"{xlabel}_e_{jdx+1}")
        q_dm = ufloat(res.params["q"].value, res.params["q"].stderr)
        m_dm = ufloat(res.params["m"].value, res.params["m"].stderr)
        c_dc = ufloat(res.params["c"].value, res.params["c"].stderr)

        q_err.append(res.params["q"].stderr)
        m_err.append(res.params["m"].stderr)

        label = (
            rf"${qlabel[jdx]}={q_dm: .2ugL}, {mlabel[jdx]}={m_dm: .2ugL},"
            + f" {clabel[jdx]}={c_dc: .2ugL}$"
        )

        plt.plot(
            x_bin,
            func_bias_quad_1D(res.params, x_bin),
            c=colors[jdx],
            label=label,
        )

        plt.errorbar(
            x_bin,
            y_bin[jdx],
            yerr=err_bin[jdx],
            c=colors[jdx],
            fmt=".",
        )

        if stats_file:
            msg1 = "{}: {}={:.2ugP}".format(xlabel, qlabel[jdx], q_dm)
            msg2 = "{}: {}={:.2ugP}".format(xlabel, mlabel[jdx], m_dm)
            print_stats(msg1, stats_file, verbose=verbose)
            print_stats(msg2, stats_file, verbose=verbose)

    # Finalise plots
    plt_xmin, plt_xmax = plt.xlim()
    plt.xlim(plt_xmin, plt_xmax)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()

    plt.title(title)
    plt.tight_layout()

    if out_path:
        plt.savefig(out_path, bbox_inches="tight")
    plt.close()

    return slope, qslope, ticks_names, m_err, q_err


def quad_corr_n_quant(
    x_arr,
    y,
    xlabel_arr,
    ylabel,
    qlabel=None,
    mlabel=None,
    clabel=None,
    weights=None,
    n_bin=30,
    out_path_arr=None,
    title="",
    colors=None,
    stats_file=None,
    verbose=False,
    seed=None,
    error_method="jackknife",
):
    """Quadratic Correlation N Quantity.

    Compute n quadratic correlations of y(m) versus x_arr[n].

    Parameters
    ----------
    x_arr: array(n, double)
        input x value
    y: array(m) of double
        input y arrays
    xlabel, ylabel : str
        x-and y-axis labels
    mlabel : str, optional, default=None
        label for slope in the plot legend
    clabel : str, optional, default=None
        label for offset in the plot legend
    weights : array of double, optional, default=None
        weights of x points
    n_bin : double, optional, default=30
        number of points onto which data are binned
    out_path_arr : array(n) of str, optional, default=None
        output file path, if not given, plot is not saved to file
    title : str, optional, default=''
        plot title
    colors(m) : array of str, optional, default=None
        line colors
    stats_file : filehandler, optional, default=None
        output file for statistics
    verbose : bool, optional, default=False
        verbose output if True
    seed: int
        Seed to initialize the randoms. [Default: None]
    error_method : str, optional, default="jackknife"
        method for computing binned error estimates:
        - "analytical": fast weighted mean/std (no resampling)
        - "jackknife": delete-1 jackknife (deterministic, fast)
        - "jackknife_random": random subsampling (original method)

    """
    master_rng = np.random.RandomState(seed)
    seeds = master_rng.randint(low=0, high=2**30, size=len(x_arr))
    slopes = []
    qslopes = []
    ticks_label = []
    merr = []
    qerr = []

    if out_path_arr is None:
        out_path_arr = [None] * len(x_arr)
    for x, xlabel, out_path, seed_tmp in zip(
        x_arr, xlabel_arr, out_path_arr, seeds
    ):
        slope, qslope, ticks_names, m_err, q_err = quad_corr_quant(
            x,
            y,
            xlabel,
            ylabel,
            mlabel=mlabel,
            clabel=clabel,
            weights=weights,
            n_bin=n_bin,
            out_path=out_path,
            title=title,
            colors=colors,
            stats_file=stats_file,
            verbose=verbose,
            seed=seed_tmp,
            error_method=error_method,
        )

        for i in range(len(slope)):
            slopes.append(slope[i])
            qslopes.append(qslope[i])
            ticks_label.append(ticks_names[i])
            merr.append(m_err[i])
            qerr.append(q_err[i])

    ticks_positions = np.arange(1, len(slopes) + 1, 1)

    # MKDEBUG TODO: Move summary plot to separate function
    # Plot slopes
    plt.figure()
    plt.errorbar(
        ticks_positions,
        slopes,
        yerr=merr,
        color="peru",
        label="m",
        fmt=".",
    )

    plt.errorbar(
        ticks_positions,
        qslopes,
        yerr=qerr,
        color="crimson",
        label="q",
        fmt=".",
    )

    plt.xticks(
        ticks_positions,
        ticks_label,
        rotation=90,
        fontsize=10,
    )

    plt.yticks(fontsize=10)
    plt.axhline(
        y=0,
        color="black",
        linestyle="--",
    )
    plt.ylabel("q and m", fontsize=10)
    title = "(e1, e2) systematic tests (quadratic)"
    plt.title(title, fontsize=10)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path_arr[-1])
    plt.close()


def func_bias_lin_1d(params, x_data):
    """Func Bias Lin 1D.

    Function for linear 1D bias model.

    Parameters
    ----------
    params : lmfit.Parameters
        fit parameters
    x_data : numpy.ndarray
        x-values of the data

    Returns
    -------
    numpy.ndarray
        y-values of the model

    """
    m = params["m"].value
    c = params["c"].value

    y_model = m * x_data + c

    return y_model


def loss_bias_lin_1d(params, x_data, y_data, err):
    """Loss Bias Lin 1D.

    Loss function for linear 1D model

    Parameters
    ----------
    params : lmfit.Parameters
        fit parameters
    x_data : numpy.ndarray
        x-values of the data
    y_data : numpy.ndarray
        y-values of the data
    err : numpy.ndarray
        error values of the data

    Returns
    -------
    numpy.ndarray
        residuals

    """
    y_model = func_bias_lin_1d(params, x_data)
    residuals = (y_model - y_data) / err
    return residuals


def weighted_linear_fit(x, y, weights):
    """Weighted Linear Fit.

    Perform weighted linear regression y = m*x + c using analytical formula.

    Parameters
    ----------
    x : numpy.ndarray
        x-values
    y : numpy.ndarray
        y-values
    weights : numpy.ndarray
        weights for each data point

    Returns
    -------
    float
        slope m
    float
        intercept c

    """
    S = np.sum(weights)
    Sx = np.sum(weights * x)
    Sy = np.sum(weights * y)
    Sxx = np.sum(weights * x**2)
    Sxy = np.sum(weights * x * y)

    delta = S * Sxx - Sx**2
    if delta == 0:
        return np.nan, np.nan

    m = (S * Sxy - Sx * Sy) / delta
    c = (Sxx * Sy - Sx * Sxy) / delta

    return m, c


def bootstrap_linear_regression(x, y, weights, n_bootstrap=1000, seed=None):
    """Bootstrap Linear Regression.

    Compute linear regression parameters and uncertainties using bootstrap
    resampling.

    Parameters
    ----------
    x : numpy.ndarray
        x-values
    y : numpy.ndarray
        y-values
    weights : numpy.ndarray
        weights for each data point
    n_bootstrap : int, optional
        number of bootstrap samples, default is 1000
    seed : int, optional
        random seed for reproducibility

    Returns
    -------
    float
        slope m (mean of bootstrap samples)
    float
        slope uncertainty (std of bootstrap samples)
    float
        intercept c (mean of bootstrap samples)
    float
        intercept uncertainty (std of bootstrap samples)

    """
    rng = np.random.RandomState(seed)
    n = len(x)

    m_samples = np.zeros(n_bootstrap)
    c_samples = np.zeros(n_bootstrap)

    for i in range(n_bootstrap):
        # Resample with replacement
        idx = rng.choice(n, size=n, replace=True)
        x_boot = x[idx]
        y_boot = y[idx]
        w_boot = weights[idx]

        m_samples[i], c_samples[i] = weighted_linear_fit(x_boot, y_boot, w_boot)

    # Remove any NaN values from failed fits
    valid = ~(np.isnan(m_samples) | np.isnan(c_samples))
    m_samples = m_samples[valid]
    c_samples = c_samples[valid]

    if len(m_samples) == 0:
        return np.nan, np.nan, np.nan, np.nan

    return (
        np.mean(m_samples),
        np.std(m_samples),
        np.mean(c_samples),
        np.std(c_samples),
    )


def loss_bias_2d(params, x_data, y_data, err, order, mix):
    """Loss Bias 2D.

    Loss function for 2D model

    Parameters
    ----------
    params : lmfit.Parameters
        fit parameters
    x_data : numpy.ndarray
        two-component x-values of the data
    y_data : numpy.ndarray
        two-component y-values of the data
    err : numpy.ndarray
        error values of the data, assumed the same for both components
    order : str
        order of fit
    mix : bool
        mixing of components if True

    Raises
    ------
    IndexError :
        if input arrays x1_data and x2_data have different lenght

    Returns
    -------
    numpy.ndarray
        residuals

    """
    # Get x and y values of the input data
    x1_data = x_data[0]
    x2_data = x_data[1]
    y1_data = y_data[0]
    y2_data = y_data[1]

    if len(x1_data) != len(x2_data):
        raise IndexError("Length of both data components has to be equal")

    # Get model 1D y1 and y2 components
    y1_model, y2_model = func_bias_2d(
        params, x1_data, x2_data, order=order, mix=mix
    )

    # Compute residuals between data and model
    res1 = (y1_model - y1_data) / err
    res2 = (y2_model - y2_data) / err

    # Concatenate both components
    residuals = np.concatenate([res1, res2])

    return residuals


def print_fit_report(res, file=None):
    """Print Fit Report.

    Print report of minimizing result.

    Parameters
    ----------
    res : class lmfit.MinimizerResult
        results of the minization
    file : filehandler, optional
        output to file; if `None` (default) output to `stdout`

    """
    # chi^2
    print(f"chi^2 = {res.chisqr}", file=file)

    # Reduced chi^2
    print(f"reduced chi^2 = {res.redchi}", file=file)

    # Akaike Information Criterium
    print(f"aic = {res.aic}", file=file)

    # Bayesian Information Criterium
    print(f"bic = {res.bic}", file=file)


def corr_2d(
    x,
    y,
    weights=None,
    order="lin",
    mix=False,
    stats_file=None,
    verbose=False,
):
    """Corr 2D.

    Compute and plot 2D linear and quadratic correlations of (y1, y2) as
    function of (x1, x2).

    Parameters
    ----------
    x : array(double)
        input x value
    y : array(m) of double
        input y arrays
    weights  : array of double, optional, default=None
        weights of x points
    order : str, optional
        order of fit, default is 'lin'
    mix : bool
        mixing of components if True
    stats_file : filehandler, optional, default=None
        output file for statistics
    verbose : bool, optional
        verbose output if ``True``; default is ``False``

    Returns
    -------
    lmfit.Parameters
        best-fit parameters

    """
    if len(y) != 2 or len(x) != 2:
        raise IndexError("Input data needs to have two components")
    if any(len(y[0]) != c for c in {len(y[1]), len(x[0]), len(x[1])}):
        raise IndexError("Input data has inconsistent length")

    # Initialise parameters of model to fit
    params = Parameters()

    val_init = 0.0

    # Affine parameters
    for p_affine in ["a11", "a22", "c1", "c2"]:
        params.add(p_affine, value=val_init)

    if mix:
        # Linear mixing pararmeters
        params.add("a12", value=val_init)
        params.add("a21", value=val_init)

    if order == "quad":
        # Quadratic parameters
        for p_quad in ["q111", "q222"]:
            params.add(p_quad, value=val_init)

        if mix:
            # Quadratic mixing parameters
            for p_quad_mix in ["q112", "q122", "q212", "q211"]:
                params.add(p_quad_mix, value=val_init)

    # Mininise loss function
    err = 1 / np.sqrt(weights) if weights is not None else np.ones_like(y[0])
    res = minimize(loss_bias_2d, params, args=(x, y, err, order, mix))
    if stats_file:
        print_stats(
            f"2D fit order={order} mix={mix}:",
            stats_file,
            verbose=verbose,
        )
        print_fit_report(res, file=stats_file)
    if verbose:
        print_fit_report(res)

    return res.params


def affine_corr(
    x,
    y,
    xlabel,
    ylabel,
    mlabel=None,
    clabel=None,
    weights=None,
    n_bin=30,
    out_path=None,
    title="",
    colors=None,
    stats_file=None,
    verbose=False,
    seed=None,
    rng=None,
    regr_on_binned=False,
    error_method="jackknife",
    regr_error_method="covariance",
    n_bootstrap=1000,
):
    """Affine Corr.

    Computes and plots affine correlation of y(n) as function of x.

    Parameters
    ----------
    x: array(double)
        input x value
    y: array(m) of double
        input y arrays
    xlabel, ylabel : str
        x-and y-axis labels
    mlabel : str, optional, default=None
        label for slope in the plot legend
    clabel : str, optional, default=None
        label for offset in the plot legend
    weights : array of double, optional, default=None
        weights of x points
    n_bin : double, optional, default=30
        number of points onto which data are binned
    out_path : str, optional, default=None
        output file path, if not given, plot is not saved to file
    title : str, optional, default=''
        plot title
    colors : array(m) of str, optional, default=None
        line colors
    stats_file : filehandler, optional, default=None
        output file for statistics
    verbose : bool, optional, default=False
        verbose output if True
    seed: int
        Seed to initialize the randoms. [Default: None]
    rng: numpy.random.RandomState
        Random generator. [Default: None]
    regr_on_binned : bool, optional, default=False
        if True, perform regression on binned data (faster for large
        catalogues); if False, perform regression on unbinned data
    error_method : str, optional, default="jackknife"
        method for computing binned error estimates:
        - "analytical": fast weighted mean/std (no resampling)
        - "jackknife": delete-1 jackknife (deterministic, fast)
        - "jackknife_random": random subsampling (original method)
    regr_error_method : str, optional, default="covariance"
        method for computing regression parameter uncertainties:
        - "covariance": use lmfit covariance matrix (fast, assumes Gaussian errors)
        - "bootstrap": use bootstrap resampling (slower, more robust)
    n_bootstrap : int, optional, default=1000
        number of bootstrap samples (only used if regr_error_method="bootstrap")

    Returns
    -------
    list
        slopes of the linear fits
    list
        errors of the slopes
    list
        offsets of the linear fits
    list
        errors of the offsets
    list
        labels of the linear fits

    """
    # Init randoms
    if isinstance(rng, np.random.RandomState):
        master_rng = rng
    else:
        master_rng = np.random.RandomState(seed)

    n_y = len(y)

    if mlabel is None:
        mlabel = np.full(n_y, r"\alpha")
    if clabel is None:
        clabel = np.full(n_y, "c")

    if weights is None:
        weights = np.ones_like(y[0])

    if colors is None:
        prop_cycle = plt.rcParams["axes.prop_cycle"]
        colors = prop_cycle.by_key()["color"]

    size_all = len(y[0])
    for idx in range(1, n_y):
        if len(y[idx]) != size_all:
            raise IndexError
            (
                f"Size {len(y[idx])} of input #{idx} is different from size "
                + f"{size_all} of input #0"
            )
    size_bin = int(size_all / n_bin)
    diff_size = size_all - size_bin

    # Prepare arrays for binned data
    x_arg_sort = np.argsort(x)
    x_bin = []
    y_bin = []
    err_bin = []

    for idx in range(len(y)):
        y_bin.append([])
        err_bin.append([])

    # Precompute bin indices (depends only on x, not y)
    bin_indices = []
    for idx in range(n_bin):
        if idx < diff_size:
            bin_size_tmp = size_bin + 1
            starter = 0
        else:
            bin_size_tmp = size_bin
            starter = diff_size
        ind = x_arg_sort[
            starter + idx * bin_size_tmp : starter + (idx + 1) * bin_size_tmp
        ]
        bin_indices.append(ind)
        x_bin.append(np.mean(x[ind]))

    # Bin y data using precomputed indices
    for idx, ind in enumerate(bin_indices):
        for j in range(len(y)):
            if error_method == "analytical":
                # Fast weighted mean/std (no resampling)
                mean, std_err = weighted_mean_std(y[j][ind], weights[ind])
            elif error_method == "jackknife":
                # Delete-1 jackknife (deterministic, fast)
                mean, std_err = jackknife_mean_std(y[j][ind], weights[ind])
            elif error_method == "jackknife_random":
                # Random subsampling (original method)
                mean, std_err = jackknife_mean_std_random(
                    y[j][ind],
                    weights[ind],
                    remove_size=0.2,
                    n_realization=50,
                )
            else:
                raise ValueError(
                    f"Unknown error_method '{error_method}'. "
                    "Use 'analytical', 'jackknife', or 'jackknife_random'."
                )
            y_bin[j].append(mean)
            err_bin[j].append(std_err)

    x_bin = np.array(x_bin)
    for jdx in range(len(y)):
        y_bin[jdx] = np.array(y_bin[jdx])
        err_bin[jdx] = np.array(err_bin[jdx])

    # Fit affine functions, plot function and data
    plt.figure(figsize=(10, 6))

    m_arr = []
    m_err_arr = []
    c_arr = []
    c_err_arr = []
    tick_name_arr = []

    # Print regression info
    regr_mode = "binned" if regr_on_binned else "unbinned"
    err_mode = regr_error_method
    n_unbinned = len(x)
    n_binned = len(x_bin)
    n_regr = n_binned if regr_on_binned else n_unbinned
    print(
        f"Regression ({regr_mode}, errors={err_mode}): {xlabel}, "
        f"n_unbinned={n_unbinned}, n_binned={n_binned}, "
        f"using n={n_regr} points"
    )

    # Print regression mode header to stats file
    if stats_file:
        print_stats(
            f"--- Regression ({regr_mode}, errors={err_mode}): {xlabel} ---",
            stats_file,
            verbose=verbose,
        )

    for jdx in range(len(y)):
        if regr_error_method == "bootstrap":
            # Use bootstrap for both best-fit values and uncertainties
            if regr_on_binned:
                # Bootstrap on binned data
                w_bin = 1.0 / np.array(err_bin[jdx]) ** 2
                m_val, m_err, c_val, c_err = bootstrap_linear_regression(
                    x_bin, np.array(y_bin[jdx]), w_bin,
                    n_bootstrap=n_bootstrap, seed=master_rng.randint(2**30)
                )
            else:
                # Bootstrap on unbinned data
                m_val, m_err, c_val, c_err = bootstrap_linear_regression(
                    x, y[jdx], weights,
                    n_bootstrap=n_bootstrap, seed=master_rng.randint(2**30)
                )
        else:
            # Use lmfit for best-fit values and covariance-based uncertainties
            params = Parameters()
            params.add("m", value=0.01)
            params.add("c", value=0.01)

            if regr_on_binned:
                # Regression on binned data (faster for large catalogues)
                res = minimize(
                    loss_bias_lin_1d,
                    params,
                    args=(x_bin, y_bin[jdx], err_bin[jdx]),
                )
            else:
                # Regression on unbinned data
                res = minimize(
                    loss_bias_lin_1d, params, args=(x, y[jdx], 1 / np.sqrt(weights))
                )

            m_val = res.params["m"].value
            m_err = float(res.params["m"].stderr)
            c_val = res.params["c"].value
            c_err = float(res.params["c"].stderr)

        m_arr.append(m_val)
        m_err_arr.append(m_err)
        c_arr.append(c_val)
        c_err_arr.append(c_err)
        tick_name_arr.append(f"{xlabel}_e{jdx+1}")

        # Plot results
        m_dm = ufloat(m_val, m_err)
        c_dc = ufloat(c_val, c_err)
        label = rf"${mlabel[jdx]}={m_dm: .2ugL}, {clabel[jdx]}={c_dc: .2ugL}$"
        plt.plot(
            x_bin,
            m_val * x_bin + c_val,
            c=colors[jdx],
            label=label,
        )
        plt.errorbar(
            x_bin, y_bin[jdx], yerr=err_bin[jdx], c=colors[jdx], fmt="."
        )

        if stats_file:
            msg_m = "{}: {}={:.2ugP}".format(xlabel, mlabel[jdx], m_dm)
            msg_c = "{}: {}={:.2ugP}".format(xlabel, clabel[jdx], c_dc)
            print_stats(msg_m, stats_file, verbose=verbose)
            print_stats(msg_c, stats_file, verbose=verbose)

    # Finalise plots
    plt_xmin, plt_xmax = plt.xlim()
    plt.xlim(plt_xmin, plt_xmax)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()

    plt.title(title)
    plt.tight_layout()

    if out_path:
        plt.savefig(out_path, bbox_inches="tight")

    plt.close()

    return m_arr, m_err_arr, c_arr, c_err_arr, tick_name_arr


def read_regr_res_from_file(path):
    """Read Regr Res From File.

    Read regression result from ASCII file.

    Parameters
    ----------
    path: str
        path to the file

    Returns
    -------
    list
        list of slopes
    list
        list of slope errors
    list
        list of offsets
    list
        list of offset errors
    list
        list of tick names

    """
    with open(path, "r") as f:
        str_m = f.readline()
        m = cs_args.my_string_split(str_m, num=2, stop=True)
        str_m_err = f.readline()
        m_err = cs_args.my_string_split(str_m_err, num=2, stop=True)
        str_c = f.readline()
        c = cs_args.my_string_split(str_c, num=2, stop=True)
        str_c_err = f.readline()
        c_err = cs_args.my_string_split(str_c_err, num=2, stop=True)
        str_tick_name = f.readline()
        tick_name = cs_args.my_string_split(str_tick_name, num=2, stop=True)

    return (
        [float(i) for i in m],
        [float(i) for i in m_err],
        [float(i) for i in c],
        [float(i) for i in c_err],
        tick_name,
    )


def write_regr_res_to_file(m, m_err, c, c_err, tick_name, path):
    """Write Regr Res To File.

    Write regression result to ASCII file.

    Parameters
    ----------
    m: list
        slopes for first and second ellipticity component
    m_err: list
        errors of the slopes for first and second ellipticity component
    c: list
        offsets for first and second ellipticity component
    c_err: list
        errors of the offsets for first and second ellipticity component
    tick_name: list
        names of the quantities associated to each slope
    path: str
        path to the file

    """
    with open(path, "w") as f:
        f.write(" ".join(map(str, m)))
        f.write("\n")
        f.write(" ".join(map(str, m_err)))
        f.write("\n")
        f.write(" ".join(map(str, c)))
        f.write("\n")
        f.write(" ".join(map(str, c_err)))
        f.write("\n")
        # Write quoted labels to deal with white spaces
        f.write(" ".join(f'"{x}"' for x in tick_name))
        f.write("\n")


def affine_corr_n(
    x_arr,
    y,
    xlabel_arr,
    ylabel,
    mlabel=None,
    clabel=None,
    weights=None,
    n_bin=30,
    out_path_arr=None,
    title="",
    colors=None,
    stats_file=None,
    verbose=False,
    seed=None,
    regr_on_binned=False,
    error_method="jackknife",
    regr_error_method="covariance",
    n_bootstrap=1000,
):
    """Affine Corr N.

    Compute n affine correlations of y(m) versus x_arr[n].

    Parameters
    ----------
    x_arr: array(n, double)
        input x value
    y: array(m) of double
        input y arrays
    xlabel, ylabel : str
        x-and y-axis labels
    mlabel : str, optional, default=None
        label for slope in the plot legend
    clabel : str, optional, default=None
        label for offset in the plot legend
    weights : array of double, optional, default=None
        weights of x points
    n_bin : double, optional, default=30
        number of points onto which data are binned
    out_path_arr : array(n) of str, optional, default=None
        output file path, if not given, plot is not saved to file
    title : str, optional, default=''
        plot title
    colors(m) : array of str, optional, default=None
        line colors
    stats_file : filehandler, optional, default=None
        output file for statistics
    verbose : bool, optional, default=False
        verbose output if True
    seed: int
        Seed to initialize the randoms. [Default: None]
    regr_on_binned : bool, optional, default=False
        if True, perform regression on binned data (faster for large
        catalogues); if False, perform regression on unbinned data
    error_method : str, optional, default="jackknife"
        method for computing binned error estimates:
        - "analytical": fast weighted mean/std (no resampling)
        - "jackknife": delete-1 jackknife (deterministic, fast)
        - "jackknife_random": random subsampling (original method)
    regr_error_method : str, optional, default="covariance"
        method for computing regression parameter uncertainties:
        - "covariance": use lmfit covariance matrix (fast, assumes Gaussian errors)
        - "bootstrap": use bootstrap resampling (slower, more robust)
    n_bootstrap : int, optional, default=1000
        number of bootstrap samples (only used if regr_error_method="bootstrap")

    """
    master_rng = np.random.RandomState(seed)
    seeds = master_rng.randint(low=0, high=2**30, size=len(x_arr))

    if out_path_arr is None:
        out_path_arr = [None] * len(x_arr)
    m_arr = []
    m_err_arr = []
    c_arr = []
    c_err_arr = []
    tick_name_arr = []
    for x, xlabel, out_path, seed_tmp in zip(
        x_arr, xlabel_arr, out_path_arr, seeds
    ):

        out_path_txt = f"{out_path}.txt"
        if os.path.exists(out_path_txt):
            print(f"Reading regression result from file {out_path_txt}.")
            m, m_err, c, c_err, tick_name = read_regr_res_from_file(out_path_txt)
        else:
            print(f"Running regression, writing result to file {out_path_txt}.")
            m, m_err, c, c_err, tick_name = affine_corr(
                x,
                y,
                xlabel,
                ylabel,
                mlabel=mlabel,
                clabel=clabel,
                weights=weights,
                n_bin=n_bin,
                out_path=out_path,
                title=title,
                colors=colors,
                stats_file=stats_file,
                verbose=verbose,
                seed=seed_tmp,
                regr_on_binned=regr_on_binned,
                error_method=error_method,
                regr_error_method=regr_error_method,
                n_bootstrap=n_bootstrap,
            )
            write_regr_res_to_file(m, m_err, c, c_err, tick_name, out_path_txt)
        m_arr.extend(m)
        m_err_arr.extend(m_err)
        c_arr.extend(c)
        c_err_arr.extend(c_err)
        tick_name_arr.extend(tick_name)

    return m_arr, m_err_arr, c_arr, c_err_arr, tick_name_arr


def save_to_file(data, fname):
    """Save To File.

    Save data to .pkl (pickle) file.

    Parameters
    ----------
    data : dict
        input data
    fname : str
        output file name

    See also
    --------
    read_from_file

    """
    with open(fname, "wb") as f:
        pickle.dump(data, f)


def read_from_file(fname):
    """Read From File.

    Read data from .pkl (pickle) file.

    Parameters
    ----------
    fname : str
        input file name

    Returns
    -------
    dict
        data

    See also
    --------
    save_to_file

    """

    with open(fname, "rb") as f:
        data = pickle.load(f)

    return data


def param_order2spin(p_dp, order, mix):
    """Param Order 2 Spin.

    Transform parameter from natural to spin coefficients.

    Parameters
    ----------
    p_dp : dict
        Parameter natural coefficients
    order : str
        expansion order, one of 'linear', 'quad'
    mix : bool
        ellipticity components are mixed if ``True``

    Returns
    -------
    dict
        Parameter spin coefficients

    """
    s_ds = {"x0": 0.5 * (p_dp["a11"] + p_dp["a22"])}

    if order == "quad" and mix:
        s_ds["x2"] = 0.5 * (p_dp["q111"] + p_dp["q122"])
        s_ds["y2"] = 0.5 * (p_dp["q211"] - p_dp["q222"])
        s_ds["x-2"] = 0.25 * (p_dp["q111"] - p_dp["q122"] + p_dp["q212"])
        s_ds["y-2"] = 0.25 * (p_dp["q211"] - p_dp["q222"] - p_dp["q112"])

    s_ds["x4"] = 0.5 * (p_dp["a11"] - p_dp["a22"])

    if mix:
        s_ds["y4"] = 0.5 * (p_dp["a12"] + p_dp["a21"])
        s_ds["y0"] = 0.5 * (-p_dp["a12"] + p_dp["a21"])

    if order == "quad" and mix:
        s_ds["x6"] = 0.25 * (p_dp["q111"] - p_dp["q122"] - p_dp["q212"])
        s_ds["y6"] = 0.25 * (p_dp["q211"] - p_dp["q222"] + p_dp["q112"])

    return s_ds
