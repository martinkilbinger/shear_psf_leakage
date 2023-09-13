"""CORRELATION.

:Name: correlation.py

:Description: This script contains methods to deal with
    auto- and cross-correlations.

:Author: Martin Kilbinger <martin.kilbinger@cea.fr>
         Axel Guinot

"""

import numpy as np


from shear_psf_leakage import leakage

import treecorr


def func_bias_lin_1d(params, x_data):
    """Func Bias Lin 1D.

    Function for linear 1D bias model.

    Parameters
    ----------
    params : lmfit.Parameters
        fit parameters
    x_data : numpy.array
        x-values of the data

    Returns
    -------
    numpy.array
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
    x_data : numpy.array
        x-values of the data
    y_data : numpy.array
        y-values of the data
    err : numpy.array
        error values of the data

    Returns
    -------
    numpy.array
        residuals

    """
    y_model = func_bias_lin_1d(params, x_data)
    residuals = (y_model - y_data) / err
    return residuals


def loss_bias_2d(params, x_data, y_data, err, order, mix):
    """Loss Bias 2D.

    Loss function for 2D model

    Parameters
    ----------
    params : lmfit.Parameters
        fit parameters
    x_data : numpy.array
        two-component x-values of the data
    y_data : numpy.array
        two-component y-values of the data
    err : numpy.array
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
    numpy.array
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
    y1_model, y2_model = leakage.func_bias_2d(
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
    dict :
        Parameter spin coefficients

    """
    s_ds = {}

    s_ds["x0"] = 0.5 * (p_dp["a11"] + p_dp["a22"])

    if order == "quad" and mix:
        s_ds["x2"] = 0.5 * (p_dp["q111"] + p_dp["q122"])
        s_ds["y2"] = 0.5 * (p_dp["q211"] - p_dp["q222"])
        s_ds["x-2"] = 0.25 * (p_dp["q111"] - p_dp["q122"] + p_dp["q212"])
        s_ds["y-2"] = 0.25 * (p_dp["q211"] - p_dp["q222"] - p_dp["q112"])

    s_ds["x4"] = 0.5 * (p_dp["a11"] - p_dp["a22"])

    if mix:
        s_ds["y4"] = p_dp["a12"]

    if order == "quad" and mix:
        s_ds["x6"] = 0.25 * (p_dp["q111"] - p_dp["q122"] - p_dp["q212"])
        s_ds["y6"] = 0.25 * (p_dp["q211"] - p_dp["q222"] + p_dp["q112"])

    return s_ds


def xi_a_b(
    ra_a,
    dec_a,
    e1_a,
    e2_a,
    w_a,
    ra_b,
    dec_b,
    e1_b,
    e2_b,
    w_b=None,
    theta_min_amin=2,
    theta_max_amin=200,
    n_theta=20,
):
    """Xi A B.

    Cross-correlation between two catalogues a and b.

    Parameters
    ----------
    ra_a : numpy.ndarray
        right ascension values of catalogue a
    dec_a : numpy.ndarray
        right ascension values of catalogue a
    e1_a : numpy.ndarray
        first ellipticity compoment values of catalogue a
    e1_a : numpy.ndarray
        second ellipticity compoment values of catalogue a
    w_a : numpy.ndarray
        weights of catalogue a
    ra_b : numpy.ndarray
        right ascension values of catalogue b
    dec_b : numpy.ndarray
        right ascension values of catalogue b
    e1_b : numpy.ndarray
        first ellipticity compoment values of catalogue b
    e1_b : numpy.ndarray
        second ellipticity compoment values of catalogue b
    w_b : numpy.ndarray, optional
        weights of catalogue b; default with ``None`` is 1 for all weights
    theta_min_amin : float, optional
        minimum angular scale in arc minutes; default is 2
    theta_min_amax : float, optional
        minimum angular scale in arc minutes; default is 200
    n_theta : int, optional
        number of angular scales; default is 20

    Returns
    -------
    treecorr.GGCorrelation
        correlation result

    """
    unit = "degrees"

    cat_a = treecorr.Catalog(
        ra=ra_a,
        dec=dec_a,
        g1=e1_a,
        g2=e2_a,
        w=w_a,
        ra_units=unit,
        dec_units=unit,
    )
    cat_b = treecorr.Catalog(
        ra=ra_b,
        dec=dec_b,
        g1=e1_b,
        g2=e2_b,
        w=w_b,
        ra_units=unit,
        dec_units=unit,
    )

    TreeCorrConfig = {
        "ra_units": unit,
        "dec_units": unit,
        "sep_units": "arcminutes",
        "min_sep": theta_min_amin,
        "max_sep": theta_max_amin,
        "nbins": n_theta,
    }
    gg = treecorr.GGCorrelation(TreeCorrConfig)

    gg.process(cat_a, cat_b)

    return gg


def correlation_ab_bb(
    ra_a,
    dec_a,
    e1_a,
    e2_a,
    weights_a,
    ra_b,
    dec_b,
    e1_b,
    e2_b,
    theta_min_amin=2,
    theta_max_amin=200,
    n_theta=20,
):
    """Correlation ab bb.

    Shear correlation functions between two samples a and b.
    Compute xi_ab and xi_bb.

    Parameters
    ----------
    ra_a, dec_a : array of float
        coordinates of sample a
    e1_a, e2_a : array of float
        ellipticities of sample a
    weights_a : array of float
        weights of sample a
    ra_b, dec_b : array of float
        coordinates of sample b
    e1_b, e2_b : array of float
        ellipticities of sample b
    theta_min_amin : float, optional
        minimum angular scale in arcmin, default is 2
    theta_max_amin : float, optional
        maximum angular scale in arcmin, default is 200
    n_theta : int, optional
        number of angular scales, default is 20

    Returns
    -------
    xi_ab, xi_bb : correlations
        correlations ab, and bb

    """
    r_corr_ab = xi_a_b(
        ra_a,
        dec_a,
        e1_a,
        e2_a,
        weights_a,
        ra_b,
        dec_b,
        e1_b,
        e2_b,
        theta_min_amin=theta_min_amin,
        theta_max_amin=theta_max_amin,
        n_theta=n_theta,
    )
    r_corr_bb = xi_a_b(
        ra_b,
        dec_b,
        e1_b,
        e2_b,
        np.ones_like(ra_b),
        ra_b,
        dec_b,
        e1_b,
        e2_b,
        theta_min_amin=theta_min_amin,
        theta_max_amin=theta_max_amin,
        n_theta=n_theta,
    )

    return r_corr_ab, r_corr_bb


def correlation_ab_bb_matrix(
    ra_a,
    dec_a,
    e1_a,
    e2_a,
    weights_a,
    ra_b,
    dec_b,
    e1_b,
    e2_b,
    theta_min_amin=2,
    theta_max_amin=200,
    n_theta=20,
):
    """Correlation 12 22 Matrix.

    Shear correlation function matrices between two samples a and b.
    Computes the xi_ab and xi_bb matrices.

    Parameters
    ----------
    ra_a, dec_a : numpy.ndarray
        coordinates of sample a
    e1_a, e2_a : numpy.ndarray
        ellipticities of sample a
    weights_a : numpy.ndarray
        weights of sample a
    ra_b, dec_b : numpy.ndarray
        coordinates of sample b
    e1_b, e2_b : numpy.ndarray
        ellipticities of sample b
    theta_min_amin : float, optional
        minimum angular scale in arcmin; default is 2
    theta_max_amin : float, optional
        maximum angular scale in arcmin; default is 200
    n_theta : int, optional
        number of angular scales; default is 20

    Returns
    -------
    treecorr.GGCorrelation
        correlations ab
    treecorr.GGCorrelation
        correlations bb

    """
    # Create zero arrays for both samples
    ell_a_zero = np.zeros_like(e1_a)
    ell_b_zero = np.zeros_like(e1_b)

    xi_ab = np.zeros((2, 2), dtype=treecorr.GGCorrelation)
    xi_bb = np.zeros((2, 2), dtype=treecorr.GGCorrelation)
    ell_a = np.empty(shape=(2, len(e1_a)))
    ell_b = np.empty(shape=(2, len(e1_b)))

    # We know that xi_+ = xi_tt + xi_xx = xi_11 + xi_22.
    # To only get one component "xi_11", we need to set
    # the other to zero.
    # xi_ab_11 = <e1_a e1_b>; e2_a = e2_b = 0
    # xi_ab_22 = <e2_a e2_b>; e1_a = e1_b = 0
    # xi_ab_12 = <e1_a e2_b>; e2_a = e1_b = 0
    # xi_ab_21 = <e2_a e1_b>; e1_a = e2_b = 0

    ell_a[0] = e1_a
    ell_a[1] = e2_a

    ell_b[0] = e1_b
    ell_b[1] = e2_b

    for idx in (0, 1):
        for jdx in (0, 1):
            xi_ab[idx][jdx] = xi_a_b(
                ra_a,
                dec_a,
                ell_a[idx],
                ell_a_zero,
                weights_a,
                ra_b,
                dec_b,
                ell_b[jdx],
                ell_b_zero,
                theta_min_amin=theta_min_amin,
                theta_max_amin=theta_max_amin,
                n_theta=n_theta,
            )
            xi_bb[idx][jdx] = xi_a_b(
                ra_b,
                dec_b,
                ell_b[idx],
                ell_b_zero,
                np.ones_like(ra_b),
                ra_b,
                dec_b,
                ell_b[jdx],
                ell_b_zero,
                theta_min_amin=theta_min_amin,
                theta_max_amin=theta_max_amin,
                n_theta=n_theta,
            )


    return xi_ab, xi_bb


def alpha(r_corr_gp, r_corr_pp, e1_gal, e2_gal, weights_gal, e1_star, e2_star):
    """Alpha.

    Compute scale-dependent PSF leakage alpha.

    Parameters
    ----------
    r_corr_gp, r_corr_pp : correlations
        correlations galaxy-star, star-star
    e1_gal, e2_gal : array of float
        galaxy ellipticities
    weights_gal : array of float
        galaxy weights
    e1_star, e2_star : array of float
        galaxy ellipticities

    Returns
    -------
    alpha, sig_alpha : float
        mean and std of alpha

    """
    complex_gal = (
        np.average(e1_gal, weights=weights_gal)
        + np.average(e2_gal, weights=weights_gal) * 1j
    )
    complex_psf = np.mean(e1_star) + np.mean(e2_star) * 1j

    alpha_leak = (
        r_corr_gp.xip - np.real(np.conj(complex_gal) * complex_psf)
    ) / (r_corr_pp.xip - np.abs(complex_psf) ** 2)
    sig_alpha_leak = np.abs(alpha_leak) * np.sqrt(
        r_corr_gp.varxip / r_corr_gp.xip**2
        + r_corr_pp.varxip / r_corr_pp.xip**2
    )

    return alpha_leak, sig_alpha_leak


def alpha_matrix(
    r_corr_gp_m, r_corr_pp_m, e1_gal, e2_gal, weights_gal, e1_star, e2_star
):
    """Alpha Matrix.

    Compute scale-dependent PSF leakage alpha matrix.

    Parameters
    ----------
    r_corr_gp_m, r_corr_pp_m : correlations
        correlations galaxy-star, star-star
    e1_gal, e2_gal : array of float
        galaxy ellipticities
    weights_gal : array of float
        galaxy weights
    e1_star, e2_star : array of float
        galaxy ellipticities

    Returns
    -------
    alpha : numpy.ndarray
        alpha, 2x2 matrix

    """
    e_g = np.matrix(
        [
            np.average(e1_gal, weights=weights_gal),
            np.average(e2_gal, weights=weights_gal),
        ]
    )
    e_p = np.matrix(
        [
            np.mean(e1_star),
            np.mean(e2_star),
        ]
    )

    n_theta = r_corr_gp_m[0][0].nbins
    xi_gp_m = np.zeros((2, 2, n_theta))
    xi_pp_m = np.zeros((2, 2, n_theta))
    for idx in (0, 1):
        for jdx in (0, 1):
            xi_gp_m[idx][jdx] = r_corr_gp_m[idx][jdx].xip
            xi_pp_m[idx][jdx] = r_corr_pp_m[idx][jdx].xip

    xi_pp_m_inv = np.zeros((2, 2, n_theta))
    for ndx in range(n_theta):
        xi_pp_m_inv[:, :, ndx] = np.linalg.inv(xi_pp_m[:, :, ndx])

    alpha_leak_m = np.zeros((2, 2, n_theta))
    for ndx in range(n_theta):
        alpha_leak_m[:, :, ndx] = np.dot(
            xi_gp_m[:, :, ndx] - e_g * e_p.transpose(),
            xi_pp_m_inv[:, :, ndx]
        )

    return alpha_leak_m


def check_consistency_scales(xi_a, xi_b):
    """Check Consistency Scales.

    Print warning if angular scales between two correlation results do
    not match.

    Parameters
    ----------
    xi_a : treecorr.GGCorrelation
        correlation a
    xi_b : treecorr.GGCorrelation
        correlation b

    """

    if any(
        np.abs(xi_a.meanr - xi_b.meanr) / xi_a.meanr > 0.1
    ):
        print("Warning: angular scales not conr_corr_gpsistent")
