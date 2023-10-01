# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.15.2
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# # Scale-dependent PSF leakage
#
# package shear_psf_leakge

# %reload_ext autoreload
# %autoreload 2

# +
import os
import sys
import matplotlib.pylab as plt
from astropy import units

from cs_util import plots as cs_plots

import shear_psf_leakage.run_scale as run
from shear_psf_leakage.leakage import *
# -

# ## Compute leakage

# Create leakage instance
obj = run.LeakageScale()


# ## Set input parameters

params_in_path = "params_leakage_scale.py"
if os.path.exists(params_in_path):
    print(f"Reading configuration script {params_in_path}")

    with open(params_in_path) as f:
        exec(f.read())

    # Set instance parameters, copy from above
    for key in params_in:
        obj._params[key] = params_in[key]

else:
    print(f"Configuration script {params_in_path} not found, asking for user input")

    for key in obj._params:
        msg = f"{key}? [{obj._params[key]}] "
        val_user = input(msg)
        if val_user != "":
            obj._params[key] = val_user


print(obj._params)

# + [markdown] jp-MarkdownHeadingCollapsed=true
# ### Run
# There are two options to run the leakage compuations:
# 1. Run all at once with single class routine.
# 2. Execute individual steps.
# -

# ### Option 1. Run all at once

# +
# obj.run()
# -

# ### Option 2. Execute individual steps

# Check parameter validity
obj.check_params()

# Prepare output directory and stats file
obj.prepare_output()

# Prepare input
obj.read_data()

# #### Compute correlation functions
# The following command calls `treecorr` to compute auto- and cross-correlation functions.
# This can take a few minutes.

# Compute various correlation functions.
obj.compute_corr_gp_pp_alpha()

# #### Scale-dependent alpha function

# Compute approximate leakage by ignoring mean ellipticity terms
obj.do_alpha(fast=True)
alpha_fast = obj.alpha_leak

# Compute correct leakage
obj.do_alpha()

# +
# Plot

xlim = [obj._params["theta_min_amin"], obj._params["theta_max_amin"]]
ylim = obj._params["leakage_alpha_ylim"]

ftheta = 1.025
theta = [obj.r_corr_gp.meanr / ftheta, obj.r_corr_gp.meanr * ftheta]
alpha_theta = [obj.alpha_leak, alpha_fast]
yerr = [obj.sig_alpha_leak] * 2
labels = ["exact", "approximate"]
xlabel = r"$\theta$ [arcmin]"
ylabel = r"$\alpha(\theta)$"
title = ""
out_path = f"{obj._params['output_dir']}/alpha_leakage.png"

cs_plots.plot_data_1d(
    theta,
    alpha_theta,
    yerr,
    title,
    xlabel,
    ylabel,
    out_path,
    labels=labels,
    xlog=True,
    xlim=xlim,
    ylim=ylim,
    close_fig=False,
)
# -

# #### xi_sys cross-correlation function

# Compute
obj.do_xi_sys()

# Plot
obj.plot_xi_sys(close_fig=False)

# Theoretical model for xi_pm
xi_p_theo, xi_m_theo = run.get_theo_xi(
    obj.r_corr_gp.meanr * units.arcmin,
    obj._params["dndz_path"],
)

# obj Plot ratio
obj.plot_xi_sys_ratio(xi_p_theo, xi_m_theo, close_fig=False)

obj.compute_corr_gp_pp_alpha_matrix()

obj.alpha_matrix()

# +
# Exact

# TODO Check equation
r = obj.Xi_pp_m[0][1] ** 2 / (obj.Xi_pp_m[0][0] * obj.Xi_pp_m[1][1])

plt.semilogx(obj.r_corr_gp.meanr, r, label="$r$")
plt.semilogx(obj.r_corr_gp.meanr, 1 / (1 - r), label="$(1 - r)^{-1}$")
plt.semilogx(obj.r_corr_gp.meanr, 1 / (-1 + 1/r), label="$(-1 + r^{-1})^{-1}$")

# approximate
r_fast = obj.xi_pp_m[0][1] ** 2 / (obj.xi_pp_m[0][0] * obj.xi_pp_m[1][1])

plt.semilogx(obj.r_corr_gp.meanr, r_fast, ":")
plt.semilogx(obj.r_corr_gp.meanr, 1 / (1 - r_fast), ":",)
plt.semilogx(obj.r_corr_gp.meanr, 1 / (-1 + 1/r_fast), ":")

plt.legend()

plt.ylim(-1, 4)
plt.savefig("r.png")

# +
# Diagonal elements
plt.clf()
plt.semilogx(obj.r_corr_gp.meanr, obj.alpha_leak_m[0][0] + obj.alpha_leak_m[1][1], "-", label=r"$\alpha_{11} + \alpha_{22}$ (spin-0)")
plt.semilogx(obj.r_corr_gp.meanr, obj.alpha_leak_m[0][0] - obj.alpha_leak_m[1][1], "-", label=r"$\alpha_{11} - \alpha_{22}$ (spin-4)")
plt.legend()
xlim = [obj._params["theta_min_amin"], obj._params["theta_max_amin"]]
plt.xlim(xlim)
plt.savefig("alpha_leakage_m_11_pm_22.png")

# -

# +
# Check alpha_m elements
plt.semilogx(obj.r_corr_gp.meanr, obj.alpha_leak_m[0][0], "-", label=r"$\alpha_{11}$")

D = obj.Xi_pp_m[0][0] * obj.Xi_pp_m[1][1] - obj.Xi_pp_m[0][1] * obj.Xi_pp_m[1][0]

a_11 = (obj.Xi_gp_m[0][0] * obj.Xi_pp_m[1][1] - obj.Xi_gp_m[0][1] * obj.Xi_pp_m[1][0]) / D
plt.semilogx(obj.r_corr_gp.meanr, a_11, "s", label=r"$a_{11}$", mfc="none")

a_11_bis = (
    obj.Xi_gp_m[0][0] / (obj.Xi_pp_m[0][0] * (1 - r))
    - obj.Xi_gp_m[0][1] / (obj.Xi_pp_m[0][1] * (-1 + 1/r))
)
plt.semilogx(obj.r_corr_gp.meanr, a_11_bis, "p", label=r"$a_{11}$ bis", mfc="none")

plt.legend()


# +
# Check alpha_m elements
plt.semilogx(obj.r_corr_gp.meanr, obj.alpha_leak_m[1][1], "-", label=r"$\alpha_{22}$")

a_22 = (obj.Xi_gp_m[1][1] * obj.Xi_pp_m[0][0] - obj.Xi_gp_m[1][0] * obj.Xi_pp_m[0][1]) / D
plt.semilogx(obj.r_corr_gp.meanr, a_22, "s", label=r"$a_{22}$", mfc="none")

a_22_bis = (
    obj.Xi_gp_m[1][1] / (obj.Xi_pp_m[1][1] * (1 - r))
    - obj.Xi_gp_m[1][0] / (obj.Xi_pp_m[1][0] * (-1 + 1/r))
)
plt.semilogx(obj.r_corr_gp.meanr, a_22_bis, "p", label=r"$a_{22}$ bis", mfc="none")

plt.legend()

# +
# Check alpha_m elements
plt.semilogx(obj.r_corr_gp.meanr, obj.alpha_leak_m[0][0] + obj.alpha_leak_m[1][1], "-", label=r"tr $\mathbf{\alpha}$")

plt.semilogx(obj.r_corr_gp.meanr, a_11 + a_22, "s", label=r"$a_{11} + a_{22}$", mfc="none")

#plt.semilogx(obj.r_corr_gp.meanr, a_11_bis + a_22_bis, "p", label=r"$a_{11} + a_{22}$ bis", mfc="none")

t_sym = (
    obj.Xi_gp_m[0][0] / ( obj.Xi_pp_m[0][0] * (1 - r) )
    + obj.Xi_gp_m[1][1] / ( obj.Xi_pp_m[1][1] * (1 - r) )
)
plt.semilogx(obj.r_corr_gp.meanr, t_sym, 'v', label=r"$\alpha$ sym")

t_asym = (
    - obj.Xi_gp_m[0][1] / ( obj.Xi_pp_m[0][1] * (-1 + 1/r) )
    - obj.Xi_gp_m[1][0] / ( obj.Xi_pp_m[1][0] * (-1 + 1/r) )
)
plt.semilogx(obj.r_corr_gp.meanr, t_asym, '^', label=r"$\alpha$ asym")

plt.semilogx(obj.r_corr_gp.meanr, t_sym + t_asym, '.', label=r"$\alpha$ sym + asym")


plt.semilogx(obj.r_corr_gp.meanr, obj.alpha_leak * 2, ':', label=r"$2 \alpha$")

plt.legend(bbox_to_anchor=(1.2, 0.5))
_ = plt.ylim(-0.25, 0.25)
# -
# #### Consistency relations for scalar leakage

# If the leakage is a scalar function, it can be expressed in three different ways.

# +
alpha_1 = obj.Xi_gp_m[0][0] / obj.Xi_pp_m[1][1]
alpha_2 = obj.Xi_gp_m[1][1] / obj.Xi_pp_m[1][1]

alpha_1_std =  np.abs(alpha_1) * np.sqrt(                              
        obj.xi_std_gp_m[0][0] ** 2 / obj.xi_gp_m[0][0] ** 2                                     
        + obj.xi_std_pp_m[0][0] ** 2 / obj.xi_pp_m[0][0] ** 2                                   
    )

alpha_2_std =  np.abs(alpha_2) * np.sqrt(                              
        obj.xi_std_gp_m[1][1] ** 2 / obj.xi_gp_m[1][1] ** 2                                     
        + obj.xi_std_pp_m[1][1] ** 2 / obj.xi_pp_m[1][1] ** 2                                   
    )

# +
xlim = [obj._params["theta_min_amin"], obj._params["theta_max_amin"]]
ylim = obj._params["leakage_alpha_ylim"]

ftheta = 1.05
theta = [obj.r_corr_gp.meanr / ftheta, obj.r_corr_gp.meanr, obj.r_corr_gp.meanr * ftheta]
alpha_theta = [obj.alpha_leak, alpha_1, alpha_2]
yerr = [obj.sig_alpha_leak, alpha_1_std, alpha_2_std]
labels = [r"$\alpha$", r"$\alpha_1$", r"$\alpha_2$"]
xlabel = r"$\theta$ [arcmin]"
ylabel = r"$\alpha(\theta)$"
title = ""
out_path = f"{obj._params['output_dir']}/alpha_leakage_scalar_consistency.png"

cs_plots.plot_data_1d(
    theta,
    alpha_theta,
    yerr,
    title,
    xlabel,
    ylabel,
    out_path,
    labels=labels,
    xlog=True,
    xlim=xlim,
    ylim=ylim,
    close_fig=False,
)
# -

# If alpha is a scalar, the mixed-component centered cross-correlation functions should be identical.

# +
xlim = [obj._params["theta_min_amin"], obj._params["theta_max_amin"]]

Xi_gp_plus = obj.Xi_gp_m[0][0] + obj.Xi_gp_m[1][1]
Xi_std_gp_plus = np.sqrt(obj.xi_std_gp_m[0][0] ** 2 + obj.xi_std_gp_m[1][1] ** 2)

ftheta = 1.025
theta = [obj.r_corr_gp.meanr / ftheta, obj.r_corr_gp.meanr, obj.r_corr_gp.meanr * ftheta]
alpha_theta = [obj.Xi_gp_m[0][1], obj.Xi_gp_m[1][0], Xi_gp_plus]
yerr = [obj.xi_std_gp_m[0][1], obj.xi_std_gp_m[1][0], Xi_std_gp_plus]
labels = [r"$\Xi_{12}^{\rm gp}$", r"$\Xi_{21}^{\rm gp}$", r"$\Xi_+^{\rm gp}$"]
xlabel = r"$\theta$ [arcmin]"
ylabel = r"Centered correlation functions"
title = ""
out_path = f"{obj._params['output_dir']}/Xi_mixed_consistency.png"

cs_plots.plot_data_1d(
    theta,
    alpha_theta,
    yerr,
    title,
    xlabel,
    ylabel,
    out_path,
    labels=labels,
    xlog=True,
    xlim=xlim,
    close_fig=False,
)

# +
# For comparison, plot the same for the PSF - PSF correlations

xlim = [obj._params["theta_min_amin"], obj._params["theta_max_amin"]]

Xi_gp_plus = obj.Xi_pp_m[0][0] + obj.Xi_pp_m[1][1]
Xi_std_gp_plus = np.sqrt(obj.xi_std_pp_m[0][0] ** 2 + obj.xi_std_pp_m[1][1] ** 2)

ftheta = 1.025
theta = [obj.r_corr_gp.meanr / ftheta, obj.r_corr_gp.meanr, obj.r_corr_gp.meanr * ftheta]
alpha_theta = [obj.Xi_pp_m[0][1], obj.Xi_pp_m[1][0], Xi_gp_plus]
yerr = [obj.xi_std_pp_m[0][1], obj.xi_std_pp_m[1][0], Xi_std_gp_plus]
labels = [r"$\Xi_{12}^{\rm pp}$", r"$\Xi_{21}^{\rm pp}$", r"$\Xi_+^{\rm pp}$"]
xlabel = r"$\theta$ [arcmin]"
ylabel = r"Centered correlation functions"
title = ""
out_path = f"{obj._params['output_dir']}/Xi_pp_mixed_consistency.png"

cs_plots.plot_data_1d(
    theta,
    alpha_theta,
    yerr,
    title,
    xlabel,
    ylabel,
    out_path,
    labels=labels,
    xlog=True,
    xlim=xlim,
    close_fig=False,
)
# -


