# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.15.2
#   kernelspec:
#     display_name: shear_psf_leakage
#     language: python
#     name: shear_psf_leakage
# ---

# # Scale-dependent PSF leakage
#
# ## Spin-consistent leakage
#
# package shear_psf_leakge
#
# Martin Kilbinger <martin.kilbinger@cea.fr>

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


# ### Set input parameters

# +
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
# -

print(obj._params)

# + [markdown] jp-MarkdownHeadingCollapsed=true
# ### Run
# -

# Check parameter validity
obj.check_params()

# Prepare output directory and stats file
obj.prepare_output()

# Prepare input
obj.read_data()

# #### Compute correlation function and alpha matrices
# The following command calls `treecorr` to compute auto- and cross-correlation functions.
# This can take a few minutes.

obj.compute_corr_gp_pp_alpha_matrix()

obj.alpha_matrix()

#### For comparison: scalar alpha leakage
obj.compute_corr_gp_pp_alpha()
obj.do_alpha()

# #### PSF auto-correlation function correlation matrix
#
# $$
# r = \frac{ \left( \Xi_{12}^\textrm{p,p} \right)^2 }
#     { \Xi_{11}^\textrm{p,p} \, \Xi_{22}^\textrm{p,p} }
# $$

# Check symmetry of PSF auto-correlation matrix
diff = obj.Xi_pp_m[0][1] - obj.Xi_pp_m[1][0]
print(
    "r is symmetrical? max abs (rel) diff ="
    +f" {max(np.abs(diff)):.3e} ({max(np.abs(diff / obj.Xi_pp_m[0][1])):.3e})",
)

# +
# Plot

# Exact calculation
r = obj.Xi_pp_m[0][1] ** 2 / (obj.Xi_pp_m[0][0] * obj.Xi_pp_m[1][1])

theta = obj.r_corr_gp_m[0][0].meanr

plt.semilogx(theta, r, label="$r$")
plt.semilogx(theta, 1 / (1 - r), label="$1/(1-r)$")
plt.semilogx(theta, r / (1 - r), label="$r/(1-r)$")

# Approximate
r_fast = obj.xi_pp_m[0][1] ** 2 / (obj.xi_pp_m[0][0] * obj.xi_pp_m[1][1])

plt.semilogx(theta, r_fast, "b:")
plt.semilogx(theta, 1 / (1 - r_fast), "o:",)
plt.semilogx(theta, 1 / (1/r_fast - 1), "g:")

plt.axhline(color="k", linewidth=0.5)

plt.legend()
plt.xlabe(r"$\theta$ [arcmin]")
plt.ylabel(r"$r(\theta)$")

plt.ylim(-0.5, 2)
plt.savefig("r.png")

# +
# Plot diagonal elements: Spin-0 and spin-4

x0 = 0.5 * ( obj.alpha_leak_m[0][0] + obj.alpha_leak_m[1][1] )
x4 = 0.5 * ( obj.alpha_leak_m[0][0] - obj.alpha_leak_m[1][1] )
dx0 = 0.5 * ( obj.alpha_leak_m[0][0] + obj.alpha_leak_m[1][1] )

plt.clf()
plt.semilogx(
    theta,
    x0,
    "-",
    label=r"$x_0 = (\alpha_{11} + \alpha_{22})/2$ (spin-0)"
)
plt.semilogx(
    theta,
    x4,
    "-",
    label=r"$x_4 = (\alpha_{11} - \alpha_{22})/2$ (spin-4)"
)
plt.legend()
xlim = [obj._params["theta_min_amin"], obj._params["theta_max_amin"]]
plt.xlim(xlim)
plt.xlabe(r"$\theta$ [arcmin]")
plt.ylabel(r"Components of leakage matrix")

plt.savefig("alpha_leakage_m_s0_s4.png")
# -


# The elements of $\alpha$ can be written as
# \begin{align}
# \newcommand{\mat}[1]{\mathrm{#1}}
#     \alpha_{11} = & \left(
#         \phantom -\Xi_{11}^\textrm{g,p} \, \Xi_{22}^\textrm{p,p}
#         - \Xi_{12}^\textrm{g,p} \, \Xi_{12}^\textrm{p,p}
#         \right) \left| \mat \Xi^\textrm{p,p} \right|^{-1} ;
#     \nonumber \\
#     \alpha_{12} = & \left(
#         - \Xi_{11}^\textrm{g,p} \, \Xi_{12}^\textrm{p,p}
#         + \Xi_{12}^\textrm{g,p} \, \Xi_{11}^\textrm{p,p}
#         \right) \left| \mat \Xi^\textrm{p,p} \right|^{-1} ;
#     \nonumber \\
#         \alpha_{21} = & \left(
#         \phantom - \Xi_{21}^\textrm{g,p} \, \Xi_{22}^\textrm{p,p}
#         - \Xi_{22}^\textrm{g,p} \, \Xi_{12}^\textrm{p,p}
#         \right) \left| \mat \Xi^\textrm{p,p} \right|^{-1} ;
#     \nonumber \\
#         \alpha_{22} = & \left(
#         - \Xi_{21}^\textrm{g,p} \, \Xi_{12}^\textrm{p,p}
#         + \Xi_{22}^\textrm{g,p} \, \Xi_{11}^\textrm{p,p}
#         \right) \left| \mat \Xi^\textrm{p,p} \right|^{-1},
# \end{align}

# +
# Check alpha_m elements equations (18)

plt.clf()

# a_11
color = "blue"

plt.semilogx(theta, obj.alpha_leak_m[0][0], "-", color=color, label=r"$\alpha_{11}$")

# Xi_pp determinant
D = obj.Xi_pp_m[0][0] * obj.Xi_pp_m[1][1] - obj.Xi_pp_m[0][1] * obj.Xi_pp_m[1][0]

a_11 = (
    (obj.Xi_gp_m[0][0] * obj.Xi_pp_m[1][1]
    - obj.Xi_gp_m[0][1] * obj.Xi_pp_m[1][0]) / D
)
plt.semilogx(theta, a_11, "s", label=r"$a_{11}$", color=color, mfc="none")

a_11_bis = (
    1 / (1 - r) * obj.Xi_gp_m[0][0] / obj.Xi_pp_m[0][0]
    - r / (1 - r) * obj.Xi_gp_m[0][1] / obj.Xi_pp_m[0][1]
)
plt.semilogx(theta, a_11_bis, "p", label=r"$a_{11}$ bis", color=color, mfc="none")

# a_22
color = "green"

plt.semilogx(theta, obj.alpha_leak_m[1][1], "--", color=color, label=r"$\alpha_{22}$")

a_22 = (
    (obj.Xi_gp_m[1][1] * obj.Xi_pp_m[0][0]
    - obj.Xi_gp_m[1][0] * obj.Xi_pp_m[0][1]) / D
)
plt.semilogx(theta, a_22, "d", label=r"$a_{22}$", color=color, mfc="none")

a_22_bis = (
     1 / (1 - r) * obj.Xi_gp_m[1][1] / obj.Xi_pp_m[1][1]
    - r / (1 - r) * obj.Xi_gp_m[1][0] / obj.Xi_pp_m[1][0]
)
plt.semilogx(theta, a_22_bis, "o", label=r"$a_{22}$ bis", color=color, mfc="none")

plt.legend()
plt.xlabel(r"$\theta$ [arcmin]")
plt.ylabel(r"$\alpha_{ii}$")

plt.savefig("alpha_m_11_22_check.png")


# +
plt.clf()

# a_12
color = "magenta"

plt.semilogx(theta, obj.alpha_leak_m[0][1], "-", color=color, label=r"$\alpha_{12}$")

a_12 = (
    (-obj.Xi_gp_m[0][0] * obj.Xi_pp_m[0][1]
    + obj.Xi_gp_m[0][1] * obj.Xi_pp_m[0][0]) / D
)
plt.semilogx(theta, a_12, "s", label=r"$a_{12}$", color=color, mfc="none")

a_12_bis = (
    -r / (1 - r) * obj.Xi_gp_m[0][0] / obj.Xi_pp_m[0][1]
    + 1 / (1 - r) * obj.Xi_gp_m[0][1] / obj.Xi_pp_m[1][1]
)
plt.semilogx(theta, a_12_bis, "p", label=r"$a_{11}$ bis", color=color, mfc="none")

# a_21
color = "green"

plt.semilogx(theta, obj.alpha_leak_m[1][0], "--", color=color, label=r"$\alpha_{21}$")

a_21 = (
    (obj.Xi_gp_m[1][0] * obj.Xi_pp_m[1][1]
    - obj.Xi_gp_m[1][1] * obj.Xi_pp_m[0][1]) / D
)
plt.semilogx(theta, a_21, "d", label=r"$a_{22}$", color=color, mfc="none")

a_21_bis = (
     1 / (1 - r) * obj.Xi_gp_m[1][0] / obj.Xi_pp_m[0][0]
    - r / (1 - r) * obj.Xi_gp_m[1][1] / obj.Xi_pp_m[0][1]
)
plt.semilogx(theta, a_21_bis, "o", label=r"$a_{22}$ bis", color=color, mfc="none")

plt.legend()
plt.xlabel(r"$\theta$ [arcmin]")
plt.ylabel(r"$\alpha_{ij}$")

plt.savefig("alpha_m_12_21_check.png")

# +
# Check alpha_m elements

plt.clf()
plt.semilogx(theta, obj.alpha_leak_m[0][0] + obj.alpha_leak_m[1][1], "-", label=r"tr $\mathbf{\alpha}$")

plt.semilogx(theta, a_11 + a_22, "s", label=r"$a_{11} + a_{22}$", mfc="none")

#plt.semilogx(theta, a_11_bis + a_22_bis, "p", label=r"$a_{11} + a_{22}$ bis", mfc="none")

t_sym = (
    obj.Xi_gp_m[0][0] / ( obj.Xi_pp_m[0][0] * (1 - r) )
    + obj.Xi_gp_m[1][1] / ( obj.Xi_pp_m[1][1] * (1 - r) )
)
plt.semilogx(theta, t_sym, 'v', label=r"$\alpha$ sym")

t_asym = (
    - obj.Xi_gp_m[0][1] / ( obj.Xi_pp_m[0][1] * (-1 + 1/r) )
    - obj.Xi_gp_m[1][0] / ( obj.Xi_pp_m[1][0] * (-1 + 1/r) )
)
plt.semilogx(theta, t_asym, '^', label=r"$\alpha$ asym")

plt.semilogx(theta, t_sym + t_asym, '.', label=r"$\alpha$ sym + asym")


plt.semilogx(theta, obj.alpha_leak * 2, ':', label=r"$2 \alpha$")

plt.legend(bbox_to_anchor=(1.2, 0.5))
_ = plt.ylim(-0.25, 0.25)

plt.savefig("x.png")
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
theta_arr = [theta / ftheta, theta, theta * ftheta]
alpha_theta = [obj.alpha_leak, alpha_1, alpha_2]
yerr = [obj.sig_alpha_leak, alpha_1_std, alpha_2_std]
labels = [r"$\alpha$", r"$\alpha_1$", r"$\alpha_2$"]
xlabel = r"$\theta$ [arcmin]"
ylabel = r"$\alpha(\theta)$"
title = ""
out_path = f"{obj._params['output_dir']}/alpha_leakage_scalar_consistency.png"

cs_plots.plot_data_1d(
    theta_arr,
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
theta_arr = [theta / ftheta, theta, theta * ftheta]
alpha_theta = [obj.Xi_gp_m[0][1], obj.Xi_gp_m[1][0], Xi_gp_plus]
yerr = [obj.xi_std_gp_m[0][1], obj.xi_std_gp_m[1][0], Xi_std_gp_plus]
labels = [r"$\Xi_{12}^{\rm gp}$", r"$\Xi_{21}^{\rm gp}$", r"tr$\Xi^{\rm gp}$"]
xlabel = r"$\theta$ [arcmin]"
ylabel = r"Centered correlation functions"
title = ""
out_path = f"{obj._params['output_dir']}/Xi_mixed_consistency.png"

cs_plots.plot_data_1d(
    theta_arr,
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
theta_arr = [theta / ftheta, theta, theta * ftheta]
alpha_theta = [obj.Xi_pp_m[0][1], obj.Xi_pp_m[1][0], Xi_gp_plus]
yerr = [obj.xi_std_pp_m[0][1], obj.xi_std_pp_m[1][0], Xi_std_gp_plus]
labels = [r"$\Xi_{12}^{\rm pp}$", r"$\Xi_{21}^{\rm pp}$", r"tr$\Xi^{\rm pp}$"]
xlabel = r"$\theta$ [arcmin]"
ylabel = r"Centered correlation functions"
title = ""
out_path = f"{obj._params['output_dir']}/Xi_pp_mixed_consistency.png"

cs_plots.plot_data_1d(
    theta_arr,
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


