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
#     display_name: shear_psf_leakage2
#     language: python
#     name: shear_psf_leakage2
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
from uncertainties import unumpy

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

# ### Run

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

theta = obj.get_theta()

# Exact calculation
r = []
r_ratio_1 = []
r_ratio_2 = []
for ndx in range(len(theta)):
    my_r = (
        obj.Xi_pp_ufloat[ndx][0, 1] ** 2
        / (obj.Xi_pp_ufloat[ndx][0, 0] * obj.Xi_pp_ufloat[ndx][1, 1])
    )
    r.append(my_r)
    r_ratio_1.append(1 / (1 - my_r))
    r_ratio_2.append(my_r / (1 - my_r)) 

print("min max mean r = ", np.min(r), np.max(r), np.mean(r))

# Approximate
r_fast = obj.xi_pp_m[0][1] ** 2 / (obj.xi_pp_m[0][0] * obj.xi_pp_m[1][1])

theta_arr = []
r_arr = []
dr_arr = []

n = 6
ftheta = 1.05
for idx in range(n):
    theta_arr.append(theta * ftheta ** (idx - n))
    
r_arr.append(unumpy.nominal_values(r))
r_arr.append(unumpy.nominal_values(r_ratio_1))
r_arr.append(unumpy.nominal_values(r_ratio_2))
r_arr.append(r_fast)
r_arr.append(1 / (1 - r_fast))
r_arr.append(r_fast / (1 - r_fast))

dr_arr.append(unumpy.std_devs(r))
dr_arr.append(unumpy.std_devs(r_ratio_1))
dr_arr.append(unumpy.std_devs(r_ratio_2))
for idx in range(3):
    dr_arr.append(np.nan)

labels = ["$r$", "$1/(1-r)$", "$r/(1-r)$", "", "", ""]
colors = ["blue", "orange", "green", "blue", "orange", "green"]
linestyles = ["-"] * 3 + ["--"] * 3
linewidths = [2] * 3 + [1] * 3

xlabel = r"$\theta$ [arcmin]"
ylabel = r"ratios of $r(\theta)$"

xlim = [obj._params["theta_min_amin"], obj._params["theta_max_amin"]]
ylim = (-0.5, 2)

out_path = f"{obj._params['output_dir']}/r.png"

title = ""

cs_plots.plot_data_1d(
    theta_arr,
    r_arr,
    dr_arr,
    title,
    xlabel,
    ylabel,
    out_path,
    labels=labels,
    xlog=True,
    xlim=xlim,
    ylim=ylim,
    colors=colors,
    linewidths=linewidths,
    linestyles=linestyles,
    close_fig=False,
)

# +
# Plot alpha matrix elements

theta = obj.get_theta()

xlim = [obj._params["theta_min_amin"], obj._params["theta_max_amin"]]
ylim = obj._params["leakage_alpha_ylim"]

n = 4
ftheta = 1.05
theta_arr = []
for idx in (range(n)):
    theta_arr.append(theta * ftheta ** (idx - n))
    
alpha = []
yerr = []
labels = []
for idx in (0, 1):
    for jdx in (0, 1):   
        alpha_ufloat = obj.get_alpha_ufloat(idx, jdx)
        alpha.append(unumpy.nominal_values(alpha_ufloat))
        yerr.append(unumpy.std_devs(alpha_ufloat))
        labels.append(rf"$\alpha_{{{idx+1}{jdx+1}}}$")

colors = ["blue", "orange", "orange", "green"]
linestyles = ["-", "-", "--", "-"]
markers = ["o", "^", "v", "s"]

xlabel = r"$\theta$ [arcmin]"
ylabel = r"$\alpha_{ij}(\theta)$"
title = ""
out_path = f"{obj._params['output_dir']}/alpha_ij.png"

cs_plots.plot_data_1d(
    theta_arr,
    alpha,
    yerr,
    title,
    xlabel,
    ylabel,
    out_path,
    labels=labels,
    xlog=True,
    xlim=xlim,
    ylim=ylim,
    colors=colors,
    linestyles=linestyles,
    markers=markers,
    close_fig=False,
)

# +
# Plot spin elements: Spin-0 and spin-4

x0 = 0.5 * (
     obj.get_alpha_ufloat(0, 0)
    + obj.get_alpha_ufloat(1, 1)
)
x4 = 0.5 * (
    obj.get_alpha_ufloat(0, 0)
    - obj.get_alpha_ufloat(1, 1)
)

y4 = 0.5 * (obj.get_alpha_ufloat(0, 1) + obj.get_alpha_ufloat(1, 0))
y0 = 0.5 * (obj.get_alpha_ufloat(1, 0) - obj.get_alpha_ufloat(0, 1))

theta = obj.get_theta()

xlim = [obj._params["theta_min_amin"], obj._params["theta_max_amin"]]
ylim = obj._params["leakage_alpha_ylim"]

n = 4
ftheta = 1.05
theta_arr = []
for idx in (range(n)):
    theta_arr.append(theta * ftheta ** (idx - n/2))

y_arr = [
    unumpy.nominal_values(x0),
    unumpy.nominal_values(x4),
    unumpy.nominal_values(y4),
    unumpy.nominal_values(y0),
]
dy_arr = [
    unumpy.std_devs(x0),
    unumpy.std_devs(x4),
    unumpy.std_devs(y4),
    unumpy.std_devs(y0),
]
labels = [
    r"$x_0 = (\alpha_{11} + \alpha_{22})/2$ (spin-0)",
    r"$x_4 = (\alpha_{11} - \alpha_{22})/2$ (spin-4)",
    r"$y_4 = (\alpha_{12} + \alpha_{21})/2$ (spin-4)",
    r"$y_0 = (\alpha_{21} - \alpha_{12})/2$ (spin-0 imaginary)"
]
colors = ["blue", "orange", "green", "magenta"]
markers = ["o", "s", "h", "^"]
linestyles = ["-"] * 4 

xlabel = r"$\theta$ [arcmin]"
ylabel = r"Components of leakage matrix"
title = ""
out_path = "alpha_leakage_m_s0_s4.png"

cs_plots.plot_data_1d(
    theta_arr,
    y_arr,
    dy_arr,
    title,
    xlabel,
    ylabel,
    out_path,
    labels=labels,
    xlog=True,
    xlim=xlim,
    ylim=ylim,
    colors=colors,
    markers=markers,
    close_fig=False,
)

# Including scalar leakage for comparison
theta_arr.append(theta * ftheta ** (4 - n/2))
y_arr.append(obj.alpha_leak)
dy_arr.append(obj.sig_alpha_leak)
labels.append(r"$\alpha$ (scalar approx.)")
colors.append("blue")
markers.append("p")
linestyles.append("--")
out_path = "alpha_leakage_m_s0_s4_as.png"

cs_plots.plot_data_1d(
    theta_arr,
    y_arr,
    dy_arr,
    title,
    xlabel,
    ylabel,
    out_path,
    labels=labels,
    xlog=True,
    xlim=xlim,
    ylim=ylim,
    colors=colors,
    markers=markers,
    linestyles=linestyles,
    close_fig=False,
)


# +
# Plot off-diagonal element: spin-4 and "spin-0 imaginary" part which should vanish

y4 = obj.get_alpha_ufloat(0, 1)
y4_bis = obj.get_alpha_ufloat(1, 0)
y0 = 0.5 * (y4_bis - y4)

theta = obj.get_theta()

xlim = [obj._params["theta_min_amin"], obj._params["theta_max_amin"]]
ylim = obj._params["leakage_alpha_ylim"]

n = 3
ftheta = 1.05
theta_arr = []
for idx in (range(n)):
    theta_arr.append(theta * ftheta ** (idx - n))
    
y = [
    unumpy.nominal_values(y4),
    unumpy.nominal_values(y4_bis),
    unumpy.nominal_values(y0),
]
dy = [
    unumpy.std_devs(y4),
    unumpy.std_devs(y4_bis),
    unumpy.std_devs(y0),
]
labels = [
    r"$\alpha_{12}$ (spin-4)",
    r"$\alpha_{21}$ (spin-4)",
    r"$y_0 = (\alpha_{21}-\alpha_{12})/2$ ('spin-0 imag')",
]
xlim = [obj._params["theta_min_amin"], obj._params["theta_max_amin"]]
ylim = obj._params["leakage_alpha_ylim"]

colors = ["orange", "orange", "magenta"]
markers = ["o", "s", "^"]

xlabel = r"$\theta$ [arcmin]"
ylabel = r"Components of leakage matrix"
title = ""
out_path = "alpha_leakage_m_s4_s4im.png"

cs_plots.plot_data_1d(
    theta_arr,
    y,
    dy,
    title,
    xlabel,
    ylabel,
    out_path,
    labels=labels,
    xlog=True,
    xlim=xlim,
    ylim=ylim,
    colors=colors,
    markers=markers,
    close_fig=False,
)
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

# #### Consistency relations for scalar leakage

# If the leakage is a scalar function, it can be expressed in three different ways.

# +
xlim = [obj._params["theta_min_amin"], obj._params["theta_max_amin"]]
ylim = obj._params["leakage_alpha_ylim"]

theta = obj.get_theta()

ftheta = 1.05
theta_arr = [theta / ftheta, theta, theta * ftheta]

alpha_1 = []
alpha_2 = []
for ndx in range(len(theta)):
    my_a1 = obj.Xi_gp_ufloat[ndx][0, 0] / obj.Xi_pp_ufloat[ndx][0, 0]
    alpha_1.append(my_a1)
    my_a2 = obj.Xi_gp_ufloat[ndx][1, 1] / obj.Xi_pp_ufloat[ndx][1, 1]
    alpha_2.append(my_a2)

y = [
    obj.alpha_leak,
    unumpy.nominal_values(alpha_1),
    unumpy.nominal_values(alpha_2),
]
dy = [
    obj.sig_alpha_leak,
    unumpy.std_devs(alpha_1),
    unumpy.std_devs(alpha_2),
]

labels = [r"$\alpha$", r"$\alpha_1$", r"$\alpha_2$"]
xlabel = r"$\theta$ [arcmin]"
ylabel = r"$\alpha(\theta)$"
title = ""
out_path = f"{obj._params['output_dir']}/alpha_leakage_scalar_consistency.png"

cs_plots.plot_data_1d(
    theta_arr,
    y,
    dy,
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

ftheta = 1.025
theta_arr = [theta / ftheta, theta, theta * ftheta]

Xi_12 = []
Xi_21 = []
Xi_tr = []
for ndx in range(len(theta)):
    Xi_12.append(obj.Xi_gp_ufloat[ndx][0, 1])
    Xi_21.append(obj.Xi_gp_ufloat[ndx][1, 0])
    Xi_tr.append(obj.Xi_gp_ufloat[ndx][0, 0] + obj.Xi_gp_ufloat[ndx][0, 0])

y = [
    unumpy.nominal_values(Xi_12),
    unumpy.nominal_values(Xi_21),
    unumpy.nominal_values(Xi_tr),
]
dy = [
    unumpy.std_devs(Xi_12),
    unumpy.std_devs(Xi_21),
    unumpy.std_devs(Xi_tr),
]


labels = [r"$\Xi_{12}^{\rm gp}$", r"$\Xi_{21}^{\rm gp}$", r"tr$\Xi^{\rm gp}$"]
xlabel = r"$\theta$ [arcmin]"
ylabel = r"Centered correlation functions"
title = ""
out_path = f"{obj._params['output_dir']}/Xi_mixed_consistency.png"

cs_plots.plot_data_1d(
    theta_arr,
    y,
    dy,
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

ftheta = 1.025
theta_arr = [theta / ftheta, theta, theta * ftheta]

Xi_12 = []
Xi_21 = []
Xi_tr = []
for ndx in range(len(theta)):
    Xi_12.append(obj.Xi_pp_ufloat[ndx][0, 1])
    Xi_21.append(obj.Xi_pp_ufloat[ndx][1, 0])
    Xi_tr.append(obj.Xi_pp_ufloat[ndx][0, 0] + obj.Xi_pp_ufloat[ndx][0, 0])

y = [
    unumpy.nominal_values(Xi_12),
    unumpy.nominal_values(Xi_21),
    unumpy.nominal_values(Xi_tr),
]
dy = [
    unumpy.std_devs(Xi_12),
    unumpy.std_devs(Xi_21),
    unumpy.std_devs(Xi_tr),
]


labels = [r"$\Xi_{12}^{\rm pp}$", r"$\Xi_{21}^{\rm pp}$", r"tr$\Xi^{\rm pp}$"]
xlabel = r"$\theta$ [arcmin]"
ylabel = r"Centered correlation functions"
title = ""
out_path = f"{obj._params['output_dir']}/Xi_pp_mixed_consistency.png"

cs_plots.plot_data_1d(
    theta_arr,
    y,
    dy,
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
