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
import matplotlib.pylab as plt
from astropy import units

from cs_util import plots as cs_plots

import shear_psf_leakage.run_scale as run
from shear_psf_leakage.leakage import *
# -

# ## Set input parameters

params_in = {}

# ### Input catalogues

# +
# Data directory
data_dir = "."
params_in["data_dir"] = data_dir

# Input galaxy shear catalogue
params_in["input_path_shear"] = f"{data_dir}/unions_shapepipe_extended_2022_W3_v1.0.3.fits"

# Input star/PSF catalogue
params_in["input_path_PSF"] = f"{data_dir}/unions_shapepipe_psf_2022_W3_v1.0.3.fits"

# Input galaxy redshift distribution (for theoretical model of xi_pm)
params_in["dndz_path"] = f"{data_dir}/dndz_SP_A.txt"
# -

# ### Other parameters

# +
# PSF ellipticty column names
params_in["e1_PSF_star_col"] = "E1_PSF_HSM"
params_in["e2_PSF_star_col"] = "E2_PSF_HSM"

# Set verbose output
params_in["verbose"] = True
# -

# ## Compute leakage

# Create leakage instance
obj = run.LeakageScale()

# Set instance parameters, copy from above
for key in params_in:
    obj._params[key] = params_in[key]

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
    params["dndz_path"],
)

# obj Plot ratio
obj.plot_xi_sys_ratio(xi_p_theo, xi_m_theo, close_fig=False)

obj.compute_corr_gp_pp_alpha_matrix()

obj.do_alpha_matrix()

# +
e1_gal = obj.dat_shear[obj._params["e1_col"]]                         
e2_gal = obj.dat_shear[obj._params["e2_col"]]                         
weights_gal = obj.dat_shear["w"]                                           
                                                                                
e1_star = obj.dat_PSF[obj._params["e1_PSF_star_col"]]                 
e2_star = obj.dat_PSF[obj._params["e2_PSF_star_col"]] 
# -

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
print(e_g, e_p)

e_gp = np.dot(e_g.transpose(), e_p)
e_pp = np.dot(e_p.transpose(), e_p)
print(e_pp)

# +
# Exact
r = obj.Xi_pp_m[0][1] ** 2 / (obj.Xi_pp_m[0][0] * obj.Xi_pp_m[1][1])

plt.semilogx(obj.r_corr_gp.meanr, r, label="$r$")
plt.semilogx(obj.r_corr_gp.meanr, 1 / (1 - r), label="$(1 - r)^{-1}$")
plt.semilogx(obj.r_corr_gp.meanr, 1 / (-1 + 1/r), label="$(-1 + r^{-1})^{-1}$")

# approximate
r_fast = obj.r_corr_pp_m[0][1].xip ** 2 / (obj.r_corr_pp_m[0][0].xip * obj.r_corr_pp_m[1][1].xip)

plt.semilogx(obj.r_corr_gp.meanr, r_fast, ":")
plt.semilogx(obj.r_corr_gp.meanr, 1 / (1 - r_fast), ":",)
plt.semilogx(obj.r_corr_gp.meanr, 1 / (-1 + 1/r_fast), ":")

plt.legend()

plt.ylim(-1, 4)

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


