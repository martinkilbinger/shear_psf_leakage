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
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# # PSF leakage: PSF auto-correlation contributions
#
# Martin Kilbinger <martin.kilbinger@cea.fr>

# +
import os
import matplotlib.pylab as plt

from cs_util import canfar
#from cs_util import plots
from uncertainties import ufloat
from uncertainties import unumpy                                                 

import shear_psf_leakage.run_object as run_object
import shear_psf_leakage.run_scale as run_scale
from shear_psf_leakage import leakage
# -

# ## Set input parameters

params_in = {}

# ### Paths

# +
# Input galaxy shear catalogue
params_in["input_path_shear"] = "unions_shapepipe_extended_2022_W3_v1.0.3.fits"

# Output directory
params_in["output_dir"] = "leakage_PSF_autocorr"
# -

# ### Job control

# Compute (spin-preserving) PSF ellipticity leakage
params_in["PSF_leakage"] = True

# ### Other parameters

# +
# PSF ellipticity column names
params_in["e1_PSF_col"] = "e1_PSF"
params_in["e2_PSF_col"] = "e2_PSF"

# Set verbose output
params_in["verbose"] = True
# -

# ### Retrieve test catalogue from VOspace if not yet downloade

vos_dir = "vos:cfis/XXXX/"
canfar.download(
    f"{vos_dir}/{params_in['input_path_shear']}",
    params_in["input_path_shear"],
    verbose=params_in["verbose"],
)

# ## Compute leakage

# Create leakage instance
obj_object = run_object.LeakageObject()

# Set instance parameters, copy from above
for key in params_in:
    obj_object._params[key] = params_in[key]

# Run commands as given in LeakageObject.run()

# +
# Check parameter validity
obj_object.check_params()

# Update parameters (here: strings to list)
obj_object.update_params()

# Prepare output directory
obj_object.prepare_output()
# -

# Read input catalogue
obj_object.read_data()

# Object-by-object spin-consistent PSF leakage
mix = True
order = "lin"
obj_object.PSF_leakage(mix=mix, order=order)

p_dp = {}                                                                    
for p in obj_object.par_best_fit:                                                         
    p_dp[p] = ufloat(obj_object.par_best_fit[p].value, obj_object.par_best_fit[p].stderr)
print(p_dp)

s_ds = leakage.param_order2spin(p_dp, order, mix)
print(s_ds)

# Get spin elements
x0 = s_ds["x0"]                                                                     
x4 = s_ds["x4"]
y4 = s_ds["y4"]
y0 = s_ds["y0"]


# Create scale-dependent leakage instance
obj_scale = run_scale.LeakageScale()

# Set instance parameters, copy from above                                   
for key in params_in:                                                        
    obj_scale._params[key] = params_in[key]


obj_scale._params["input_path_PSF"] = "unions_shapepipe_psf_2022_W3_v1.0.3.fits"
obj_scale._params["dndz_path"] = "dndz_SP_A.txt"

# ### Run                                                                        

# Check parameter validity                                                       
obj_scale.check_params()                                                               

# Prepare output directory and stats file                                        
obj_scale.prepare_output()                                                             

# Prepare input                                                                  
obj_scale.read_data()                                                                  

# #### Compute correlation function and alpha matrices                           
# The following command calls `treecorr` to compute auto- and cross-correlation functions.
# This can take a few minutes.                                                   

obj_scale.compute_corr_gp_pp_alpha_matrix()                                            

obj_scale.alpha_matrix()                                                               

#### For comparison: scalar alpha leakage                                        
obj_scale.compute_corr_gp_pp_alpha()                                                   
obj_scale.do_alpha()


# Plot terms
theta = obj_scale.get_theta()

term_1 = (x0 ** 2 + x4 **2) * (obj_scale.xi_pp_m[0][0] + obj_scale.xi_pp_m[1][1])
term_2 = 2 * x0 * x4 * (obj_scale.xi_pp_m[0][0] - obj_scale.xi_pp_m[1][1])
term_3a = x0 * (
    obj_scale.xi_pp_m[0][0] + 2 * obj_scale.xi_pp_m[0][1] + obj_scale.xi_pp_m[1][1]
)
term_3b = x4 * (obj_scale.xi_pp_m[0][0] - obj_scale.xi_pp_m[1][1])
term_3c = 2 * y4 * obj_scale.xi_pp_m[0][1] 
term_3 = y4 * (term_3a + term_3b + term_3c)

term_1

# +
y = [
    unumpy.nominal_values(term_1), 
    unumpy.nominal_values(term_2),
    unumpy.nominal_values(term_3),
]
dy = [
    unumpy.std_devs(term_1), 
    unumpy.std_devs(term_2),
    unumpy.std_devs(term_3),
]
x = [theta] * 3
title = "PSF auto-correlation additive terms"
xlabel = r"$\theta$ [arcmin]" 
ylabel = "terms"
out_path = f"{obj_scale._params['output_dir']}/auto_corr_terms.png"                                  
labels = ["t1" , "t2", "t3"]

plots.plot_data_1d(
    x,
    y,
    dy,
    title,
    xlabel,
    ylabel,
    out_path,
    labels=labels,
)
# -


