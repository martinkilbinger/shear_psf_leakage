import sys
sys.path.append('../shear_psf_leakage')

import numpy as np
import matplotlib.pyplot as plt
from rho_tau_stat import RhoStat, TauStat


#Run tests on shapepipe V1 and DESY3

#First specify  the paths to your data
paths_gal = [
    "/n17data/mkilbing/astro/data/CFIS/v1.0/ShapePipe/unions_shapepipe_extended_2022_v1.1.fits",
    "/home/mkilbing/astro/data/DES/DES_Y3_cut.fits",
    "/home/hervas/n23_fhervas/shapepipe_clean_uppercut/unions_shapepipe_extended_2022_v1.0.4_theli_4096.fits"
]
paths_psf = [
    "/n17data/mkilbing/astro/data/CFIS/v1.0/ShapePipe/unions_shapepipe_psf_2022_v1.0.2.fits",
    "/home/mkilbing/astro/data/DES/psf_y3a1-v29.fits",
    "/n17data/mkilbing/astro/data/CFIS/v1.0/ShapePipe/unions_shapepipe_psf_2022_v1.0.2.fits"
]

#Specify if you want to mask some data using flags
masks = [
    True,
    False,
    True
]

#Specify is the sizes have to be squared. It can depend on the catalog.
square_sizes = [
    True,
    False,
    True
]

#Contains the params of your catalog. Don't forget to specify them since they can vary between different catalogs
params_des = {
    "e1_col": "e1_cal",
    "e2_col": "e2_cal",
    "w_col": "w",
    "ra_col": "RA",
    "dec_col": "Dec",
    "e1_PSF_col": "piff_e1",
    "e2_PSF_col": "piff_e2",
    "e1_star_col": "obs_e1",
    "e2_star_col": "obs_e2",
    "PSF_size": "piff_T",
    "star_size": "obs_T",
    "output_dir": ".",
    "patch_number": 120,
    "ra_units": "deg",
    "dec_units": "deg" 
}

params = [
    None,
    params_des,
    None
]

colors = ['blue', 'red', 'green'] #Colors for the plot
catalog_ids = ['SPV1', 'DESY3', 'SPV104'] #Ids of the catalogs

rho_stat_handler = RhoStat(verbose=True) #Create your class to compute, save, load and plot rho_stats

for path_gal, path_psf, cat_id, param, mask, square_size in zip(paths_gal, paths_psf, catalog_ids, params, masks, square_sizes): #Iterate on the different catalogs

    if param is None:
        rho_stat_handler.catalogs.params_default()
    else:
        rho_stat_handler.catalogs.set_params(param) #Set the right parameters
    rho_stat_handler.build_cat_to_compute_rho(path_psf, catalog_id=cat_id, square_size=square_size, mask=mask) #Build the different catalogs
    rho_stat_handler.compute_rho_stats(cat_id, 'rho_stats_'+cat_id+'.fits') #Compute and save the rho statistics

filenames = ['rho_stats_'+cat_id+'.fits' for cat_id in catalog_ids]

rho_stat_handler.plot_rho_stats(filenames, colors, catalog_ids, abs=True, savefig='rho_stats.png') #Plot

tau_stat_handler = TauStat(catalogs=rho_stat_handler.catalogs, verbose=True) #Create your class to compute, save, load and plot tau_stats

for path_gal, path_psf, cat_id, param, mask, square_size in zip(paths_gal, paths_psf, catalog_ids, params, masks, square_sizes): #Iterate on the catalogs

    if param is None:
        tau_stat_handler.catalogs.params_default()
    else:
        tau_stat_handler.catalogs.set_params(param) #Set the parameters
    tau_stat_handler.build_cat_to_compute_tau(path_gal, cat_type='gal', catalog_id=cat_id, square_size=square_size, mask=mask) #Build the catalog of galaxies. PSF was computed above
    tau_stat_handler.compute_tau_stats(cat_id, 'tau_stats_'+cat_id+'.fits') #Compute and save the tau statistics


filenames = ['tau_stats_'+cat_id+'.fits' for cat_id in catalog_ids]
tau_stat_handler.plot_tau_stats(filenames, colors, catalog_ids, savefig='tau_stats.png', plot_tau_m=False) #Plot