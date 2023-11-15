import sys
sys.path.append('../shear_psf_leakage')

import numpy as np
import matplotlib.pyplot as plt
from rho_tau_stat import RhoStat, TauStat


#Run tests on shapepipe V1

path_gal = "/n17data/mkilbing/astro/data/CFIS/v1.0/ShapePipe/unions_shapepipe_extended_2022_v1.1.fits"
path_psf = "/n17data/mkilbing/astro/data/CFIS/v1.0/ShapePipe/unions_shapepipe_psf_2022_v1.0.2.fits"

rho_stat_handler = RhoStat(verbose=True)

rho_stat_handler.build_cat_to_compute_rho(path_psf, catalog_id='SPV1', square_size=True)

rho_stat_handler.compute_rho_stats('SPV1', 'rho_stats_spv1.fits')


tau_stat_handler = TauStat(catalogs=rho_stat_handler.catalogs, verbose=True)

tau_stat_handler.build_cat_to_compute_tau(path_gal, cat_type='gal', catalog_id='SPV1', square_size=True)

tau_stat_handler.compute_tau_stats('SPV1', 'tau_stats_spv1.fits')