import sys
sys.path.append('../shear_psf_leakage')

import numpy as np
import matplotlib.pyplot as plt
from rho_tau_stat import RhoStat, TauStat


#Run tests on shapepipe V1

path_gal = "~/Documents/UNIONS/data/unions_shapepipe_extended_2022_v1.1.fits"
path_psf = "~/Documents/UNIONS/data/unions_shapepipe_psf_2022_v1.0.2.fits"

rho_stat_handler = RhoStat(verbose=True)

rho_stat_handler.build_cat_to_compute_rho(path_psf, catalog_id='SPV1', square_size=True)