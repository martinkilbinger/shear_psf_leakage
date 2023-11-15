"""
This module sets up a class to compute the rho stats computation

Author: Sacha Guerrini
"""

import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.table import Table

import treecorr

class Catalogs():
    """
    Catalogs

    Class to build the different treecorr catalogs given a shape catalog that will be 
    used to compute the different statistics
    """

    def __init__(self, params=None, output=None):
        #set default parameters
        if (params is None) or (output is None): 
            self.params_default()
        else:
            self.set_params(params, output)

        self.catalogs_dict = dict()
        self.dat_shear = None
        self.dat_psf = None

    def params_default(self):
        """
        Params Default.

        Initialize the parameters of the class with columns name from SPV1.
        For the treecorr configuration, default parameters are:
        -coord_units: degree
        -sep_units: arcmin
        -theta_min: 0.1
        -theta_max: 100
        -n_theta: 20
        -var_method: jackknife !!Requires to set a patch number for the different catalogues!!
        """

        self._params = {
            "e1_col": "e1",
            "e2_col": "e2",
            "w_col": "w",
            "ra_col": "RA",
            "dec_col": "Dec",
            "e1_PSF_col": "E1_PSF_HSM",
            "e2_PSF_col": "E2_PSF_HSM",
            "e1_star_col": "E1_STAR_HSM",
            "e2_star_col": "E2_STAR_HSM",
            "PSF_size": "SIGMA_PSF_HSM",
            "star_size": "SIGMA_STAR_HSM",
            "PSF_flag": "FLAG_PSF_HSM",
            "star_flag": "FLAG_STAR_HSM",
            "output_dir": ".",
            "patch_number": 120,
            "ra_units": "deg",
            "dec_units": "deg" 
        }

        self._output = "."

    def set_params(self, params, output):
        """
        set_params:

        Initialize the parameters to the given configuration. Can also update the current params.
        """
        if params is not None:
            self._params = params
        if output is not None:
            self._output = output

    def read_shear_cat(self, path_gal, path_psf):
        """
        read_shear_cat

        Read a shear catalogue with galaxies ('gal') or stars ('psf').
        Only one such catalogue can be loaded at a time.

        Raises
        ------
        AssertionError: Please specify a path for the shear catalog you want to read.
        """
        assert ((path_gal is not None) or (path_psf is not None)), ("Please specify a path for the shear catalog you want to read.")
        if path_gal is not None:
            self.dat_shear = fits.getdata(path_gal)
        if path_psf is not None:
            self.dat_psf = fits.getdata(path_psf)
    
    def get_cat_fields(self, cat_type, square_size=False):
        """
        Get Cat Fields

        Get catalogue fields for correlation.

        Parameters
        ----------
        cat_type : str
            catalogue type, allowed are 'gal', 'psf', 'psf_error' or 'psf_size_error'

        square_size : bool
            If True, the size computed in the catalogue is squared (Default: False)

        Returns
        -------
        np.array
            ra
        np.array
            dec
        np.array
            e1
        np.array
            e2
        np.array
            weights; 'None' is cat_type is not 'gal'. Returns a one_like array if weights are not specified

        Raises
        ------
        AssertionError
            If the specified cat_type does not belong to the allowed list.
        """

        allowed_types = ['gal', 'psf', 'psf_error', 'psf_size_error']

        assert cat_type in allowed_types, ("The specified catalogue type is invalid. Check the one you use is allowed."
                                           "Allowed cat_type: 'gal', 'psf', 'psf_error', 'psf_size_error'.")
        
        if cat_type=="gal":
            assert self.dat_shear is not None, ("Check you read the shear catalogs correctly.")
            ra = self.dat_shear[self._params["ra_col"]]
            dec = self.dat_shear[self._params["dec_col"]]
            g1 = self.dat_shear[self._params["e1_col"]] - self.dat_shear[self._params["e1_col"]].mean()
            g2 = self.dat_shear[self._params["e2_col"]] - self.dat_shear[self._params["e2_col"]].mean()
            if self._params["w_col"] is not None:
                weights = self.dat_shear[self._params["w_col"]]
            else:
                weights = np.ones.like(ra)
        else:
            assert self.dat_psf is not None, ("Check you read the shear catalogs correctly.")
            #Add a mask?
            #mask = (self.dat_psf[self._params["FLAG_PSF_HSM"]]==0) & (self.dat_psf[self._params["FLAG_STAR_HSM"]]==0)
            ra = self.dat_psf[self._params["ra_col"]]
            dec = self.dat_psf[self._params["dec_col"]]
            weights = None
            
            if cat_type=="psf":
                g1 = self.dat_psf[self._params["e1_PSF_col"]] - self.dat_psf[self._params["e1_PSF_col"]].mean()
                g2 = self.dat_psf[self._params["e2_PSF_col"]] - self.dat_psf[self._params["e2_PSF_col"]].mean()
            
            elif cat_type=="psf_error":
                g1 = (self.dat_psf[self._params["e1_star_col"]] - self.dat_psf[self._params["e1_PSF_col"]])
                g1 -= g1.mean()
                g2 = (self.dat_psf[self._params["e2_star_col"]] - self.dat_psf[self._params["e2_PSF_col"]])
                g2 -= g2.mean()

            else:
                size_star = self.dat_psf[self._params["star_size"]]**2 if square_size else  self.dat_psf[self._params["star_size"]]
                size_psf = self.dat_psf[self._params["PSF_size"]]**2 if square_size else  self.dat_psf[self._params["PSF_size"]]
                g1 = self.dat_psf[self._params["e1_star_col"]] * (size_star - size_psf)/size_psf
                g2 = self.dat_psf[self._params["e2_star_col"]] * (size_star - size_psf)/size_psf

            return ra, dec, g1, g2, weights
    
    def build_catalog(self, cat_type, key, npatch=None, square_size=False):
        """
        build_catalogue

        Build a treecorr.Catalog of a certain type using the class _params. A key is given as input
        to identify the catalog in self.catalogs_dict.

        Parameters
        ----------
        cat_type : str
            catalogue type, allowed are 'gal', 'psf', 'psf_error' or 'psf_size_error'.

        key : str
            String used to key the catalog to an entry of the dict of catalogs.

        npatch : int
            number of patch used to compute variance with jackknife or bootstrap. (Default: value in self._params)

        square_size : bool
            If True, the size computed in the catalogue is squared (Default: False)
        """

        if npatch is None:
            npatch = self._params["patch_number"]
        
        ra, dec, g1, g2, weights = self.get_cat_fields(cat_type, square_size)

        cat = treecorr.Catalog(
            ra=ra,
            dec=dec,
            g1=g1,
            g2=g2,
            w=weights,
            ra_units=self._params["ra_units"],
            dec_units=self._params["dec_units"],
            npatch=npatch
        )

        self.catalogs_dict.update(
            {key: cat}
        )

    def delete_catalog(self, key):
        """
        delete_catalog

        Delete the catalog instance mapped by the string key.
        """

        try:
            self.catalogs_dict.pop(key)
        except KeyError:
            print("This entry did not exist.")

    def show_catalogs(self):
        """
        show_catalogs

        Print the keys of the element stored in self.catalogs_dict
        """
        for key in self.catalogs_dict.keys():
            print(key)   


class RhoStat():
    """
    RhoStat

    Class to compute the rho statistics (Rowe 2010) of a PSF catalogue.
    """

    def __init__(self, params=None, output=None, treecorr_config=None, verbose=False):

        self.catalogs = Catalogs(params, output)

        if treecorr_config is None:
            self._treecorr_config = {
                "ra_units": "deg",
                "dec_units": "deg",
                "sep_units": "arcmin",
                "min_sep": 0.1,
                "max_sep": 100,
                "n_bins": 20,
                "var_method": "jackknife"
            }
        else:
            self._treecorr_config = treecorr_config

        self.verbose = verbose
    

    def check_params(self):
        """Check Params.

        Check whether parameter values are valid.

        Raises
        ------
        ValueError
            if a parameter value is not valid

        """
        pass
        #TO DO

    def build_cat_to_compute_rho(self, path_cat_star, catalog_id='', square_size=False):
        """
        build_cat_to_compute_rho

        Parameters
        ----------
        path_cat_star : str
            Path to the catalog of stars used to compute the rho-statistics.

        catalog_id : str
            An id to identify the catalog used in the keys of the stored treecorr.Catalog.

        square_size : bool
            If True, the size computed in the catalogue is squared (Default: False)
        """

        self.catalogs.read_shear_cat(path_gal=None, path_psf=path_cat_star)

        if self.verbose:
            print("Building catalogs...")

        self.catalogs.build_catalog(cat_type='psf', key='psf_'+catalog_id, square_size=square_size)
        self.catalogs.build_catalog(cat_type='psf_error', key='psf_error_'+catalog_id, square_size=square_size)
        self.catalogs.build_catalog(cat_type='psf_size_error', key='psf_size_error_'+catalog_id, square_size=square_size)

        if self.verbose:
            print("Catalogs successfully built...")
            self.catalogs.show_catalogs()

    def compute_rho_stats(self, catalog_id, filename):
        """
        compute_rho_stats

        Compute the rho statistics of your psf. Store it as an attribute of the class which could be save afterwards.

        Parameters:
        ----------
        catalog_id : str
            The id of the catalog used to compute the rho statistics.

        filename : str
            The path where the rho stats will be saved.
        """
        if self.verbose:
            print("Computation of the rho statistics in progress...")
        rho_0 = treecorr.GGCorrelation(self._treecorr_config)
        rho_0.process(self.catalogs['psf_'+catalog_id], self.catalogs['psf_'+catalog_id])
        rho_1 = treecorr.GGCorrelation(self._treecorr_config)
        rho_1.process(self.catalogs['psf_error_'+catalog_id], self.catalogs['psf_error_'+catalog_id])
        rho_2 = treecorr.GGCorrelation(self._treecorr_config)
        rho_2.process(self.catalogs['psf_'+catalog_id], self.catalogs['psf_error_'+catalog_id])
        rho_3 = treecorr.GGCorrelation(self._treecorr_config)
        rho_3.process(self.catalogs['psf_size_error_'+catalog_id], self.catalogs['psf_size_error_'+catalog_id])
        rho_4 = treecorr.GGCorrelation(self._treecorr_config)
        rho_4.process(self.catalogs['psf_error_'+catalog_id], self.catalogs['psf_size_error_'+catalog_id])
        rho_5 = treecorr.GGCorrelation(self._treecorr_config)
        rho_5.process(self.catalogs['psf_'+catalog_id], self.catalogs['psf_size_error_'+catalog_id])

        self.rho_stats = Table(
            [
                rho_0.rnom,
                rho_0.xip,
                rho_0.varxip,
                rho_0.xim,
                rho_1.varxim,
                rho_1.xip,
                rho_1.varxip,
                rho_1.xim,
                rho_1.varxim,
                rho_2.xip,
                rho_2.varxip,
                rho_2.xim,
                rho_2.varxim,
                rho_3.xip,
                rho_3.varxip,
                rho_3.xim,
                rho_3.varxim,
                rho_4.xip,
                rho_4.varxip,
                rho_4.xim,
                rho_4.varxim,
                rho_5.xip,
                rho_5.varxip,
                rho_5.xim,
                rho_5.varxim,
            ],
            names=(
                'theta',
                'rho_0_p',
                'varrho_0_p',
                'rho_0_m',
                'varrho_0_m',
                'rho_1_p',
                'varrho_1_p',
                'rho_1_m',
                'varrho_1_m',
                'rho_2_p',
                'varrho_2_p',
                'rho_2_m',
                'varrho_2_m',
                'rho_3_p',
                'varrho_3_p',
                'rho_3_m',
                'varrho_3_m',
                'rho_4_p',
                'varrho_4_p',
                'rho_4_m',
                'varrho_4_m',
                'rho_5_p',
                'varrho_5_p',
                'rho_5_m',
                'varrho_5_m',
            )
        )

        self.save_rho_stats(filename) #A bit dirty just because of consistency of the datatype :/
        self.load_rho_stats(filename)

    def save_rho_stats(self, filename):
        self.rho_stats.writeto(filename, format='fits')

    def load_rho_stats(self, filename):
        self.rho_stats = fits.getdata(filename)

    def plot_rho_stats(self):
        """
        plot_rho_stats

        Method to plot Rho + statistics.
        """

        pass


class TauStat():
    """
    TauStat

    Class to compute the tau statistics (Gatti 2022) of a PSF and gal catalogue.
    """

    def __init__(self, params, output, treecorr_config=None, catalogs=None, verbose=False):

        if catalogs is None:
            self.catalogs = Catalogs(params, output)
        else:
            self.catalogs = catalogs

        if treecorr_config is None:
            self._treecorr_config = {
                "ra_units": "deg",
                "dec_units": "deg",
                "sep_units": "arcmin",
                "min_sep": 0.1,
                "max_sep": 100,
                "n_bins": 20,
                "var_method": "jackknife"
            }
        else:
            self._treecorr_config = treecorr_config

        self.verbose = verbose

    def build_cat_to_compute_tau(self, path_cat, cat_type, catalog_id='', square_size=False):
        """
        build_cat_to_compute_tau

        Parameters
        ----------
        path_cat : str
            Path to the catalog built to compute the tau-statistics.

        cat_type : str
            Specify the type of the catalogue to build. 'gal' or 'psf'

        catalog_id : str
            An id to identify the catalog used in the keys of the stored treecorr.Catalog.

        square_size : bool
            If True, the size computed in the catalogue is squared (Default: False)
        """

        if cat_type=="psf":
            self.catalogs.read_shear_cat(path_gal=None, path_psf=path_cat)

            if self.verbose:
                print("Building catalogs...")

            self.catalogs.build_catalog(cat_type='psf', key='psf_'+catalog_id, square_size=square_size)
            self.catalogs.build_catalog(cat_type='psf_error', key='psf_error_'+catalog_id, square_size=square_size)
            self.catalogs.build_catalog(cat_type='psf_size_error', key='psf_size_error_'+catalog_id, square_size=square_size)

        else:
            self.catalogs.read_shear_cat(path_gal=path_cat, path_psf=None)

            if self.verbose:
                print("Building catalog...")
            
            self.catalogs.build_catalog(cat_type='gal', key='gal_'+catalog_id)

        if self.verbose:
            print("Catalogs successfully built...")
            self.catalogs.show_catalogs()

    def compute_tau_stats(self, catalog_id, filename):
        """
        compute_tau_stats

        Compute the tau statistics of your catalog and save it.

        Parameters:
        ----------
        catalog_id : str
            The id of the catalog used to compute the rho statistics.

        filename : str
            The path where the rho stats will be saved.
        """
        if self.verbose:
            print("Computation of the tau statistics in progress...")
        tau_0 = treecorr.GGCorrelation(self._treecorr_config)
        tau_0.process(self.catalogs['gal_'+catalog_id], self.catalogs['psf_'+catalog_id])
        tau_2 = treecorr.GGCorrelation(self._treecorr_config)
        tau_2.process(self.catalogs['gal_'+catalog_id], self.catalogs['psf_error_'+catalog_id])
        tau_5 = treecorr.GGCorrelation(self._treecorr_config)
        tau_5.process(self.catalogs['gal_'+catalog_id], 'psf_size_error_'+catalog_id)

        self.tau_stats = Table(
            [
                tau_0.rnom,
                tau_0.xip,
                tau_0.varxip,
                tau_0.xim,
                tau_2.xip,
                tau_2.varxip,
                tau_2.xim,
                tau_2.varxim,
                tau_5.xip,
                tau_5.varxip,
                tau_5.xim,
                tau_5.varxim,
            ],
            names=(
                'theta',
                'tau_0_p',
                'vartau_0_p',
                'tau_0_m',
                'vartau_0_m',
                'tau_2_p',
                'vartau_2_p',
                'tau_2_m',
                'vartau_2_m',
                'tau_5_p',
                'vartau_5_p',
                'tau_5_m',
                'vartau_5_m',
            )
        )

        self.save_tau_stats(filename) #A bit dirty just because of consistency of the datatype :/
        self.load_tau_stats(filename)

    def save_tau_stats(self, filename):
        self.tau_stats.writeto(filename, format='fits')

    def load_tau_stats(self, filename):
        self.tau_stats = fits.getdata(filename)

    def plot_tau_stats(self):
        pass