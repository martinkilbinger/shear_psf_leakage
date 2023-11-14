"""
This module sets up a class to compute the rho stats computation

Author: Sacha Guerrini
"""

import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits

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
        assert (path_gal is None) and (path_psf is None), ("Please specify a path for the shear catalog you want to read.")
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
        
        assert (self.dat_shear is not None) and (self.dat_psf is not None), ("Check you read the shear catalogs correctly.")
        
        if cat_type=="gal":
            ra = self.dat_shear[self._params["ra_col"]]
            dec = self.dat_shear[self._params["dec_col"]]
            g1 = self.dat_shear[self._params["e1_col"]] - self.dat_shear[self._params["e1_col"]].mean()
            g2 = self.dat_shear[self._params["e2_col"]] - self.dat_shear[self._params["e2_col"]].mean()
            if self._params["w_col"] is not None:
                weights = self.dat_shear[self._params["w_col"]]
            else:
                weights = np.ones.like(ra)
        else:
            #Add a mask?
            #mask = (self.dat_psf[self._params["FLAG_PSF_HSM"]]==0) & (self.dat_psf[self._params["FLAG_STAR_HSM"]]==0)
            ra = self.dat_psf[self._params["ra_col"]]
            dec = self.dat_psf[self._params["dec_psf_col"]]
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
                size_star = self.dat_psf[self._params["SIGMA_STAR_HSM"]]**2 if square_size else  self.dat_psf[self._params["SIGMA_STAR_HSM"]]
                size_psf = self.dat_psf[self._params["SIGMA_STAR_HSM"]]**2 if square_size else  self.dat_psf[self._params["SIGMA_STAR_HSM"]]
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

    def __init__(self, params, output, treecorr_config=None, verbose=False):

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
        self.catalogs.build_catalog(cat_type='psf_error', key='psf_'+catalog_id, square_size=square_size)
        self.catalogs.build_catalog(cat_type='psf_size_error', key='psf_'+catalog_id, square_size=square_size)

        if self.verbose:
            print("Catalogs successfully built...")
            self.catalogs.show_catalogs()

    def compute_rho_stats(self):
        pass

    def save_rho_stats(self, filename):
        pass

    def load_rho_stats(self, filename):
        pass

    def plot_rho_stats(self):
        pass
