"""
This module sets up a class to compute the rho stats computation

Author: Sacha Guerrini
"""

import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.table import Table

import treecorr

def neg_dash(
    ax,
    x_in,
    y_in,
    yerr_in,
    vertical_lines=True,
    xlabel='',
    ylabel='',
    rho_nb='',
    tau_nb='',
    cat_id='',
    ylim=None,
    semilogx=False,
    semilogy=False,
    **kwargs
):
    r"""Neg Dash.

    This function is for making plots with vertical errorbars,
    where negative values are shown in absolute value as dashed lines.
    The resulting plot can either be saved by specifying a file name as
    ``plot_name``, or be kept as a pyplot instance (for instance to combine
    several neg dashes).

    Parameters
    ----------
    ax : 
        The matplotlib object on which the plot is performed
    x_in : numpy.ndarray
        X-axis inputs
    y_in : numpy.ndarray
        Y-axis inputs
    yerr_in : numpy.ndarray
        Y-axis error inputs
    vertical_lines : bool, optional
        Option to plot vertical lines; default is ``True``
    xlabel : str, optional
        X-axis label
    ylabel : str, optional
        Y-axis label
    rho_nb : str, optional
        Rho number
    ylim : float, optional
        Y-axis limit
    semilogx : bool
        Option to plot the x-axis in log scale; default is ``False``
    semilogy : bool
        Option to plot the y-axis in log scale; default is ``False``

    """
    x = np.copy(x_in)
    y = np.copy(y_in)
    if yerr_in is not None:
        yerr = np.copy(yerr_in)
    else:
        yerr = np.zeros_like(x)
    # catch and separate errorbar-specific keywords from Lines2D ones
    safekwargs = dict(kwargs)
    errbkwargs = dict()
    if 'linestyle' in kwargs.keys():
        print(
            'Warning: linestyle was provided but that would kind of defeat'
            + 'the purpose, so I will just ignore it. Sorry.'
        )
        del safekwargs['linestyle']
    for errorbar_kword in [
        'fmt', 'ecolor', 'elinewidth', 'capsize', 'barsabove', 'errorevery'
    ]:
        if errorbar_kword in kwargs.keys():
            # posfmt = '-'+kwargs['fmt']
            # negfmt = '--'+kwargs['fmt']
            errbkwargs[errorbar_kword] = kwargs[errorbar_kword]
            del safekwargs[errorbar_kword]
    errbkwargs = dict(errbkwargs, **safekwargs)

    # plot up to next change of sign
    current_sign = np.sign(y[0])
    first_change = np.argmax(current_sign * y < 0)
    while first_change:
        if current_sign > 0:
            ax.errorbar(
                x[:first_change],
                y[:first_change],
                yerr=yerr[:first_change],
                linestyle='-',
                **errbkwargs,
            )
            if vertical_lines:
                ax.vlines(
                    x[first_change - 1],
                    0,
                    y[first_change - 1],
                    linestyle='-',
                    **safekwargs,
                )
                ax.vlines(
                    x[first_change],
                    0,
                    np.abs(y[first_change]),
                    linestyle='--',
                    **safekwargs,
                )
        else:
            ax.errorbar(
                x[:first_change],
                np.abs(y[:first_change]),
                yerr=yerr[:first_change],
                linestyle='--',
                **errbkwargs,
            )
            if vertical_lines:
                ax.vlines(
                    x[first_change - 1],
                    0,
                    np.abs(y[first_change - 1]),
                    linestyle='--',
                    **safekwargs,
                )
                ax.vlines(
                    x[first_change],
                    0,
                    y[first_change],
                    linestyle='-',
                    **safekwargs,
                )
        x = x[first_change:]
        y = y[first_change:]
        yerr = yerr[first_change:]
        current_sign *= -1
        first_change = np.argmax(current_sign * y < 0)
    # one last time when `first_change'==0 ie no more changes:
    if rho_nb:
        lab = fr'$\rho_{rho_nb}(\theta)$ '+cat_id
    elif tau_nb:
        lab = fr'$\tau_{tau_nb}(\theta)$' +cat_id
    else:
        lab = cat_id
    if current_sign > 0:
        ax.errorbar(x, y, yerr=yerr, linestyle='-', label=lab, **errbkwargs)
    else:
        ax.errorbar(x, np.abs(y), yerr=yerr, linestyle='--', label=lab,
                     **errbkwargs)
    if semilogx:
        ax.set_xscale('log')
    if semilogy:
        ax.set_yscale('log')
    if ylim is not None:
        ax.set_ylim(ylim)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

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

    def set_params(self, params=None, output=None):
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
    
    def build_catalog(self, cat_type, key, npatch=None, square_size=False, mask=False):
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

        mask : bool
            If True, use PSF and star flags to mask the data. (Default: False)
        """

        if npatch is None:
            npatch = self._params["patch_number"]
        
        ra, dec, g1, g2, weights = self.get_cat_fields(cat_type, square_size)

        if mask:
            flag_psf = self.dat_psf[self._params["PSF_flag"]]
            flag_star = self.dat_psf[self._params["star_flag"]]
            mask_arr = (flag_psf==0) & (flag_star==0)
            if weights is not None:
                weights = weights[mask_arr]
        else:
            mask_arr = np.array([True for i in ra])

        cat = treecorr.Catalog(
            ra=ra[mask_arr],
            dec=dec[mask_arr],
            g1=g1[mask_arr],
            g2=g2[mask_arr],
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

    def get_cat(self, key):
        """
        get_cat

        Return the catalogue stored with the given key

        Parameters
        ----------
        key : str
            The key used to identify the catalogue in the dictionary

        Returns
        -------
        treecorr.Catalog
            The requested treecorr.Catalog
        """ 
        return self.catalogs_dict[key]


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
                "nbins": 20,
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

    def build_cat_to_compute_rho(self, path_cat_star, catalog_id='', square_size=False, mask=False):
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

        mask : bool
            If True, use PSF and star flags to mask the data. (Default: False)
        """

        self.catalogs.read_shear_cat(path_gal=None, path_psf=path_cat_star)

        if self.verbose:
            print("Building catalogs...")

        self.catalogs.build_catalog(cat_type='psf', key='psf_'+catalog_id, square_size=square_size, mask=mask)
        self.catalogs.build_catalog(cat_type='psf_error', key='psf_error_'+catalog_id, square_size=square_size, mask=mask)
        self.catalogs.build_catalog(cat_type='psf_size_error', key='psf_size_error_'+catalog_id, square_size=square_size, mask=mask)

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
            print("Computation of the rho statistics of "+catalog_id+" in progress...")
        rho_0 = treecorr.GGCorrelation(self._treecorr_config)
        rho_0.process(self.catalogs.get_cat('psf_'+catalog_id), self.catalogs.get_cat('psf_'+catalog_id))
        rho_1 = treecorr.GGCorrelation(self._treecorr_config)
        rho_1.process(self.catalogs.get_cat('psf_error_'+catalog_id), self.catalogs.get_cat('psf_error_'+catalog_id))
        rho_2 = treecorr.GGCorrelation(self._treecorr_config)
        rho_2.process(self.catalogs.get_cat('psf_'+catalog_id), self.catalogs.get_cat('psf_error_'+catalog_id))
        rho_3 = treecorr.GGCorrelation(self._treecorr_config)
        rho_3.process(self.catalogs.get_cat('psf_size_error_'+catalog_id), self.catalogs.get_cat('psf_size_error_'+catalog_id))
        rho_4 = treecorr.GGCorrelation(self._treecorr_config)
        rho_4.process(self.catalogs.get_cat('psf_error_'+catalog_id), self.catalogs.get_cat('psf_size_error_'+catalog_id))
        rho_5 = treecorr.GGCorrelation(self._treecorr_config)
        rho_5.process(self.catalogs.get_cat('psf_'+catalog_id), self.catalogs.get_cat('psf_size_error_'+catalog_id))

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

        if self.verbose:
            print("Done...")

        self.save_rho_stats(filename) #A bit dirty just because of consistency of the datatype :/
        self.load_rho_stats(filename)

    def save_rho_stats(self, filename):
        self.rho_stats.write(self.catalogs._output+'/'+filename, format='fits', overwrite=True)

    def load_rho_stats(self, filename):
        self.rho_stats = fits.getdata(self.catalogs._output+'/'+filename)

    def plot_rho_stats(self, filenames, colors, catalog_ids, abs=True, savefig=None):
        """
        plot_rho_stats

        Method to plot Rho + statistics of several catalogues given in argument. Figures are saved in PNG format.

        Parameters:
        ----------
        filenames : list str
            List of the files containing the rho statistics. They can be computed using the method `compute_rho_stats` of this class.
        
        colors : list str
            Color of the plot for the different catalogs. We recommend using different colors for different catalogs for readability.

        catalogs_id : list str
            A list of catalogs id to label accurately the legend.

        abs : bool
            If True, plot the absolute value of the rho-statistics. Otherwise, plot the negative values with dashed lines.

        savefig : str
            If not None, saves the figure with the name given in savefig.
        """

        fig, ax = plt.subplots(nrows=2, ncols=3, figsize=(15,9))
        ax = ax.flatten()

        for filename, color, cat_id in zip(filenames, colors, catalog_ids): #Plot for the different catalogs
            self.load_rho_stats(filename)

            for i in range(6):
                xlabel=r"$\theta$ [arcmin]" if i>2 else ''
                ylabel=r"$\rho-$statistics" if (i==0 or i==3) else ''
                label = fr'$\rho_{i}(\theta)$ '+cat_id
                if abs:
                    ax[i].errorbar(self.rho_stats['theta'], np.abs(self.rho_stats['rho_'+str(i)+'_p']), yerr=np.sqrt(self.rho_stats['varrho_'+str(i)+'_p']),
                    label=label, color=color, capsize=2)
                    ax[i].set_xlabel(xlabel)
                    ax[i].set_ylabel(ylabel)
                    ax[i].set_xscale('log')
                    ax[i].set_yscale('log')
                else:
                    #Plot the negative values of the rho-stats in dashed lines
                    neg_dash(
                        ax[i], self.rho_stats['theta'], self.rho_stats['rho_'+str(i)+'_p'], yerr_in=np.sqrt(self.rho_stats['varrho_'+str(i)+'_p']),
                        vertical_lines=False, rho_nb=str(i), cat_id=cat_id, xlabel=xlabel, ylabel=ylabel, semilogx=True, semilogy=True, capsize=True, color=color
                    )

                ax[i].set_xlim(self._treecorr_config["min_sep"], self._treecorr_config["max_sep"])
                ax[i].legend(loc='upper right')

        if savefig is not None:
            plt.savefig(self.catalogs._output+'/'+savefig)



class TauStat():
    """
    TauStat

    Class to compute the tau statistics (Gatti 2022) of a PSF and gal catalogue.
    """

    def __init__(self, params=None, output=None, treecorr_config=None, catalogs=None, verbose=False):

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
                "max_sep": 200,
                "nbins": 20,
                "var_method": "jackknife"
            }
        else:
            self._treecorr_config = treecorr_config

        self.verbose = verbose

    def build_cat_to_compute_tau(self, path_cat, cat_type, catalog_id='', square_size=False, mask=False):
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

        mask : bool
            If True, use PSF and star flags to mask the data. (Default: False)
        """

        if cat_type=="psf":
            self.catalogs.read_shear_cat(path_gal=None, path_psf=path_cat)

            if self.verbose:
                print("Building catalogs...")

            self.catalogs.build_catalog(cat_type='psf', key='psf_'+catalog_id, square_size=square_size, mask=mask)
            self.catalogs.build_catalog(cat_type='psf_error', key='psf_error_'+catalog_id, square_size=square_size, mask=mask)
            self.catalogs.build_catalog(cat_type='psf_size_error', key='psf_size_error_'+catalog_id, square_size=square_size, mask=mask)

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
            print("Computation of the tau statistics of "+catalog_id+" in progress...")
        tau_0 = treecorr.GGCorrelation(self._treecorr_config)
        tau_0.process(self.catalogs.get_cat('gal_'+catalog_id), self.catalogs.get_cat('psf_'+catalog_id))
        tau_2 = treecorr.GGCorrelation(self._treecorr_config)
        tau_2.process(self.catalogs.get_cat('gal_'+catalog_id), self.catalogs.get_cat('psf_error_'+catalog_id))
        tau_5 = treecorr.GGCorrelation(self._treecorr_config)
        tau_5.process(self.catalogs.get_cat('gal_'+catalog_id), self.catalogs.get_cat('psf_size_error_'+catalog_id))

        self.tau_stats = Table(
            [
                tau_0.rnom,
                tau_0.xip,
                tau_0.varxip,
                tau_0.xim,
                tau_0.varxim,
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

        if self.verbose:
            print("Done...")

        self.save_tau_stats(filename) #A bit dirty just because of consistency of the datatype :/
        self.load_tau_stats(filename)

    def save_tau_stats(self, filename):
        self.tau_stats.write(self.catalogs._output+'/'+filename, format='fits', overwrite=True)

    def load_tau_stats(self, filename):
        self.tau_stats = fits.getdata(self.catalogs._output+'/'+filename)

    def plot_tau_stats(self, filenames, colors, catalog_ids, savefig=None, plot_tau_m=True):
        """
        plot_tau_stats

        Method to plot Tau + (and -) statistics of several catalogues given in argument. Figures are saved in PNG format.

        Parameters:
        ----------
        filenames : list str
            List of the files containing the rho statistics. They can be computed using the method `compute_rho_stats` of this class.
        
        colors : list str
            Color of the plot for the different catalogs. We recommend using different colors for different catalogs for readability.

        catalogs_id : list str
            A list of catalogs id to label accurately the legend.

        savefig : str
            If not None, saves the figure with the name given in savefig.

        plot_tau_m : bool
            If True, plot the tau - additionally.
        """

        nrows=1 + plot_tau_m

        fig, ax = plt.subplots(nrows=nrows, ncols=3, figsize=(15,9))

        if nrows==1:
            ax = ax.reshape(1, 3)

        for filename, color, cat_id in zip(filenames, colors, catalog_ids): #Plot for the different catalogs
            self.load_tau_stats(filename)

            for i in range(3):
                for j in range(nrows):
                    xlabel=r"$\theta$ [arcmin]" if (j==nrows-1) else ''
                    ylabel=r"$\tau-$statistics" if (i==0) else ''
                    p_or_m = 'm' if j else 'p'
                    p_or_m_label = '-' if j else '+'
                    factor_theta = np.ones_like(self.tau_stats["theta"]) if i==0 else self.tau_stats["theta"]
                    y = self.tau_stats['tau_'+str(int(0.5*i**2+1.5*i))+'_'+p_or_m]*factor_theta
                    yerr_in = np.sqrt(self.tau_stats['vartau_'+str(int(0.5*i**2+1.5*i))+'_'+p_or_m])*factor_theta
                    label = rf'$\tau_{{{int(0.5*i**2+1.5*i)}, {p_or_m_label}}}(\theta)$ '+cat_id if i==0 else rf'$\tau_{{{int(0.5*i**2+1.5*i)}, {p_or_m_label}}}(\theta)\theta$ '+cat_id

                    ax[j, i].errorbar(self.tau_stats["theta"], y, yerr=yerr_in, label=label, color=color, capsize=2)
                    ax[j, i].set_xlim(self._treecorr_config["min_sep"], self._treecorr_config["max_sep"])
                    ax[j, i].set_xlabel(xlabel)
                    ax[j, i].set_ylabel(ylabel)
                    ax[j, i].set_xscale('log')
                    ax[j, i].legend(loc='upper right')

        if savefig is not None:
            plt.savefig(self.catalogs._output+'/'+savefig)

            