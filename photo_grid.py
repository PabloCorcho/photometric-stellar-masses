import numpy as np
from astropy import units as u
from astropy.cosmology import FlatLambdaCDM
from astropy.table import Table

from pst.SSP import PyPopStar, BC03_2013
from pst.models import LogNormalZPowerLawCEM, ExponentialDelayedZPowerLawCEM
from pst.observables import Filter, load_photometric_filters
from pst.dust import DustScreen

from matplotlib import pyplot as plt

import h5py


class YJH_MLR:
    def __init__(self, model_y, model_j, model_h, z_grid, phys_params=None):
        self.model_yj = np.nan_to_num(model_y - model_j, nan=99)
        self.model_jh = np.nan_to_num(model_j - model_h, nan=99)
        self.model_h = model_h
        self.redshift = z_grid
        self.phys_params = phys_params

    def get_mass(self, y, j, h, yerr=0.01, jerr=0.01, herr=0.01, z_obs=0.0, maxlike=False):
        
        idx = np.searchsorted(self.redshift, z_obs).clip(min=1, max=self.redshift.size)
        widx = 1 - (self.redshift[idx] - z_obs) / (self.redshift[idx] - self.redshift[idx - 1])
        model_yj = self.model_yj[idx] * widx + self.model_yj[idx - 1] * (1 - widx)
        model_jh = self.model_jh[idx] * widx + self.model_jh[idx - 1] * (1 - widx)
        model_h = self.model_h[idx] * widx + self.model_h[idx - 1] * (1 - widx)

        loglike = -0.5 * (
            (model_yj - (y - j))**2 / (yerr**2 + jerr**2)
            + (model_jh - (j - h))**2 / (jerr**2 + herr**2))
        
        if maxlike:
            best_fit = np.nanargmax(loglike)
            best_fit_h_logmass = (model_h[best_fit] - h) / 2.5
            return best_fit_h_logmass
        else:
            mean_h_logmass = np.nansum(np.exp(loglike) * (
            model_h - h) / 2.5) / np.nansum(np.exp(loglike))
            return mean_h_logmass

    @classmethod
    def from_pickle():
        pass

    @classmethod
    def from_hdf5(cls, path):
        data = h5py.File(path, "r")
        photo_grid = data["photometry_grid"][()]
        z_grid = data["z_grid"][()]
        if "phys_prop" in data.keys():
            phys_prop = data["phys_prop"][()]
        data.close()
        return cls(photo_grid[:, :, 0], photo_grid[:, :, 1], photo_grid[:, :, 2],
                   z_grid, phys_prop)
        

if __name__ == "__main__":
    # CREATE THE GRID
    cosmo = FlatLambdaCDM(H0=70, Om0=0.28)
    ssp = PyPopStar(IMF="KRO")
    # ssp = BC03_2013(model='stelib', imf='chabrier',
    #                 path="/home/pcorchoc/Research/SSP_TEMPLATES/BC03/ff")

    ssp.interpolate_sed(np.arange(5000.0, 22000.0, 10) * u.angstrom)

    dust_model = DustScreen("ccm89")

    ism_metallicity_range = np.geomspace(0.004, 0.1, 15)
    t0_range = np.geomspace(0.1, 30, 15)
    scale_range = np.geomspace(0.01, 10, 15)
    av_range = np.linspace(0, 1, 1)
    z_grid = np.arange(0, 2, 0.05)
    filter_names = [
        # "Subaru_HSC.g",
        # "CFHT_MegaCam.r",
        # "PANSTARRS_PS1.i",
        # "Subaru_HSC.z",
        "Euclid_NISP.Y",
        "Euclid_NISP.J",
        "Euclid_NISP.H",
        # "Euclid_VIS.vis"
        ]
    
    # For COSMOS
    # filter_names = [
    #     # "Subaru_HSC.g",
    #     # "CFHT_MegaCam.r",
    #     # "PANSTARRS_PS1.i",
    #     # "Subaru_HSC.z",
    #     "Paranal_VISTA.Y",
    #     "Paranal_VISTA.J",
    #     "Paranal_VISTA.H",
    #     # "Paranal_VISTA.K",
    #     ]

    filters = load_photometric_filters(filter_names)
    av_all, ism_met_grid, t0_grid, scale_grid  = np.meshgrid(
        av_range, ism_metallicity_range, t0_range, scale_range, indexing="ij")
    phys_prop = np.vstack([av_all.flatten(),
                           ism_met_grid.flatten(), t0_grid.flatten(),
                           scale_grid.flatten()])
    model_photometry = np.zeros((z_grid.size, ism_met_grid.size, len(filters)))

    for ith, z in enumerate(z_grid):

        photo_grid = [dust_model.redden_ssp_model(
            ssp, a_v=av).compute_photometry(filters, z_obs=z)  for av in av_range]

        for jth, (met, t0, scale) in enumerate(zip(ism_met_grid.flatten(),
                                                 t0_grid.flatten(),
                                                 scale_grid.flatten())):
            print(ith)
            model = LogNormalZPowerLawCEM(ism_metallicity_today=met,
                                          alpha_powerlaw=1.0,
                                          t0=t0 << u.Gyr,
                                          scale=scale, today=cosmo.age(z),
                                          mass_today=1 << u.Msun)
            # model = ExponentialDelayedZPowerLawCEM(ism_metallicity_today=z,
            #                                        alpha_powerlaw=1.0)
            model_photometry[ith, jth] = np.array(
                [model.compute_photometry(
                    ssp, t_obs=cosmo.age(z),
                    photometry=photo).to_value("3631 Jy") for photo in photo_grid])

    # model_photometry = model_photometry.reshape((z_grid.size, *ism_met_grid.shape, len(filters)))

    output = h5py.File("photometry_grid_pypopstar_euc.hdf5", "w")
    output.create_dataset(
        "photometry_grid", data=-2.5 * np.log10(model_photometry))
    output["photometry_grid"].attrs["Filters"] = filter_names
    output.create_dataset("z_grid", data=z_grid)
    output.create_dataset("phys_prop", data=phys_prop)
    output["phys_prop"].attrs["names"] = ["av", "ism_metallicity", "t0", "scale"]
    output.close()

    # %%
    ir_mass_model = YJH_MLR.from_hdf5("photometry_grid_pypopstar_euc.hdf5")

    # ir_mass_model = YJH_MLR(model_y=-2.5 * np.log10(model_photometry[:, :, 0]),
    #                         model_j=-2.5 * np.log10(model_photometry[:, :, 1]),
    #                         model_h=-2.5 * np.log10(model_photometry[:, :,  2]),
    #                         z_grid=z_grid,
    #                         phys_params=phys_prop)

    # =============================================================================
    # Test with TNG
    # =============================================================================
    z_obs = 0.0
    
    if z_obs == 0.3:
        tng = Table.read("/home/pcorchoc/Research/Euclid/data/euc_q1_quenching/sfh-catalogs/IllustrisTNG100-1_sfh_catalog_z_0.3_v1.fits.input_cat.join.fits")
    elif z_obs == 0.0:
        tng = Table.read("/home/pcorchoc/Research/Euclid/data/euc_q1_quenching/sfh-catalogs/IllustrisTNG100-1_sfh_catalog_z_0.0_v4.fits.input_cat.join.fits")
    
    
    tng_gr = -2.5 * np.log10(tng["Subaru_HSC.g"].value / tng["CFHT_MegaCam.r"].value)
    tng_visJ = -2.5 * np.log10(tng["Euclid_VIS.vis"].value / tng["Euclid_NISP.J"].value)
    
    tng_visY = -2.5 * np.log10(tng["Euclid_VIS.vis"].value / tng["Euclid_NISP.Y"].value)
    tng_yj = -2.5 * np.log10(tng["Euclid_NISP.Y"].value / tng["Euclid_NISP.J"].value)
    tng_jh = -2.5 * np.log10(tng["Euclid_NISP.J"].value / tng["Euclid_NISP.H"].value)
    
    
    # best_fit = np.array([np.argmin((yj - col1)**2 + (jh - col2)**2
    #                                 ) for col1, col2 in zip(tng_yj, tng_jh)])
    # lmr = model_photometry[:, :, :, :, -2].flatten()[best_fit]
    # mass = np.log10(tng["Euclid_NISP.H"].value / 1e9 / lmr)
    
    mass = np.array(
        [ir_mass_model.get_mass(y, j, h, z_obs=z_obs, maxlike=True) for y, j, h in zip(
        -2.5 * np.log10(tng["Euclid_NISP.Y"].value / 1e9),
        -2.5 * np.log10(tng["Euclid_NISP.J"].value / 1e9),
        -2.5 * np.log10(tng["Euclid_NISP.H"].value / 1e9))])
    
    
    
    logmass_offset = mass - np.log10(tng["stellar_mass"]) + 2 * np.log10(0.7)
    
    plt.figure()
    plt.hist(logmass_offset, bins="auto",
             range=[-1, 1])
    plt.axvline(np.nanmedian(logmass_offset),
                color="k")
    plt.axvline(0, ls=":", color="k")
    plt.xlabel("model logM - ref logM")

