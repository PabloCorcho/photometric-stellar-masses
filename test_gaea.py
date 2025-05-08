#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May  8 13:36:13 2025

@author: pcorchoc
"""

import h5py
import numpy as np
from astropy import units as u
from astropy.cosmology import FlatLambdaCDM
from astropy.table import Table

from pst.SSP import PyPopStar
from pst.models import LogNormalZPowerLawCEM, ExponentialDelayedZPowerLawCEM
from pst.observables import Filter, load_photometric_filters
from pst.dust import DustScreen

from matplotlib import pyplot as plt

from photo_grid import YJH_MLR

from astropy.cosmology import WMAP9 as cosmo


cosmos = Table.read("/home/pcorchoc/Research/obs_data/COSMOS/COSMOS2020_FARMER_R1_v2.2_p3.fits")

mask = (cosmos["UVISTA_Y_VALID"]) & (cosmos["UVISTA_J_VALID"]) & (cosmos["UVISTA_H_VALID"])
#mask &= cosmos["VALID_SOURCE"] == True
mask &= cosmos["MODEL_FLAG"] == False
mask &= (cosmos["FLAG_UVISTA"] == False)
mask &= (cosmos["FLAG_COMBINED"] == False)

redshift = cosmos["lp_zBEST"].value.data
mask &= (redshift > 0.01) & (redshift < 0.95)

mask &= ~cosmos["UVISTA_Y_MAG"].value.mask
mask &= ~cosmos["UVISTA_J_MAG"].value.mask
mask &= ~cosmos["UVISTA_H_MAG"].value.mask

np.random.seed(50)
mask = np.sort(np.random.choice(np.where(mask)[0], size=10000,
                                replace=False))

redshift = redshift[mask]
y = cosmos["UVISTA_Y_MAG"][mask]
yerr = cosmos["UVISTA_Y_MAGERR"][mask]
j = cosmos["UVISTA_J_MAG"][mask]
jerr = cosmos["UVISTA_J_MAGERR"][mask]
h = cosmos["UVISTA_H_MAG"][mask]
herr = cosmos["UVISTA_H_MAGERR"][mask]

cosmos_mass = cosmos["lp_mass_best"][mask]
dist_mod = cosmo.distmod(redshift).value

ir_mass_model = YJH_MLR.from_hdf5("photometry_grid_cosmos.hdf5")



# dist_mod = 0
mass = np.array([ir_mass_model.get_mass(y, j, h, z_obs=z_obs, maxlike=True) for y, j, h, z_obs in zip(
    - dist_mod, j - dist_mod, h - dist_mod, redshift)])

logmass_offset = mass - cosmos_mass

plt.figure(figsize=(8, 4))
plt.subplot(121)
plt.hist(logmass_offset, bins="auto",
         range=[-2, 2])
plt.axvline(np.nanmedian(logmass_offset),
            color="k")
plt.axvline(0, ls=":", color="k")
plt.subplot(122)
plt.scatter(redshift, logmass_offset, s=1,
            c= j - h,
            vmin=ir_mass_model.model_jh.min(),
            vmax=ir_mass_model.model_jh.min() + 1,
            cmap="nipy_spectral")
plt.colorbar()
plt.ylim(-2, 2)