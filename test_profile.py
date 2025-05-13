#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May  9 13:36:13 2025

@author: pmsastro
"""
#%%

import numpy as np
from matplotlib import pyplot as plt
from photo_grid import YJH_MLR
from astropy.cosmology import Planck18 as cosmo  # Use Planck 2018 cosmology
import pandas as pd
from astropipe.utils import arcsec_to_kpc, average_bin
from scipy.interpolate import interp1d

def arcsec_to_kpc(arcsec, redshift, H0=70, Tcmb0=2.725, Om0=0.3):
    '''Function that given a projected angular size of an object
    and its redshift it converts it into the
    physical size in same units. 

    Parameters
    ----------
        arcsec : float
            Angular size in arcsec.
        redshift : float
            Redshift of the object.
        H0 : float, optional
            Hubble constant. The default is 70. [km/s/Mpc]
        Tcmb0 : float, optional
            CMB temperature. The default is 2.725. [K]
        Om0 : float, optional
            Matter density. The default is 0.3.
    
    Returns
    -------
        kpc : float
            Physical size in kpc.
      '''
    da = cosmo.angular_diameter_distance(redshift)*1e3  # Mpc --> kpc
    arcsec_to_rad = np.pi/(180*3600)
    return (arcsec*arcsec_to_rad*da).value



filename = '/home/pmsa/code/Euclid/analysis/catalogues/besta_catalogue_param.csv'
df = pd.read_csv(filename)

# object_id = -514086977285521890
object_id = -538439791285946076
# object_id = np.unique(df['object_id'])[123]
arg_obj = df['object_id'] == object_id
sma = np.unique(df['radius'][arg_obj])
y_flux = df[(df['band'] == 'Euclid_NISP.Y')*arg_obj]['intensity']
j_flux = df[(df['band'] == 'Euclid_NISP.J')*arg_obj]['intensity']
h_flux = df[(df['band'] == 'Euclid_NISP.H')*arg_obj]['intensity']

y_mags = 23.9 - 2.5*np.log10(y_flux) - 5
j_mags = 23.9 - 2.5*np.log10(j_flux) - 5
h_mags = 23.9 - 2.5*np.log10(h_flux) - 5

z = np.zeros(len(h_mags)) + df['phz_pp_median_redshift'][arg_obj].iloc[0]

scale_to_pc = arcsec_to_kpc(1,z[0])*1e3

luminosity_distance = cosmo.luminosity_distance(z).to('pc').value  # Convert to parsecs
dist_mod = np.log10(luminosity_distance / 10)


ir_mass_model = YJH_MLR.from_hdf5("photometry_grid_cosmos.hdf5")


masses = np.array([ir_mass_model.get_mass(y, j, h, z_obs=z_obs, maxlike=True) for y, j, h, z_obs in zip(
    y_mags - dist_mod, j_mags - dist_mod, h_mags - dist_mod, z)])

masses += -np.log10(1/scale_to_pc**2)

fig, (ax1,ax2) = plt.subplots(1, 2,sharex=True, figsize=(12, 6))
ax1.plot(sma, y_mags, 'r-*', label='Y')
ax1.plot(sma, j_mags, 'b-*', label='J')
ax1.plot(sma, h_mags, 'g-*', label='H')
ax1.set_xlabel('semi-major axis [arcsec]')
ax1.set_ylabel('$\mu$ [mag/arcsec$^2$]')
ax1.legend()
ax1.invert_yaxis()
ax2.plot(sma, masses, 'o', label='Masses')
ax2.set_xlabel('semi-major axis [arcsec]')
ax2.set_ylabel('$\log(\Sigma_*)$ [$M_\odot/$pc$^2$]')
fig.tight_layout()
# %%
