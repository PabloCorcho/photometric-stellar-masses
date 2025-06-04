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



filename = '/home/pmsa/code/Euclid/analysis/catalogues/besta_catalogue_param_1.6.csv'
df = pd.read_csv(filename)

# object_id = -514086977285521890
# object_id = -538439791285946076
object_id = np.unique(df['object_id'])[108]
arg_obj = df['object_id'] == object_id
arg_obj
sma = np.unique(df['radius'][arg_obj])
y_flux = df[(df['band'] == 'Euclid_NISP.Y')*arg_obj]['intensity'].to_numpy()
j_flux = df[(df['band'] == 'Euclid_NISP.J')*arg_obj]['intensity'].to_numpy()
h_flux = df[(df['band'] == 'Euclid_NISP.H')*arg_obj]['intensity'].to_numpy()

y_mags = 23.9 - 2.5*np.log10(y_flux) - 5
j_mags = 23.9 - 2.5*np.log10(j_flux) - 5
h_mags = 23.9 - 2.5*np.log10(h_flux) - 5

z = np.zeros(len(h_mags)) + df['phz_pp_median_redshift'][arg_obj].iloc[0]

scale_to_pc = arcsec_to_kpc(1,z[0])*1e3

luminosity_distance = cosmo.luminosity_distance(z).to('pc').value  # Convert to parsecs
dist_mod = np.log10(luminosity_distance / 10)


ir_mass_model = YJH_MLR.from_hdf5("photometry_grid_pypopstar.hdf5")


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
# %% Colour grid comparison

sma = np.unique(df['radius'][arg_obj])
y_flux = df[(df['band'] == 'Euclid_NISP.Y')]['intensity'].to_numpy()
j_flux = df[(df['band'] == 'Euclid_NISP.J')]['intensity'].to_numpy()
h_flux = df[(df['band'] == 'Euclid_NISP.H')]['intensity'].to_numpy()


y_mags = 23.9 - 2.5*np.log10(y_flux) - 5
j_mags = 23.9 - 2.5*np.log10(j_flux) - 5
h_mags = 23.9 - 2.5*np.log10(h_flux) - 5

z = df['phz_pp_median_redshift'][(df['band'] == 'Euclid_NISP.Y')]


plt.figure()
im = plt.scatter(j_mags - h_mags,y_mags-j_mags,marker='s',s=1,alpha=0.3, c=h_mags,vmax=29, cmap='nipy_spectral')
plt.colorbar(im,alpha=1)
plt.scatter(ir_mass_model.model_jh.flatten(),ir_mass_model.model_yj.flatten(),alpha=0.6,s=2,c='grey')#c=ir_mass_model.redshift.tolist()*ir_mass_model.model_yj.shape[1])
plt.xlim([-5,5])
plt.ylim([-5,5])
plt.xlabel('$J_{\mathrm{E}}$ - $H_{\mathrm{E}}$ ')
plt.ylabel('$Y_{\mathrm{E}}$ - $J_{\mathrm{E}}$ ')
plt.tight_layout()

#%%
# Define redshift bins
redshift_bins = np.logspace(np.log10(0.01), np.log10(1), 10)  # 9 bins between 0 and 3
bin_indices = np.digitize(z, redshift_bins) - 1  # Assign each redshift to a bin

# Create subplots
fig, axes = plt.subplots(3, 3, figsize=(15, 15), sharex=True, sharey=True)
axes = axes.flatten()

for i in range(9):
    ax = axes[i]
    bin_mask = (bin_indices == i)
    
    # Filter data for the current redshift bin
    jh_colors = (j_mags - h_mags)[bin_mask]
    yj_colors = (y_mags - j_mags)[bin_mask]
    h_mags_bin = h_mags[bin_mask]
    
    # Plot scatter for the current bin
    scatter = ax.scatter(jh_colors, yj_colors, c=h_mags_bin, s=1, alpha=0.3, cmap='nipy_spectral', vmax=29)
    ax.set_xlim([-5, 5])
    ax.set_ylim([-5, 5])
    ax.axvline(1, color='grey', lw=1, ls='--')
    ax.axhline(1, color='grey', lw=1, ls='--')
    ax.axvline(-1, color='grey', lw=1, ls='--')
    ax.axhline(-1, color='grey', lw=1, ls='--')
    ax.set_title(f"Redshift bin {redshift_bins[i]:.2f} - {redshift_bins[i+1]:.2f}")
    ax.set_xlabel('$J_{\mathrm{E}}$ - $H_{\mathrm{E}}$')
    ax.set_ylabel('$Y_{\mathrm{E}}$ - $J_{\mathrm{E}}$')

# Add a colorbar
fig.colorbar(scatter, ax=axes, location='bottom', orientation='horizontal', fraction=0.02, pad=-0.2)
fig.tight_layout()
plt.show()

#%%
import seaborn as sns

# Create the density plot
fig, (ax1,ax2) = plt.subplots(2,1, sharex=True, sharey=True)
cmap = plt.get_cmap('nipy_spectral')
cmap.set_under(alpha=0)


data = pd.DataFrame({
    'h_mags': h_mags,
    'jh_color': j_mags - h_mags
})

sns.kdeplot(
    data=data,
    x='h_mags',
    y='jh_color',
    cmap=cmap,
    fill=True,
    levels=50,
    thresh=0.01,
    ax=ax1
)

data = pd.DataFrame({
    'h_mags': h_mags,
    'yj_color': y_mags - j_mags
})

sns.kdeplot(
    data=data,
    x='h_mags',
    y='yj_color',
    cmap=cmap,
    fill=True,
    levels=50,
    thresh=0.01,
    ax=ax2
)
# ax1.plot(h_mags, y_mags-j_mags, 'k.',alpha=0.02)
ax1.set_xlabel('$H_{\mathrm{E}}$')
ax1.set_ylabel('$Y_{\mathrm{E}}$ - $J_{\mathrm{E}}$ ')
# ax2.plot(h_mags, j_mags-h_mags, 'k.',alpha=0.02)
ax2.set_xlabel('$H_{\mathrm{E}}$')
ax2.set_ylabel('$J_{\mathrm{E}}$ - $H_{\mathrm{E}}$ ')
ax1.set_ylim([-5,5])

# %%
from astropy.stats import sigma_clipped_stats
plt.figure()

_,med,std = sigma_clipped_stats(j_mags-h_mags)
_ = plt.hist(j_mags-h_mags,bins=40,histtype='step', label='$J_{\mathrm{E}}$ - $H_{\mathrm{E}}$ ', color='magenta')
plt.axvline(med,color='magenta',lw=2)
plt.axvline(med+5*std,color='magenta',ls='--',lw=2)
plt.axvline(med-5*std,color='magenta',ls='--',lw=2, label='5$\sigma$')
_,med,std = sigma_clipped_stats(y_mags-j_mags)
_ = plt.hist(y_mags-j_mags,bins=40,histtype='step', label='$Y_{\mathrm{E}}$ - $J_{\mathrm{E}}$ ',color='orange')
plt.axvline(med,color='orange',lw=2)
plt.axvline(med+5*std,color='orange',ls='--',lw=2, label='5$\sigma$')
plt.axvline(med-5*std,color='orange',ls='--',lw=2)
plt.legend()
plt.xlabel('Colour')
plt.ylabel('Counts')
plt.yscale('log')





