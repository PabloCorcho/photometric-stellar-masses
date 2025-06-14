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
from astropy.stats import sigma_clipped_stats

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

plt.rcParams['figure.figsize'] = (10,7)
plt.rcParams['xtick.labelsize'] = 16
plt.rcParams['ytick.labelsize'] = 16
plt.rcParams['axes.labelsize'] = 20
plt.rcParams['ytick.direction']='in'
plt.rcParams['xtick.direction']='in'
plt.rcParams['ytick.major.size']=10
plt.rcParams['xtick.major.size']=10
plt.rcParams['ytick.minor.size']=5
plt.rcParams['xtick.minor.size']=5
plt.rcParams['xtick.top']=True
plt.rcParams['ytick.right']=True
plt.rcParams['xtick.minor.visible'] = True
plt.rcParams['ytick.minor.visible'] = True
plt.rcParams['legend.fancybox'] = True
plt.rcParams['legend.fontsize'] = 16
plt.rcParams['axes.titlesize']=16
plt.rcParams['axes.linewidth'] = 1.3


# filename = '/home/pmsa/code/Euclid/analysis/catalogues/besta_catalogue_param_1.6.csv'
filename = 'test_galaxies_1.3.csv'          # The suffix indicate the growth rate of the isophotes
df = pd.read_csv(filename)

# Iterate through any of the galaxies by changing the index
# Index that the observe profiles looks nice:  [*1.3.csv]
# 20, 8, 10, 12, 40, 34
index_object = 105  # Change this index to select a different object
z_label = 'z_spec' # Change this to 'phz_pp_median_redshift' if you want to use photometric redshift
object_id = np.unique(df['object_id'])[index_object]         
arg_obj = df['object_id'] == object_id
arg_obj
sma = np.unique(df['radius'][arg_obj])
y_flux = df[(df['band'] == 'Euclid_NISP.Y')*arg_obj]['intensity'].to_numpy()
j_flux = df[(df['band'] == 'Euclid_NISP.J')*arg_obj]['intensity'].to_numpy()
h_flux = df[(df['band'] == 'Euclid_NISP.H')*arg_obj]['intensity'].to_numpy()

y_mags = 23.9 - 2.5*np.log10(y_flux) - 5
j_mags = 23.9 - 2.5*np.log10(j_flux) - 5
h_mags = 23.9 - 2.5*np.log10(h_flux) - 5

z = np.zeros(len(h_mags)) + df[z_label][arg_obj].iloc[0]

scale_to_pc = arcsec_to_kpc(1,z[0])*1e3

luminosity_distance = cosmo.luminosity_distance(z).to('pc').value  # Convert to parsecs
dist_mod = np.log10(luminosity_distance / 10)


# ir_mass_model = YJH_MLR.from_hdf5("photometry_grid_pypopstar_euc.hdf5")
ir_mass_model = YJH_MLR.from_hdf5("photometry_grid_emiles_euc.hdf5")

logprior = 0
masses = np.zeros_like(y_mags)
masses_err = np.zeros_like(y_mags)

masses_prior = np.zeros_like(y_mags)
masses_err_prior = np.zeros_like(y_mags)

for idx, (y, j, h, z_obs) in enumerate(
        zip(y_mags - dist_mod, j_mags - dist_mod, h_mags - dist_mod, z)):
    logpost, logmass_g = ir_mass_model.get_posterior(y, j, h, z_obs=z_obs,
                                                     logprior=None)
    masses[idx], masses_err[idx] = ir_mass_model.get_mean_mass(y, j, h, z_obs=z_obs,
                                                              logprior=logprior)
    masses_prior[idx], masses_err_prior[idx] = ir_mass_model.get_mean_mass(
        y, j, h, z_obs=z_obs, logprior=logprior)
    logprior = logpost
    
    # plt.figure()
    # plt.hist(logmass_g, weights=np.exp(logpost),  bins=20,
    #          range=[masses_prior[idx] - 5 * masses_err_prior[idx],
    #                 masses_prior[idx] + 5 * masses_err_prior[idx]])
    # break

masses += -np.log10(1/scale_to_pc**2)
masses_prior += -np.log10(1/scale_to_pc**2)

fig, (ax1,ax2) = plt.subplots(1, 2,sharex=True, figsize=(12, 6))
ax1.plot(sma, y_mags, 'r-*', label='Y')
ax1.plot(sma, j_mags, 'b-*', label='J')
ax1.plot(sma, h_mags, 'g-*', label='H')
ax1.set_xlabel('semi-major axis [arcsec]')
ax1.set_ylabel('$\mu$ [mag/arcsec''$^2$]')
ax2.text(0.95, 0.95, r'$z_{\rm spec}'+f' = {z[0]:.2f}$', transform=ax2.transAxes, fontsize=16, va='top',ha='right')
ax1.legend()
ax1.invert_yaxis()
ax2.plot(sma, masses, 'o', label='Masses')
ax2.set_xlabel('semi-major axis [arcsec]')
ax2.set_ylabel('$\log(\Sigma_*)$ [$M_\odot/$pc$^2$]')
fig.tight_layout()

fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(10, 4), constrained_layout=True)
ax = axs[0]
twax = ax.twinx()
ax.errorbar(sma, masses, yerr=masses_err, label="Masses")
ax.errorbar(sma, masses_prior, yerr=masses_err_prior, label="Masses")

twax.plot(sma, y_mags - j_mags, 'r-*', label='Y - J')
twax.plot(sma, j_mags - h_mags, 'b-*', label='J - H')
twax.legend(loc="lower left")
ax.set_ylabel('$\log(\Sigma_*)$ [$M_\odot/$pc$^2$]')
ax.set_xlabel('semi-major axis [arcsec]')
twax.set_ylabel('color index')

ax = axs[1]
ax.scatter(ir_mass_model.model_jh.flatten(),ir_mass_model.model_yj.flatten(),
           alpha=0.6,s=2,c='grey')
mappable = ax.scatter(j_mags - h_mags, y_mags - j_mags,
                      c=sma)
ax.set_xlim(-1, 1)
ax.set_ylim(-1, 1)
ax.set_xlabel('J - H')
ax.set_ylabel('Y - J')
plt.colorbar(mappable, ax=ax, label='semi-major axis [arcsec]')


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

# %% Colour Grid Comparison with Log Density 

fig, ax = plt.subplots(figsize=(8, 6))

# Compute the 2D histogram
hist, xedges, yedges = np.histogram2d(
    j_mags - h_mags, 
    y_mags - j_mags, 
    bins=1000
)

# Apply logarithmic transformation to the histogram data
log_hist = np.log10(hist + 1)  # Add 1 to avoid log(0)

# Plot the histogram using pcolormesh
im = ax.pcolormesh(
    xedges, 
    yedges, 
    log_hist.T, 
    cmap='nipy_spectral', 
    vmin=0, 
    vmax=np.max(log_hist)
)

# Overlay scatter points
ax.scatter(
    ir_mass_model.model_jh.flatten(),
    ir_mass_model.model_yj.flatten(),
    alpha=0.6,
    s=2,
    c='white'
)

# Set axis limits and labels
ax.set_xlim([-2, 2])
ax.set_ylim([-2, 2])
cbar = fig.colorbar(im, ax=ax)  # Create the colorbar
cbar.set_label('Log Density')
ax.set_xlabel('$J_{\mathrm{E}}$ - $H_{\mathrm{E}}$')
ax.set_ylabel('$Y_{\mathrm{E}}$ - $J_{\mathrm{E}}$')


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

#%% Colours vs Surface Brightness

# Prepare data
jh_color = j_mags - h_mags
yj_color = y_mags - j_mags
cmap = plt.get_cmap('nipy_spectral')
cmap.set_under(alpha=0)

# Define bins for the histogram
bins = [500, 500]  # Number of bins for x and y axes
x_range = [h_mags.min(), h_mags.max()]
y_range_jh = [jh_color.min(), jh_color.max()]
y_range_yj = [yj_color.min(), yj_color.max()]

# Compute 2D histograms
hist_jh, xedges_jh, yedges_jh = np.histogram2d(h_mags, jh_color, bins=bins, range=[x_range, y_range_jh])
hist_yj, xedges_yj, yedges_yj = np.histogram2d(h_mags, yj_color, bins=bins, range=[x_range, y_range_yj])


# Plot the histograms
fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, sharey=True, figsize=(8, 8))

# First plot: $H_{\mathrm{E}}$ vs $J_{\mathrm{E}}$ - $H_{\mathrm{E}}$
im1 = ax1.pcolormesh(xedges_jh, yedges_jh, hist_jh.T, cmap=cmap, vmin=1)
fig.colorbar(im1, ax=ax1, label='Counts')
ax1.set_xlabel('$H_{\mathrm{E}}$')
ax1.set_ylabel('$J_{\mathrm{E}}$ - $H_{\mathrm{E}}$')

# Second plot: $H_{\mathrm{E}}$ vs $Y_{\mathrm{E}}$ - $J_{\mathrm{E}}$
im2 = ax2.pcolormesh(xedges_yj, yedges_yj, hist_yj.T, cmap=cmap, vmin=1)
fig.colorbar(im2, ax=ax2, label='Counts')
ax2.set_xlabel('$H_{\mathrm{E}}$')
ax2.set_ylabel('$Y_{\mathrm{E}}$ - $J_{\mathrm{E}}$')

ax1.set_ylim([-5,5])


plt.tight_layout()
plt.show()


# %% Histogram of the colours

plt.figure()

_,med,std = sigma_clipped_stats(j_mags-h_mags)
_ = plt.hist(j_mags-h_mags,bins=600,histtype='step', label='$J_{\mathrm{E}}$ - $H_{\mathrm{E}}$ ', color='magenta')
plt.axvline(med,color='magenta',lw=2)
plt.axvline(med+5*std,color='magenta',ls='--',lw=2)
plt.axvline(med-5*std,color='magenta',ls='--',lw=2, label='5$\sigma$')
_,med,std = sigma_clipped_stats(y_mags-j_mags)
_ = plt.hist(y_mags-j_mags,bins=600,histtype='step', label='$Y_{\mathrm{E}}$ - $J_{\mathrm{E}}$ ',color='orange')
plt.axvline(med,color='orange',lw=2)
plt.axvline(med+5*std,color='orange',ls='--',lw=2, label='5$\sigma$')
plt.axvline(med-5*std,color='orange',ls='--',lw=2)
plt.legend()
plt.xlabel('Colour')
plt.ylabel('Counts')
plt.yscale('log')

