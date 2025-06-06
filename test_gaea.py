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

import h5py

gaea = h5py.File("/home/pcorchoc/Research/Euclid/SIMS/GAEA/data/lc_EUCLID_hdf5_ECLM_vH25_RADEC")

yjh = gaea["Mag"][:, [1, 2, 3]]

plt.figure()
plt.scatter(yjh[:, 0] - yjh[:, 1],
            yjh[:, 1] - yjh[:, 2], s=1,
            c=gaea["z"])
plt.colorbar(label="Redshift")
plt.xlabel("J-H")
plt.ylabel("Y-J")