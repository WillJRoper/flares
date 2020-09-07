#!/cosma/home/dp004/dc-rope1/.conda/envs/flares-env/bin/python
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import astropy.units as u
from matplotlib.colors import LogNorm
import matplotlib.gridspec as gridspec
from scipy.stats import binned_statistic
import eagle_IO.eagle_IO as E
import seaborn as sns
from scipy.spatial import cKDTree
import pickle
import itertools
matplotlib.use('Agg')

sns.set_style('whitegrid')


snap = '010_z005p000'
reg = "24"

path1 = '/cosma7/data/dp004/FLARES/FLARES-HD/FLARES_HR_' + reg + '/data/'
path2 = '/cosma7/data/dp004/FLARES/FLARES-1/G-EAGLE_' + reg + '/data/'

# Get the data for the standard res simulations
hmrs1 = E.read_array('SUBFIND', path1, snap, 'Subhalo/HalfMassRad', noH=True, numThreads=8) * 1e3
ms1 = E.read_array('SUBFIND', path1, snap, 'Subhalo/ApertureMeasurements/Mass/030kpc',
                  noH=True, numThreads=8)[:, 4] * 10**10
cops1 = E.read_array('SUBFIND', path1, snap, 'Subhalo/CentreOfPotential', physicalUnits=True,
                    noH=True, numThreads=8)

# Get the data for the high resolution
hmrs2 = E.read_array('SUBFIND', path2, snap, 'Subhalo/HalfMassRad', noH=True, numThreads=8) * 1e3
ms2 = E.read_array('SUBFIND', path2, snap, 'Subhalo/ApertureMeasurements/Mass/030kpc',
                  noH=True, numThreads=8)[:, 4] * 10**10
cops2 = E.read_array('SUBFIND', path2, snap, 'Subhalo/CentreOfPotential', physicalUnits=True,
                    noH=True, numThreads=8)

# Build the tree
tree = cKDTree(cops2)

res_hmr_1 = []
res_hmr_2 = []
res_ms_1 = []
res_ms_2 = []
for ind, cop in enumerate(cops1):

    print(ind)

    # Find the nearest neighbour
    ds, inds = tree.query(cop, k=1)

    res_hmr_2.append(hmrs2[inds[1]])
    res_hmr_1.append(cop)
    res_hmr_2.append(ms2[inds[1]])
    res_hmr_1.append(ms1[ind])

fig = plt.figure()
ax1 = fig.add_subplot(121)
ax2 = fig.add_subplot(122)

cbar = ax1.hexbin(res_hmr_1, res_hmr_2, gridsize=50, mincnt=1, xscale='log', yscale='log', norm=LogNorm(),
                 linewidths=0.2, cmap='viridis', alpha=0.7)
cbar = ax2.hexbin(res_ms_1, res_ms_2, gridsize=50, mincnt=1, xscale='log', yscale='log', norm=LogNorm(),
                 linewidths=0.2, cmap='viridis', alpha=0.7)

ax1.set_xlabel(r"$R_{1/2, std} / [pkpc]$")
ax1.set_ylabel(r"$R_{1/2, hi} / [pkpc]$")
ax1.set_xlabel(r"$M_{\star, std} / M_\odot$")
ax1.set_ylabel(r"$M_{\star, hi} / M_\odot$")

fig.savefig("plots/res_galaxy_match.png", bbox_inches="tight")





