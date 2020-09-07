#!/cosma/home/dp004/dc-rope1/.conda/envs/flares-env/bin/python
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import astropy.units as u
import astropy.constants as const
from matplotlib.colors import LogNorm
import matplotlib.gridspec as gridspec
from scipy.stats import binned_statistic
import eagle_IO.eagle_IO as E
import seaborn as sns
from scipy.spatial import cKDTree
import pickle
import itertools
matplotlib.use("Agg")


sns.set_style("whitegrid")

snap = "010_z005p000"
z_str = snap.split('z')[1].split('p')
redshift = float(z_str[0] + '.' + z_str[1])
reg = "26"

path1 = "/cosma7/data/dp004/FLARES/FLARES-HD/FLARES_HR_" + reg + "/data/"
path2 = "/cosma7/data/dp004/FLARES/FLARES-1/G-EAGLE_" + reg + "/data/"

# Define the gravitational constant
G = (const.G.to(u.km ** 3 * u.M_sun ** -1 * u.s ** -2)).value

# Compute the mean density
mean_den = 2520**3 * 8.01 * 10**10 * u.M_sun / 3.2 ** 3 / u.Gpc ** 3 * (1 + redshift) ** 3
mean_den = mean_den.to(u.M_sun / u.km ** 3)

# Define the velocity space linking length
vlinkl_indp = (np.sqrt(G / 2) * (4 * np.pi * 200 * mean_den / 3) ** (1 / 6) * (1 + redshift) ** 0.5).value

print("Matching on...")
print(path1)
print(path2)

# Get the data for the standard res simulations
hmrs1 = E.read_array("SUBFIND", path1, snap, "Subhalo/HalfMassRad", noH=True, numThreads=8) * 1e3
ms1 = E.read_array("SUBFIND", path1, snap, "Subhalo/ApertureMeasurements/Mass/030kpc",
                  noH=True, numThreads=8)[:, 4] * 10**10
cops1 = E.read_array("SUBFIND", path1, snap, "Subhalo/CentreOfPotential", physicalUnits=True,
                    noH=True, numThreads=8)
vs1 = E.read_array("SUBFIND", path1, snap, "Subhalo/Velocity", physicalUnits=True,
                    noH=True, numThreads=8)
ns1 = E.read_array("SUBFIND", path1, snap, "Subhalo/SubLength", physicalUnits=True,
                    noH=True, numThreads=8)

# Get the data for the high resolution
hmrs2 = E.read_array("SUBFIND", path2, snap, "Subhalo/HalfMassRad", noH=True, numThreads=8) * 1e3
ms2 = E.read_array("SUBFIND", path2, snap, "Subhalo/ApertureMeasurements/Mass/030kpc",
                  noH=True, numThreads=8)[:, 4] * 10**10
cops2 = E.read_array("SUBFIND", path2, snap, "Subhalo/CentreOfPotential", physicalUnits=True,
                    noH=True, numThreads=8)
vs2 = E.read_array("SUBFIND", path1, snap, "Subhalo/Velocity", physicalUnits=True,
                    noH=True, numThreads=8)
print(vs2)
# Build the tree
tree = cKDTree(cops2)

res_hmr_1 = []
res_hmr_2 = []
res_ms_1 = []
res_ms_2 = []
for ind, cop in enumerate(cops1):

    # ===================== Matching on phase =====================

    # Define the phase space linking length
    vlinkl = vlinkl_indp * (8.01 * 10**10)**(1 / 3) * ns1[ind] ** (1 / 3)
    phases = np.concatenate((cops2 / 0.003, vs2 / vlinkl), axis=1)
    ptree = cKDTree(phases)

    # Find the 5 nearest neighbours
    ds, inds = ptree.query(phases[ind], k=1)

    res_hmr_2.append(hmrs2[inds])
    res_hmr_1.append(hmrs1[ind])
    res_ms_2.append(ms2[inds])
    res_ms_1.append(ms1[ind])

    # # ===================== Matching on COP and Velocity =====================
    #
    # # Find the 10 nearest neighbours
    # ds, inds = tree.query(cop, k=10)
    #
    # # Build velocity tree for these particles
    # vtree = cKDTree(vs2[inds])
    #
    # # Find the nearest neighbour in velocity space
    # dvs, vind = vtree.query(vs1[ind], k=1)
    #
    # res_hmr_2.append(hmrs2[inds[vind]])
    # res_hmr_1.append(hmrs1[ind])
    # res_ms_2.append(ms2[inds[vind]])
    # res_ms_1.append(ms1[ind])

    # ===================== Matching on COP and Mass =====================

    # # Find the 5 nearest neighbours
    # ds, inds = tree.query(cop, k=5)

    # nn_ms = ms2[inds]
    # nn_max = ms1[ind]
    # nn_ind = np.argmin(np.abs(nn_ms - nn_max))
    #
    # res_hmr_2.append(hmrs2[inds[nn_ind]])
    # res_hmr_1.append(hmrs1[ind])
    # res_ms_2.append(ms2[inds[nn_ind]])
    # res_ms_1.append(ms1[ind])

    # # ===================== Matching only on COP =====================
    #
    # # Find the 5 nearest neighbours
    # ds, inds = tree.query(cop, k=1)
    #
    # res_hmr_2.append(hmrs2[inds])
    # res_hmr_1.append(hmrs1[ind])
    # res_ms_2.append(ms2[inds])
    # res_ms_1.append(ms1[ind])

res_hmr_2 = np.array(res_hmr_2)
res_hmr_1 = np.array(res_hmr_1)
res_ms_2 = np.array(res_ms_2)
res_ms_1 = np.array(res_ms_1)

okinds = np.logical_and(res_hmr_2 > 0, res_hmr_1 > 0)
res_hmr_1 = res_hmr_1[okinds]
res_hmr_2 = res_hmr_2[okinds]

okinds = np.logical_and(res_ms_2 > 0, res_ms_1 > 0)
res_ms_1 = res_ms_1[okinds]
res_ms_2 = res_ms_2[okinds]

fig = plt.figure(figsize=(12, 8))
ax1 = fig.add_subplot(121)
ax2 = fig.add_subplot(122)

cbar = ax1.hexbin(res_hmr_1, res_hmr_2, gridsize=50, mincnt=1, xscale="log", yscale="log", norm=LogNorm(),
                 linewidths=0.2, cmap="viridis", alpha=0.7)
cbar = ax2.hexbin(res_ms_1, res_ms_2, gridsize=50, mincnt=1, xscale="log", yscale="log", norm=LogNorm(),
                 linewidths=0.2, cmap="viridis", alpha=0.7)

ax1.plot([res_hmr_1.min(), res_hmr_1.max()], [res_hmr_1.min(), res_hmr_1.max()])
ax2.plot([res_ms_1.min(), res_ms_1.max()], [res_ms_1.min(), res_ms_1.max()])

ax1.set_xlabel(r"$R_{1/2, std} / [pkpc]$")
ax1.set_ylabel(r"$R_{1/2, hi} / [pkpc]$")
ax2.set_xlabel(r"$M_{\star, std} / M_\odot$")
ax2.set_ylabel(r"$M_{\star, hi} / M_\odot$")

fig.savefig("plots/res_galaxy_match_cop+vel+mass.png", bbox_inches="tight")





