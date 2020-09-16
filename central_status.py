#!/cosma/home/dp004/dc-rope1/.conda/envs/flares-env/bin/python
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1 import make_axes_locatable
import astropy.units as u
import astropy.constants as cons
from astropy.cosmology import Planck13 as cosmo
from matplotlib.colors import LogNorm
import eagle_IO.eagle_IO as E
import seaborn as sns
import h5py
import os
from unyt import mh, cm, Gyr, g, Msun, Mpc
matplotlib.use('Agg')

sns.set_style('whitegrid')


regions = []
for reg in range(20, 21):

    if reg < 10:
        regions.append('0' + str(reg))
    else:
        regions.append(str(reg))

snaps = ['000_z015p000', '001_z014p000', '002_z013p000', '003_z012p000', '004_z011p000', '005_z010p000',
         '006_z009p000', '007_z008p000', '008_z007p000', '009_z006p000', '010_z005p000', '011_z004p770']

subgrps = []
dists = []
shmrs = []
ghmrs = []

for reg in regions:

    for snap in snaps:

        print(reg, snap)

        path = '/cosma/home/dp004/dc-rope1/FLARES/FLARES-1/G-EAGLE_' + reg + '/data'

        try:
            subfind_grp_ids = E.read_array('SUBFIND', path, snap, 'Subhalo/GroupNumber', numThreads=8)
            subfind_subgrp_ids = E.read_array('SUBFIND', path, snap, 'Subhalo/SubGroupNumber', numThreads=8)
            gal_hmrs = E.read_array('SUBFIND', path, snap, 'Subhalo/HalfMassRad', numThreads=8,
                                    noH=True, physicalUnits=True) * 1e3
            coms = E.read_array('SUBFIND', path, snap, 'Subhalo/CentreOfMass', numThreads=8,
                                noH=True, physicalUnits=True) * 1e3
            grp_coms = E.read_array('SUBFIND', path, snap, 'FOF/GroupCentreOfPotential', numThreads=8,
                                    noH=True, physicalUnits=True) * 1e3
            ms = E.read_array('SUBFIND', path, snap, 'Subhalo/ApertureMeasurements/Mass/030kpc',
                              noH=True, numThreads=8)[:, 4] * 10**10
        except ValueError:
            continue
        except KeyError:
            continue
        except OSError:
            continue

        for (ind, grp), subgrp in zip(enumerate(subfind_grp_ids), subfind_subgrp_ids):

            if ms[ind] < 1e9:
                continue

            grpcom = grp_coms[grp, :]
            com = coms[ind, :]

            subgrps.append(subgrp)
            dists.append(np.sqrt(np.sum((grpcom - com)**2)))
            shmrs.append(gal_hmrs[ind, 4])
            ghmrs.append(gal_hmrs[ind, 1])


fig = plt.figure()
gs = gridspec.GridSpec(1, 2)
gs.update(wspace=0.0, hspace=0.0)
ax1 = fig.add_subplot(gs[0, 0])
ax2 = fig.add_subplot(gs[0, 1])
print(shmrs, ghmrs, np.array(subgrps == 0))
print(shmrs, ghmrs, dists)
im1 = ax1.hexbin(shmrs, ghmrs, C=np.array(subgrps == 0, dtype=int), gridsize=100, mincnt=1, cmap="coolwarm",
                 vmin=0, vmax=1, linewidths=0.2, reduce_C_function=np.mean, xscale='log', yscale='log')
im2 = ax2.hexbin(shmrs, ghmrs, C=dists, gridsize=100, mincnt=1, cmap="plasma", linewidths=0.2,
                 reduce_C_function=np.mean, xscale='log', yscale='log')

ax1.set_xlabel('$R_{1/2,*}/ [\mathrm{pkpc}]$')
ax2.set_xlabel('$R_{1/2,*}/ [\mathrm{pkpc}]$')
ax1.set_ylabel('$R_{1/2,Gas}/ [\mathrm{pkpc}]$')

divider = make_axes_locatable(ax1)
cax = divider.append_axes('top', size='5%', pad=0.05)
cbar1 = fig.colorbar(im1, cax=cax, orientation='horizontal')
divider = make_axes_locatable(ax2)
cax = divider.append_axes('top', size='5%', pad=0.05)
cbar2 = fig.colorbar(im2, cax=cax, orientation='horizontal')
cbar1.set_label('Central?')
cbar2.set_label('$\Delta D_{\mathrm{Group}-\mathrm{Galaxy}}$')

fig.savefig("central_status.png", bbox_inches="tight")



