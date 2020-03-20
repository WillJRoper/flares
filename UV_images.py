#!/cosma/home/dp004/dc-rope1/.conda/envs/flares-env/bin/python
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import astropy.units as u
from matplotlib.colors import LogNorm
import matplotlib.gridspec as gridspec
import eagle_IO as E
import seaborn as sns
import pickle
import itertools
import os
os.environ['FLARE'] = '/cosma7/data/dp004/dc-wilk2/flare/data/'
from SynthObs.SED import models
matplotlib.use('Agg')


def create_img(res, gal_poss, mean, lumins,  dim):

    # Centre galaxy on mean
    if gal_poss.shape[0] != 0:
        gal_poss -= mean

    # Set up dictionaries to store images
    galimgs = {}
    extents = {}

    for (i, j) in [(0, 1), (0, 2), (1, 2)]:

        # Compute extent for the 2D square image
        extents[str(i) + '-' + str(j)] = [-dim, dim, -dim, dim]
        posrange = ((-dim, dim), (-dim, dim))

        # Create images
        galimgs[str(i) + '-' + str(j)], gxbins, gybins = np.histogram2d(gal_poss[:, i], gal_poss[:, j],
                                                                        bins=int(dim / res), weights=lumins,
                                                                        range=posrange)

    return galimgs, extents


def img_main(path, snap, reg, res, npart_lim=10**3, lim=0.08):

    # Get the redshift
    z_str = snap.split('z')[1].split('p')
    z = float(z_str[0] + '.' + z_str[1])

    # Define stellar particle type
    part_type = 4

    # Initialise galaxy position dictionaries
    all_gal_poss = {}
    means = {}

    # Load all necessary arrays
    subgrp_ids = E.read_array('PARTDATA', path, snap, 'PartType' + str(part_type) + '/SubGroupNumber', numThreads=8)
    all_poss = E.read_array('SNAP', path, snap, 'PartType' + str(part_type) + '/Coordinates', noH=True, numThreads=8)
    part_ids = E.read_array('SNAP', path, snap, 'PartType' + str(part_type) + '/ParticleIDs', numThreads=8)
    group_part_ids = E.read_array('PARTDATA', path, snap, 'PartType' + str(part_type) + '/ParticleIDs', numThreads=8)
    grp_ids = E.read_array('PARTDATA', path, snap, 'PartType' + str(part_type) + '/GroupNumber', numThreads=8)
    halo_ids = np.zeros_like(grp_ids, dtype=float)
    for (ind, g), sg in zip(enumerate(grp_ids), subgrp_ids):
        halo_ids[ind] = float(str(g) + '.' + str(sg + 1))

    # Translate ID into indices
    ind_to_pid = {}
    pid_to_ind = {}
    for ind, pid in enumerate(part_ids):
        ind_to_pid[ind] = pid
        pid_to_ind[pid] = ind

    # Get the IDs above the npart threshold
    ids, counts = np.unique(halo_ids, return_counts=True)
    ids = set(ids[counts > npart_lim])

    # Get the particles in the halos
    halo_id_part_inds = {}
    for pid, simid in zip(group_part_ids, halo_ids):
        if simid not in ids:
            continue
        try:
            halo_id_part_inds.setdefault(simid, set()).update({pid_to_ind[pid]})
        except KeyError:
            ind_to_pid[len(part_ids) + 1] = pid
            pid_to_ind[pid] = len(part_ids) + 1
            halo_id_part_inds.setdefault(simid, set()).update({pid_to_ind[pid]})

    print('There are', len(ids), 'galaxies above the cutoff')

    # If there are no galaxies exit
    if len(ids) == 0:
        return

    # Get the position of each of these galaxies
    for id in ids:

        all_gal_poss[id] = all_poss[list(halo_id_part_inds[id]), :]

        means[id] = all_gal_poss[id].mean(axis=0)

    print('Extracted galaxy positions')

    axlabels = [r'$x$', r'$y$', r'$z$']

    # Create images for these galaxies
    for id in ids:

        print('Computing images for', id)

        lumins = np.ones_like(all_gal_poss[id])

        # Get the images
        galimgs, extents = create_img(res, all_gal_poss[id], means[id], lumins, lim)

        # Loop over dimensions
        for key in galimgs.keys():

            i, j = key.split('-')

            # Set up figure
            fig = plt.figure()
            ax1 = fig.add_subplot(111)

            # Draw images
            ax1.imshow(np.arcsinh(galimgs[1][key]), extent=extents[1][key], cmap='Greys')

            # Label axes
            ax1.set_xlabel(axlabels[int(i)])
            ax1.set_ylabel(axlabels[int(j)])

            fig.savefig('plots/massdistributions/UV_reg' + str(reg) + '_snap' + snap +
                        '_gal' + str(id).split('.')[0] + 'p' + str(id).split('.')[1] + '_coords' + key + 'png',
                        bbox_inches='tight', dpi=300)

            plt.close(fig)


# Define comoving softening length in Mpc
csoft = 0.001802390/0.677

# Define resolution
res = csoft
print(100 / res, 'pixels in', '100 kpc')

# Define region variables
reg = '0000'
snap = '010_z005p000'
path = '/cosma7/data/dp004/dc-love2/data/G-EAGLE/geagle_' + reg + '/data/'

img_main(path, snap, reg, res, npart_lim=10**4)