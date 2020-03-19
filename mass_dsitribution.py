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
matplotlib.use('Agg')


def get_parts_in_gal(ID, poss, IDs):

    # Get particle positions
    gal_poss = poss[IDs == ID, :]

    return gal_poss


def get_parts_around_gal(all_poss, mean, lim):

    # Get galaxy particle indices
    seps = np.abs(all_poss - mean)
    xcond = seps[:, 0] < lim
    ycond = seps[:, 1] < lim
    zcond = seps[:, 2] < lim

    # Get particle positions
    surnd_poss = all_poss[np.logical_and(xcond, ycond, zcond), :]

    return surnd_poss


def create_img(ID, res, all_poss, poss, IDs):

    # Get the galaxy particle positions
    gal_poss = get_parts_in_gal(ID, poss, IDs)

    # Compute mean particle position
    mean = gal_poss.mean(axis=0)

    # Centre galaxy on mean
    gal_poss -= mean

    # Find the max and minimum position on each axis
    xmax, xmin = np.max(gal_poss[:, 0]), np.min(gal_poss[:, 0])
    ymax, ymin = np.max(gal_poss[:, 1]), np.min(gal_poss[:, 1])
    zmax, zmin = np.max(gal_poss[:, 2]), np.min(gal_poss[:, 2])

    # Set up lists of mins and maximums
    mins = [xmin, ymin, zmin]
    maxs = [xmax, ymax, ymax]

    # Compute extent 3D
    lim = np.max([np.abs(xmin), np.abs(ymin), np.abs(zmin), xmax, ymax, zmax]) + 0.05

    # Get the surrounding distribution
    surnd_poss = get_parts_around_gal(all_poss, mean, lim)

    # Centre particle distribution
    surnd_poss -= mean

    # Set up dictionaries to store images
    galimgs = {}
    surundimgs = {}
    extents = {}

    for (i, j) in [(0, 1), (0, 2), (1, 2)]:

        # Compute extent for the 2D square image
        dim = np.max([np.abs(mins[i]), np.abs(mins[j]), maxs[i], maxs[j]]) + 0.05
        extents[str(i) + '-' + str(j)] = [-dim, dim, -dim, dim]
        posrange = ((-dim, dim), (-dim, dim))

        # Create images
        galimgs[str(i) + '-' + str(j)], _, _ = np.histogram2d(gal_poss[:, [i, j]], bins=dim / res, range=posrange)
        surundimgs[str(i) + '-' + str(j)], _, _ = np.histogram2d(surnd_poss[:, [i, j]], bins=dim / res, range=posrange)

    return galimgs, surundimgs, extents


def img_main(path, snap, reg, res, part_type, npart_lim=10**4):

    # Load all necessary arrays
    subgrp_ids = E.read_array('PARTDATA', path, snap, 'PartType' + str(part_type) + '/SubGroupNumber', numThreads=8)
    subnpart = E.read_array('SUBFIND', path, snap, 'Subhalo/SubLengthType', numThreads=8)[:, part_type]
    subID = E.read_array('SUBFIND', path, snap, 'Subhalo/SubGroupNumber', numThreads=8)
    all_poss = E.read_array('SNAP', path, snap, 'PartType' + str(part_type) + '/Coordinates', noH=True, numThreads=8)
    part_ids = E.read_array('SNAP', path, snap, 'PartType' + str(part_type) + '/ParticleIDs', numThreads=8)
    group_part_ids = E.read_array('PARTDATA', path, snap, 'PartType' + str(part_type) + '/ParticleIDs',
                                  numThreads=8)

    # Translate ID into indices
    ind_to_pid = {}
    pid_to_ind = {}
    for ind, pid in enumerate(part_ids):
        ind_to_pid[ind] = pid
        pid_to_ind[pid] = ind

    # Get IDs
    ids = set(subID[subnpart > npart_lim])

    # Get the particles in the halos
    halo_id_part_inds = {}
    for pid, simid in zip(group_part_ids, subgrp_ids):
        if simid in ids:
            continue
        try:
            halo_id_part_inds.setdefault(simid, set()).update({pid_to_ind[pid]})
        except KeyError:
            ind_to_pid[len(part_ids) + 1] = pid
            pid_to_ind[pid] = len(part_ids) + 1
            halo_id_part_inds.setdefault(simid, set()).update({pid_to_ind[pid]})

    print('There are', len(ids), 'galaxies above the cutoff')

    # Get the position of each of these galaxies
    all_gal_poss = {}
    for id in ids:

        all_gal_poss[id] = all_poss[list(halo_id_part_inds[id]), :]

    axlabels = [r'$x$', r'$y$', r'$z$']

    # Create images for these galaxies
    for id in ids:

        # Get the images
        galimgs, surundimgs, extents = create_img(id, res, all_poss, all_gal_poss[id], subgrp_ids)

        # Loop over dimensions
        for key in galimgs.keys():

            # Extract data
            i, j = key.split('-')
            extent = extents[key]
            galimg = galimgs[key]
            surundimg = surundimgs[key]

            # Set up figure
            fig = plt.figure()
            ax1 = fig.add_subplot(121)
            ax2 = fig.add_subplot(121)

            # Draw images
            ax1.imshow(np.arcsinh(galimg), extent=extent, cmap='Greys')
            ax2.imshow(np.arcsinh(surundimg), extent=extent, cmap='Greys')

            # Label axes
            ax1.set_xlabel(axlabels[i])
            ax1.set_ylabel(axlabels[j])
            ax2.set_xlabel(axlabels[i])

            fig.savefig('plots/massdistributions/reg' + str(reg) + '_snap' + snap +
                        '_gal' + str(id) + '_coords' + key + 'png',
                        bbox_inches='tight')

            plt.close(fig)

        break


# Define comoving softening length in Mpc
csoft = 0.001802390/0.677

# Define resolution
res = csoft / 2
print(100 / res, 'pixels in', '100 kpc')

# Define region variables
reg = 00
snap = '010_z005p000'
path = '/cosma7/data/dp004/dc-love2/data/G-EAGLE/geagle_' + reg + '/data/'

img_main(path, snap, reg, res, part_type=4, npart_lim=10**4)
