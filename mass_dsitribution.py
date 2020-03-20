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


def create_img(res, all_poss, gal_poss, mean):

    # Centre galaxy on mean
    gal_poss -= mean

    # Find the max and minimum position on each axis
    xmax, xmin = np.max(gal_poss[:, 0]), np.min(gal_poss[:, 0])
    ymax, ymin = np.max(gal_poss[:, 1]), np.min(gal_poss[:, 1])
    zmax, zmin = np.max(gal_poss[:, 2]), np.min(gal_poss[:, 2])

    # Set up lists of mins and maximums
    mins = [xmin, ymin, zmin]
    maxs = [xmax, ymax, zmax]

    # Compute extent 3D
    lim = np.max([np.abs(xmin), np.abs(ymin), np.abs(zmin), xmax, ymax, zmax])

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
        # dim = np.max([np.abs(mins[i]), np.abs(mins[j]), maxs[i], maxs[j]])
        dim = 0.08
        extents[str(i) + '-' + str(j)] = [-dim, dim, -dim, dim]
        posrange = ((-dim, dim), (-dim, dim))

        # Create images
        galimgs[str(i) + '-' + str(j)], gxbins, gybins = np.histogram2d(gal_poss[:, i], gal_poss[:, j],
                                                                        bins=int(dim / res), range=posrange)
        surundimgs[str(i) + '-' + str(j)], sxbins, sybins = np.histogram2d(surnd_poss[:, i], surnd_poss[:, j],
                                                                           bins=int(dim / res), range=posrange)

    return galimgs, surundimgs, extents


def img_main(path, snap, reg, res, soft, part_types=(4, 0, 1), npart_lim=10**3, imgtype='compact'):

    # Get the redshift
    z_str = snap.split('z')[1].split('p')
    z = float(z_str[0] + '.' + z_str[1])

    # Initialise galaxy position dictionary
    all_gal_poss = {}
    all_poss = {}
    means = {}

    for part_type in part_types:

        print('Loading particle type', part_type)

        # Load all necessary arrays
        subgrp_ids = E.read_array('PARTDATA', path, snap, 'PartType' + str(part_type) + '/SubGroupNumber', numThreads=8)
        all_poss[part_type] = E.read_array('SNAP', path, snap, 'PartType' + str(part_type) + '/Coordinates', noH=True, numThreads=8)
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

        # Get IDs
        if part_type == 4 and imgtype == 'compact':
            half_mass_rads = E.read_array('SUBFIND', path, snap, 'Subhalo/HalfMassRad', noH=True, numThreads=8)[:, 4]
            grp_ID = E.read_array('SUBFIND', path, snap, 'Subhalo/GroupNumber', numThreads=8)
            subgrp_ID = E.read_array('SUBFIND', path, snap, 'Subhalo/SubGroupNumber', numThreads=8)

            # Get the half mass radii for each group
            half_mass_rads_dict = {}
            for r, g, sg in zip(half_mass_rads, grp_ID, subgrp_ID):
                half_mass_rads_dict[str(g) + '.' + str(sg + 1)] = r

            print(len(half_mass_rads), len(grp_ID), len(subgrp_ID))

            # Get the IDs above the npart threshold
            ids, counts = np.unique(halo_ids, return_counts=True)
            ids = set(ids[counts > npart_lim])

            for id in list(ids):
                if half_mass_rads_dict[str(id)] > soft / (1 + z) * 1.2:
                    ids.remove(id)

        elif part_type == 4 and imgtype == 'DMless':
            masses = E.read_array('SUBFIND', path, snap, 'Subhalo/ApertureMeasurements/Mass/030kpc', noH=True,
                                  numThreads=8)
            grp_ID = E.read_array('SUBFIND', path, snap, 'Subhalo/GroupNumber', numThreads=8)
            subgrp_ID = E.read_array('SUBFIND', path, snap, 'Subhalo/SubGroupNumber', numThreads=8)

            # Get the half mass radii for each group
            masses_dict = {}
            for ms, g, sg in zip(masses, grp_ID, subgrp_ID):
                masses_dict[str(g) + '.' + str(sg + 1)] = ms

            # Get the IDs above the npart threshold
            ids, counts = np.unique(halo_ids, return_counts=True)
            ids = set(ids[ids >= 0])

            for id in list(ids):
                if str(id).split('.')[1] == '1073741825':
                    ids.remove(id)
                elif any(masses_dict[str(id)][[0, 1, 5]] > 0.0):
                    ids.remove(id)

        elif part_type == 4 and imgtype not in ['compact', 'DMless']:
            print('Invalid type, should be one of:', ['compact', 'DMless'])

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
        all_gal_poss[part_type] = {}
        for id in ids:

            try:
                all_gal_poss[part_type][id] = all_poss[part_type][list(halo_id_part_inds[id]), :]
            except KeyError:
                all_gal_poss[part_type][id] = np.array([])

            if part_type == 4:
                means[id] = all_gal_poss[part_type][id].mean(axis=0)

    print('Extracted galaxy positions')

    axlabels = [r'$x$', r'$y$', r'$z$']

    # Create images for these galaxies
    for id in ids:

        galimgs = {}
        surundimgs = {}
        extents = {}

        for part_type in part_types:

            print('Computing images for', id, 'and particle type', part_type)

            # Get the images
            galimgs[part_type], surundimgs[part_type], extents[part_type] = create_img(res, all_poss[part_type],
                                                                                       all_gal_poss[part_type][id],
                                                                                       means[id])

        # Loop over dimensions
        for key in galimgs[4].keys():

            i, j = key.split('-')

            # Set up figure
            fig = plt.figure()
            ax1 = fig.add_subplot(321)
            ax2 = fig.add_subplot(322)
            ax3 = fig.add_subplot(323)
            ax4 = fig.add_subplot(324)
            ax5 = fig.add_subplot(325)
            ax6 = fig.add_subplot(326)

            # Draw images
            ax1.imshow(np.arcsinh(galimgs[1][key]), extent=extents[1][key], cmap='Greys')
            ax2.imshow(np.arcsinh(surundimgs[1][key]), extent=extents[1][key], cmap='Greys')
            ax3.imshow(np.arcsinh(galimgs[0][key]), extent=extents[0][key], cmap='Greys')
            ax4.imshow(np.arcsinh(surundimgs[0][key]), extent=extents[0][key], cmap='Greys')
            ax5.imshow(np.arcsinh(galimgs[4][key]), extent=extents[4][key], cmap='Greys')
            ax6.imshow(np.arcsinh(surundimgs[4][key]), extent=extents[4][key], cmap='Greys')

            # Label axes
            ax5.set_xlabel(axlabels[int(i)])
            ax3.set_ylabel(axlabels[int(j)])
            ax6.set_xlabel(axlabels[int(i)])

            # Set titles
            ax2.set_title('Surrounding particles')
            ax1.set_title('Galaxy particles')

            fig.savefig('plots/massdistributions/'+ imgtype + '_reg' + str(reg) + '_snap' + snap +
                        '_gal' + str(id).split('.')[0] + 'p' + str(id).split('.')[1] + '_coords' + key + 'png',
                        bbox_inches='tight', dpi=300)

            plt.close(fig)


# Define comoving softening length in Mpc
csoft = 0.001802390/0.677

# Define resolution
res = csoft / 20
print(100 / res, 'pixels in', '100 kpc')

# Define region variables
reg = '0001'
snap = '010_z005p000'
path = '/cosma7/data/dp004/dc-love2/data/G-EAGLE/geagle_' + reg + '/data/'

img_main(path, snap, reg, res, soft=csoft, npart_lim=10**4, imgtype='DMless')
