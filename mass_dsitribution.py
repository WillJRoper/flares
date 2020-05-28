#!/cosma/home/dp004/dc-rope1/.conda/envs/flares-env/bin/python
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import eagle_IO.eagle_IO as E
from matplotlib.lines import Line2D
import matplotlib.gridspec as gridspec
import seaborn as sns
import pickle
import itertools
matplotlib.use('Agg')

sns.set_style("white")


def get_part_ids(sim, snapshot, part_type, all_parts=False):

    # Get the particle IDs
    if all_parts:
        part_ids = E.read_array('SNAP', sim, snapshot, 'PartType' + str(part_type) + '/ParticleIDs', numThreads=8)
    else:
        part_ids = E.read_array('PARTDATA', sim, snapshot, 'PartType' + str(part_type) + '/ParticleIDs',
                                numThreads=8)

    # Extract the halo IDs (group names/keys) contained within this snapshot
    group_part_ids = E.read_array('PARTDATA', sim, snapshot, 'PartType' + str(part_type) + '/ParticleIDs',
                                  numThreads=8)
    grp_ids = E.read_array('PARTDATA', sim, snapshot, 'PartType' + str(part_type) + '/GroupNumber',
                           numThreads=8)
    subgrp_ids = E.read_array('PARTDATA', sim, snapshot, 'PartType' + str(part_type) + '/SubGroupNumber',
                              numThreads=8)

    # Remove particles not associated to a subgroup
    okinds = subgrp_ids != 1073741824
    group_part_ids = group_part_ids[okinds]
    grp_ids = grp_ids[okinds]
    subgrp_ids = subgrp_ids[okinds]

    # Convert IDs to float(groupNumber.SubGroupNumber) format, i.e. group 1 subgroup 11 = 1.00011
    halo_ids = np.zeros(grp_ids.size, dtype=float)
    for (ind, g), sg in zip(enumerate(grp_ids), subgrp_ids):
        halo_ids[ind] = float(str(int(g)) + '.%05d' % int(sg))

    # Sort particle IDs
    unsort_part_ids = np.copy(part_ids)
    sinds = np.argsort(part_ids)
    part_ids = part_ids[sinds]

    # Get the index of particles in the snapshot array from the in group array
    sorted_index = np.searchsorted(part_ids, group_part_ids)
    yindex = np.take(sinds, sorted_index, mode="raise")
    mask = unsort_part_ids[yindex] != group_part_ids
    result = np.ma.array(yindex, mask=mask)

    # Apply mask to the id arrays
    part_groups = halo_ids[np.logical_not(result.mask)]
    parts_in_groups = result.data[np.logical_not(result.mask)]

    # Produce a dictionary containing the index of particles in each halo
    halo_part_inds = {}
    for ind, grp in zip(parts_in_groups, part_groups):
        halo_part_inds.setdefault(grp, set()).update({ind})

    return halo_part_inds, halo_ids


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


def create_img(res, all_poss, gal_poss, mean, lim):

    # Centre galaxy on mean
    if gal_poss.shape[0] != 0:
        gal_poss -= mean

    # # Find the max and minimum position on each axis
    # xmax, xmin = np.max(gal_poss[:, 0]), np.min(gal_poss[:, 0])
    # ymax, ymin = np.max(gal_poss[:, 1]), np.min(gal_poss[:, 1])
    # zmax, zmin = np.max(gal_poss[:, 2]), np.min(gal_poss[:, 2])

    # # Set up lists of mins and maximums
    # mins = [xmin, ymin, zmin]
    # maxs = [xmax, ymax, zmax]

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
        dim = lim
        extents[str(i) + '-' + str(j)] = [-dim, dim, -dim, dim]
        posrange = ((-dim, dim), (-dim, dim))

        # Create images
        try:
            galimgs[str(i) + '-' + str(j)], gxbins, gybins = np.histogram2d(gal_poss[:, i], gal_poss[:, j],
                                                                            bins=int(dim / res), range=posrange)
        except IndexError:
            galimgs[str(i) + '-' + str(j)] = np.array([])
        surundimgs[str(i) + '-' + str(j)], sxbins, sybins = np.histogram2d(surnd_poss[:, i], surnd_poss[:, j],
                                                                           bins=int(dim / res), range=posrange)

    return galimgs, surundimgs, extents


def img_main(path, snap, reg, res, soft, part_types=(4, 0, 1), npart_lim=10**3, imgtype='compact', lim=0.035):

    # Get the redshift
    z_str = snap.split('z')[1].split('p')
    z = float(z_str[0] + '.' + z_str[1])

    # Initialise galaxy position dictionary
    all_gal_poss = {}
    all_poss = {}
    means = {}

    assert part_types[0] == 4, "Initial particle type must be 4 (star)"

    for part_type in part_types:

        print('Loading particle type', part_type)

        # Get positions
        all_poss[part_type] = E.read_array('SNAP', path, snap, 'PartType' + str(part_type) + '/Coordinates',
                                           noH=True, numThreads=8)

        # Get the particles in the halos
        halo_id_part_inds, halo_ids = get_part_ids(path, snap, part_type, all_parts=True)

        # Get IDs
        if part_type == 4 and imgtype == 'compact':
            half_mass_rads = E.read_array('SUBFIND', path, snap, 'Subhalo/HalfMassRad', noH=True, numThreads=8)[:, 4]
            cops = E.read_array('SUBFIND', path, snap, 'Subhalo/CentreOfPotential', noH=True,
                                  numThreads=8)
            grp_ID = E.read_array('SUBFIND', path, snap, 'Subhalo/GroupNumber', numThreads=8)
            subgrp_ID = E.read_array('SUBFIND', path, snap, 'Subhalo/SubGroupNumber', numThreads=8)

            # Get the half mass radii for each group
            half_mass_rads_dict = {}
            cops_dict = {}
            for r, cop, g, sg in zip(half_mass_rads, cops, grp_ID, subgrp_ID):
                half_mass_rads_dict[float(str(int(g)) + '.%05d' % int(sg))] = r
                cops_dict[float(str(int(g)) + '.%05d' % int(sg))] = cop

            # Get the IDs above the npart threshold
            ids, counts = np.unique(halo_ids, return_counts=True)
            ids = set(ids[counts > npart_lim])

            for id in list(ids):
                if half_mass_rads_dict[id] > soft / (1 + z) * 1.2:
                    ids.remove(id)

        elif part_type == 4 and imgtype == 'DMless':
            masses = E.read_array('SUBFIND', path, snap, 'Subhalo/ApertureMeasurements/Mass/030kpc', noH=True,
                                  numThreads=8)
            cops = E.read_array('SUBFIND', path, snap, 'Subhalo/CentreOfPotential', noH=True,
                                  numThreads=8)
            grp_ID = E.read_array('SUBFIND', path, snap, 'Subhalo/GroupNumber', numThreads=8)
            subgrp_ID = E.read_array('SUBFIND', path, snap, 'Subhalo/SubGroupNumber', numThreads=8)

            # Get the half mass radii for each group
            masses_dict = {}
            cops_dict = {}
            for ms,cop,  g, sg in zip(masses, cops, grp_ID, subgrp_ID):
                masses_dict[float(str(int(g)) + '.%05d' % int(sg))] = ms
                cops_dict[float(str(int(g)) + '.%05d' % int(sg))] = cop

            # Get the IDs above the npart threshold
            ids, counts = np.unique(halo_ids, return_counts=True)
            ids = set(ids[ids >= 0])

            for id in list(ids):
                if str(id).split('.')[1] == '1073741825':
                    ids.remove(id)
                elif any(masses_dict[id][[0, 1, 5]] > 0.0):
                    ids.remove(id)

        elif part_type == 4 and imgtype not in ['compact', 'DMless']:
            print('Invalid type, should be one of:', ['compact', 'DMless'])

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
                means[id] = cops_dict[id]

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
                                                                                       means[id], lim)

        # Remove empty images
        imgshape = galimgs[4]['0-1'].shape
        for key in galimgs[4].keys():
            for part_type in part_types:
                if part_type == 4:
                    continue
                if galimgs[part_type][key].shape == (0,):
                    galimgs[part_type][key] = np.full(imgshape, np.nan)

        # Loop over dimensions
        for key in galimgs[4].keys():

            i, j = key.split('-')

            # Set up figure
            fig = plt.figure(figsize=(4, 7))
            gs = gridspec.GridSpec(ncols=2, nrows=3)
            gs.update(wspace=0.0, hspace=0.0)
            ax1 = fig.add_subplot(gs[0, 0])
            ax2 = fig.add_subplot(gs[0, 1])
            ax3 = fig.add_subplot(gs[1, 0])
            ax4 = fig.add_subplot(gs[1, 1])
            ax5 = fig.add_subplot(gs[2, 0])
            ax6 = fig.add_subplot(gs[2, 1])

            # Draw images
            ax1.imshow(np.arcsinh(galimgs[1][key]), extent=extents[1][key], cmap='Greys')
            ax2.imshow(np.arcsinh(surundimgs[1][key]), extent=extents[1][key], cmap='Greys')
            ax3.imshow(np.arcsinh(galimgs[0][key]), extent=extents[0][key], cmap='Greys')
            ax4.imshow(np.arcsinh(surundimgs[0][key]), extent=extents[0][key], cmap='Greys')
            ax5.imshow(np.arcsinh(galimgs[4][key]), extent=extents[4][key], cmap='Greys')
            ax6.imshow(np.arcsinh(surundimgs[4][key]), extent=extents[4][key], cmap='Greys')

            circle1 = plt.Circle((0., 0.), soft, facecolor='none', edgecolor='r', linestyle='-')
            circle2 = plt.Circle((0., 0.), soft, facecolor='none', edgecolor='r', linestyle='-')
            circle3 = plt.Circle((0., 0.), soft, facecolor='none', edgecolor='r', linestyle='-')
            circle4 = plt.Circle((0., 0.), soft, facecolor='none', edgecolor='r', linestyle='-')
            circle5 = plt.Circle((0., 0.), soft, facecolor='none', edgecolor='r', linestyle='-')
            circle6 = plt.Circle((0., 0.), soft, facecolor='none', edgecolor='r', linestyle='-')

            app1 = plt.Circle((0., 0.), 0.03, facecolor='none', edgecolor='g', linestyle='--')
            app2 = plt.Circle((0., 0.), 0.03, facecolor='none', edgecolor='g', linestyle='--')
            app3 = plt.Circle((0., 0.), 0.03, facecolor='none', edgecolor='g', linestyle='--')
            app4 = plt.Circle((0., 0.), 0.03, facecolor='none', edgecolor='g', linestyle='--')
            app5 = plt.Circle((0., 0.), 0.03, facecolor='none', edgecolor='g', linestyle='--')
            app6 = plt.Circle((0., 0.), 0.03, facecolor='none', edgecolor='g', linestyle='--')

            ax1.add_artist(circle1)
            ax2.add_artist(circle2)
            ax3.add_artist(circle3)
            ax4.add_artist(circle4)
            ax5.add_artist(circle5)
            ax6.add_artist(circle6)

            ax1.add_artist(app1)
            ax2.add_artist(app2)
            ax3.add_artist(app3)
            ax4.add_artist(app4)
            ax5.add_artist(app5)
            ax6.add_artist(app6)

            # Label axes
            ax5.set_xlabel(axlabels[int(i)])
            ax3.set_ylabel(axlabels[int(j)])
            ax6.set_xlabel(axlabels[int(i)])

            ax1.text(0.1, 0.85, 'Dark Matter', bbox=dict(boxstyle="round,pad=0.3", fc='w', ec="k", lw=1, alpha=0.8),
                    transform=ax1.transAxes, horizontalalignment='left', fontsize=6)
            ax3.text(0.1, 0.85, 'Gas', bbox=dict(boxstyle="round,pad=0.3", fc='w', ec="k", lw=1, alpha=0.8),
                    transform=ax3.transAxes, horizontalalignment='left', fontsize=6)
            ax5.text(0.1, 0.85, f'Stars', bbox=dict(boxstyle="round,pad=0.3", fc='w', ec="k", lw=1, alpha=0.8),
                    transform=ax5.transAxes, horizontalalignment='left', fontsize=6)

            # Remove ticks
            ax1.tick_params(axis='both', left=False, top=False, right=False, bottom=False, labelleft=False,
                            labeltop=False, labelright=False, labelbottom=False)
            ax2.tick_params(axis='both', left=False, top=False, right=False, bottom=False, labelleft=False,
                            labeltop=False, labelright=False, labelbottom=False)
            ax3.tick_params(axis='both', left=False, top=False, right=False, bottom=False, labelleft=False,
                            labeltop=False, labelright=False, labelbottom=False)
            ax4.tick_params(axis='both', left=False, top=False, right=False, bottom=False, labelleft=False,
                            labeltop=False, labelright=False, labelbottom=False)
            ax5.tick_params(axis='both', left=False, top=False, right=False, bottom=False, labelleft=False,
                            labeltop=False, labelright=False, labelbottom=False)
            ax6.tick_params(axis='both', left=False, top=False, right=False, bottom=False, labelleft=False,
                            labeltop=False, labelright=False, labelbottom=False)

            custom_lines = [Line2D([0], [0], color='r', lw=3),
                            Line2D([0], [0], color='g', lw=3, linestyle='--')]

            ax6.legend(custom_lines, ['Softening', '$30$ ckpc aperture'], fontsize=6)

            # Set titles
            ax2.set_title('Surrounding particles', fontsize=8)
            ax1.set_title('Galaxy particles', fontsize=8)

            fig.savefig('plots/massdistributions/'+ imgtype + '_reg' + str(reg) + '_snap' + snap +
                        '_gal' + str(id).split('.')[0] + 'p' + str(id).split('.')[1] + '_coords' + key + 'png',
                        bbox_inches='tight', dpi=300)

            plt.close(fig)


# Define comoving softening length in Mpc
csoft = 0.001802390 / 0.677

# Define resolution
res = csoft / 20
print(100 / res, 'pixels in', '100 kpc')

# Define region variables
reg = '0000'
snap = '010_z005p000'
path = '/cosma7/data/dp004/dc-love2/data/G-EAGLE/geagle_' + reg + '/data/'

img_main(path, snap, reg, res, soft=csoft, npart_lim=10**4, imgtype='DMless')
