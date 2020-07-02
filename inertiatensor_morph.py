#!/cosma/home/dp004/dc-rope1/.conda/envs/flares-env/bin/python
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LogNorm
import matplotlib.gridspec as gridspec
import numba as nb
import eagle_IO.eagle_IO as E


sns.set_context("paper")
sns.set_style('whitegrid')


@nb.njit(nogil=True, parallel=True)
def shape_tensor(masses, pos):

    s_tensor = np.zeros((3, 3))

    for i in range(3):
        for j in range(3):
            for k in range(len(masses)):
                s_tensor[i, j] += masses[k] * pos[k, i] * pos[k, j]

    return s_tensor / np.sum(masses)


@nb.njit(nogil=True, parallel=True)
def get_diag_stensor(masses, pos):

    s_tensor = shape_tensor(masses, pos)

    return np.linalg.eigvals(s_tensor)


def get_part_inds(halo_ids, part_ids, group_part_ids, sorted):
    """ A function to find the indexes and halo IDs associated to particles/a particle producing an array for each

    :param halo_ids:
    :param part_ids:
    :param group_part_ids:
    :return:
    """

    # Sort particle IDs if required and store an unsorted version in an array
    if sorted:
        part_ids = np.sort(part_ids)
    unsort_part_ids = np.copy(part_ids)

    # Get the indices that would sort the array (if the array is sorted this is just a range form 0-Npart)
    if sorted:
        sinds = np.arange(part_ids.size)
    else:
        sinds = np.argsort(part_ids)
        part_ids = part_ids[sinds]

    # Get the index of particles in the snapshot array from the in particles in a group array
    sorted_index = np.searchsorted(part_ids, group_part_ids)  # find the indices that would sort the array
    yindex = np.take(sinds, sorted_index, mode="raise")  # take the indices at the indices found above
    mask = unsort_part_ids[yindex] != group_part_ids  # define the mask based on these particles
    result = np.ma.array(yindex, mask=mask)  # create a mask array

    # Apply the mask to the id arrays to get halo ids and the particle indices
    part_groups = halo_ids[np.logical_not(result.mask)]  # halo ids
    parts_in_groups = result.data[np.logical_not(result.mask)]  # particle indices

    return parts_in_groups, part_groups


@nb.njit(nogil=True, parallel=True)
def calc_3drad(poss):

    # Get galaxy particle indices
    rs = np.sqrt(poss[:, 0]**2 + poss[:, 1]**2 + poss[:, 2]**2)

    return rs


def main(snaps):

    regions = []
    for reg in range(0, 1):
        if reg < 10:
            regions.append('0' + str(reg))
        else:
            regions.append(str(reg))

    b_a_dict = {}
    c_a_dict = {}

    for snap in snaps:
        b_a_dict[snap] = []
        c_a_dict[snap] = []

    for reg in regions:

        path = '/cosma/home/dp004/dc-rope1/FLARES/FLARES-1/G-EAGLE_' + reg + '/data'

        for snap in snaps:

            # Get the redshift
            z_str = snap.split('z')[1].split('p')
            z = float(z_str[0] + '.' + z_str[1])

            print(reg, z)

            # Load all necessary arrays
            subfind_grp_ids = E.read_array('SUBFIND', path, snap, 'Subhalo/GroupNumber', numThreads=8)
            subfind_subgrp_ids = E.read_array('SUBFIND', path, snap, 'Subhalo/SubGroupNumber', numThreads=8)
            gal_cops = E.read_array('SUBFIND', path, snap, 'Subhalo/CentreOfPotential', noH=True,
                                    physicalUnits=True, numThreads=8)
            all_gal_ns = E.read_array('SUBFIND', path, snap, 'Subhalo/SubLengthType', numThreads=8)
            all_gal_ms = E.read_array('SUBFIND', path, snap, 'Subhalo/ApertureMeasurements/Mass/030kpc',
                                      numThreads=8) * 10**10

            # Remove particles not in a subgroup
            okinds = np.logical_and(subfind_subgrp_ids != 1073741824,
                                    np.logical_and((all_gal_ns[:, 4] + all_gal_ns[:, 0]) >= 100,
                                                   np.logical_and(all_gal_ms[:, 4] >= 1e8, all_gal_ms[:, 0] > 0)
                                                   )
                                    )
            subfind_grp_ids = subfind_grp_ids[okinds]
            subfind_subgrp_ids = subfind_subgrp_ids[okinds]
            gal_cops = gal_cops[okinds]

            # Convert IDs to float(groupNumber.SubGroupNumber) format, i.e. group 1 subgroup 11 = 1.00011
            halo_ids = np.zeros(subfind_grp_ids.size, dtype=float)
            for (ind, g), sg in zip(enumerate(subfind_grp_ids), subfind_subgrp_ids):
                halo_ids[ind] = float(str(int(g)) + '.%05d' % int(sg))

            star_halo_ids = np.copy(halo_ids)

            try:
                # Load data for luminosities
                all_poss = E.read_array('PARTDATA', path, snap, 'PartType4/Coordinates', noH=True,
                                        physicalUnits=True, numThreads=8)
                masses = E.read_array('PARTDATA', path, snap, 'PartType4/Mass', noH=True, physicalUnits=True,
                                      numThreads=8) * 10 ** 10
                grp_ids = E.read_array('PARTDATA', path, snap, 'PartType4/GroupNumber', noH=True,
                                       physicalUnits=True, verbose=False, numThreads=8)

                subgrp_ids = E.read_array('PARTDATA', path, snap, 'PartType4/SubGroupNumber', noH=True,
                                          physicalUnits=True, verbose=False, numThreads=8)

                part_ids = E.read_array('PARTDATA', path, snap, 'PartType4/ParticleIDs', noH=True,
                                        physicalUnits=True, verbose=False, numThreads=8)
            except OSError:
                return
            except KeyError:
                return

            # A copy of this array is needed for the extraction method
            group_part_ids = np.copy(part_ids)

            print("There are", len(subgrp_ids), "particles")

            # Convert IDs to float(groupNumber.SubGroupNumber) format, i.e. group 1 subgroup 11 = 1.00011
            part_halo_ids = np.zeros(grp_ids.size, dtype=float)
            for (ind, g), sg in zip(enumerate(grp_ids), subgrp_ids):
                part_halo_ids[ind] = float(str(int(g)) + '.%05d' % int(sg))

            okinds = np.isin(part_halo_ids, star_halo_ids)
            part_halo_ids = part_halo_ids[okinds]
            part_ids = part_ids[okinds]
            group_part_ids = group_part_ids[okinds]
            all_poss = all_poss[okinds]
            masses = masses[okinds]

            print("There are", len(part_halo_ids), "particles")

            print("Got halo IDs")

            parts_in_groups, part_groups = get_part_inds(part_halo_ids, part_ids, group_part_ids, False)

            # Produce a dictionary containing the index of particles in each halo
            halo_part_inds = {}
            for ind, grp in zip(parts_in_groups, part_groups):
                halo_part_inds.setdefault(grp, set()).update({ind})

            # Now the dictionary is fully populated convert values from sets to arrays for indexing
            for key, val in halo_part_inds.items():
                halo_part_inds[key] = np.array(list(val))

            # Get the position of each of these galaxies
            gal_ms = {}
            all_gal_poss = {}
            means = {}
            for id, cop in zip(star_halo_ids, gal_cops):
                mask = halo_part_inds[id]
                all_gal_poss[id] = all_poss[mask, :]
                gal_ms[id] = masses[mask]
                means[id] = cop

            for halo in star_halo_ids:

                # Compute the eigenvalues of vectors
                eigs = get_diag_stensor(gal_ms[halo], all_gal_poss[halo] - means[halo])
                b_a_dict[snap].append(np.sqrt(eigs[1] / eigs[0]))
                c_a_dict[snap].append(np.sqrt(eigs[2] / eigs[0]))

    return b_a_dict, c_a_dict


# Define snapshots
snaps = ['003_z012p000', '004_z011p000', '005_z010p000',
         '006_z009p000', '007_z008p000', '008_z007p000',
         '009_z006p000', '010_z005p000', '011_z004p770']

b_a_dict, c_a_dict = main(snaps)

# Set up plot
fig = plt.figure(figsize=(18, 10))
gs = gridspec.GridSpec(3, 6)
gs.update(wspace=0.0, hspace=0.0)
ax1 = fig.add_subplot(gs[0, 0])
ax2 = fig.add_subplot(gs[0, 1])
ax3 = fig.add_subplot(gs[0, 2])
ax4 = fig.add_subplot(gs[1, 0])
ax5 = fig.add_subplot(gs[1, 1])
ax6 = fig.add_subplot(gs[1, 2])
ax7 = fig.add_subplot(gs[2, 0])
ax8 = fig.add_subplot(gs[2, 1])
ax9 = fig.add_subplot(gs[2, 2])

for ax, snap, (i, j) in zip([ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8, ax9], snaps,
                            [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2), (2, 0), (2, 1), (2, 2)]):

    z_str = snap.split('z')[1].split('p')
    z = float(z_str[0] + '.' + z_str[1])

    cbar = ax.hexbin(b_a_dict[snap], c_a_dict[snap], gridsize=100, mincnt=1, xscale='log', yscale='log',
                     norm=LogNorm(),
                     linewidths=0.2, cmap='viridis')

    ax.text(0.8, 0.9, f'$z={z}$', bbox=dict(boxstyle="round,pad=0.3", fc='w', ec="k", lw=1, alpha=0.8),
            transform=ax.transAxes, horizontalalignment='right', fontsize=8)

    # Label axes
    if i == 2:
        ax.set_xlabel(r'$b/a$')
    if j == 0:
        ax.set_ylabel('$c/a$')

for ax in [ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8, ax9]:
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

# Remove axis labels
ax1.tick_params(axis='x', top=False, bottom=False, labeltop=False, labelbottom=False)
ax2.tick_params(axis='both', left=False, top=False, right=False, bottom=False, labelleft=False, labeltop=False,
                labelright=False, labelbottom=False)
ax3.tick_params(axis='both', left=False, top=False, right=False, bottom=False, labelleft=False, labeltop=False,
                labelright=False, labelbottom=False)
ax4.tick_params(axis='x', top=False, bottom=False, labeltop=False, labelbottom=False)
ax5.tick_params(axis='both', left=False, top=False, right=False, bottom=False, labelleft=False, labeltop=False,
                labelright=False, labelbottom=False)
ax6.tick_params(axis='both', left=False, top=False, right=False, bottom=False, labelleft=False, labeltop=False,
                labelright=False, labelbottom=False)
ax8.tick_params(axis='y', left=False, right=False, labelleft=False, labelright=False)
ax9.tick_params(axis='y', left=False, right=False, labelleft=False, labelright=False)

fig.savefig('plots/morph_inertiatensor.png',
            bbox_inches='tight')

plt.close(fig)
