import matplotlib.pyplot as plt
import numpy as np
import pickle
import eagle_IO.eagle_IO as E
import matplotlib.gridspec as gridspec
from unyt import mh, cm, Gyr, g, Msun, Mpc
from matplotlib.colors import LogNorm
import numba as nb


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


def get_subgroup_part_inds(sim, snapshot, part_type, all_parts=False, sorted=False):
    ''' A function to efficiently produce a dictionary of particle indexes from EAGLE particle data arrays
        for SUBFIND subgroups.

    :param sim:        Path to the snapshot file [str]
    :param snapshot:   Snapshot identifier [str]
    :param part_type:  The integer representing the particle type
                       (0, 1, 4, 5: gas, dark matter, stars, black hole) [int]
    :param all_parts:  Flag for whether to use all particles (SNAP group)
                       or only particles in halos (PARTDATA group)  [bool]
    :param sorted:     Flag for whether to produce indices in a sorted particle ID array
                       or unsorted (order they are stored in) [bool]
    :return:
    '''

    # Get the particle IDs for this particle type using eagle_IO
    if all_parts:

        # Get all particles in the simulation
        part_ids = E.read_array('SNAP', sim, snapshot, 'PartType' + str(part_type) + '/ParticleIDs',
                                numThreads=8)

        # Get only those particles in a halo
        group_part_ids = E.read_array('PARTDATA', sim, snapshot, 'PartType' + str(part_type) + '/ParticleIDs',
                                      numThreads=8)

    else:

        # Get only those particles in a halo
        part_ids = E.read_array('PARTDATA', sim, snapshot, 'PartType' + str(part_type) + '/ParticleIDs',
                                numThreads=8)

        # A copy of this array is needed for the extraction method
        group_part_ids = np.copy(part_ids)

    # Extract the group ID and subgroup ID each particle is contained within
    grp_ids = E.read_array('PARTDATA', sim, snapshot, 'PartType' + str(part_type) + '/GroupNumber',
                           numThreads=8)
    subgrp_ids = E.read_array('PARTDATA', sim, snapshot, 'PartType' + str(part_type) + '/SubGroupNumber',
                              numThreads=8)

    # Remove particles not associated to a subgroup (subgroupnumber == 2**30 == 1073741824)
    okinds = subgrp_ids != 1073741824
    group_part_ids = group_part_ids[okinds]
    grp_ids = grp_ids[okinds]
    subgrp_ids = subgrp_ids[okinds]

    # Ensure no subgroup ID exceeds 99999
    assert subgrp_ids.max() < 99999, "Found too many subgroups, need to increase subgroup format string above %05d"

    # Convert IDs to float(groupNumber.SubGroupNumber) format, i.e. group 1 subgroup 11 = 1.00011
    halo_ids = np.zeros(grp_ids.size, dtype=float)
    for (ind, g), sg in zip(enumerate(grp_ids), subgrp_ids):
        halo_ids[ind] = float(str(int(g)) + '.%05d' % int(sg))

    parts_in_groups, part_groups = get_part_inds(halo_ids, part_ids, group_part_ids, sorted)

    # Produce a dictionary containing the index of particles in each halo
    halo_part_inds = {}
    for ind, grp in zip(parts_in_groups, part_groups):
        halo_part_inds.setdefault(grp, set()).update({ind})

    # Now the dictionary is fully populated convert values from sets to arrays for indexing
    for key, val in halo_part_inds.items():
        halo_part_inds[key] = np.array(list(val))

    return halo_part_inds


@nb.njit(nogil=True, parallel=True)
def rms_radii(pos, cent):

    # Get the seperation between particles and halo centre
    sep = pos - cent

    # Get radii squared
    rad_sepxy = sep[:, 0]**2 + sep[:, 1]**2
    rad_sepyz = sep[:, 1] ** 2 + sep[:, 2] ** 2
    rad_sepxz = sep[:, 0] ** 2 + sep[:, 2] ** 2

    # Get rms radii
    rms_rad = np.array([np.sqrt(np.mean(rad_sepxy)), np.sqrt(np.mean(rad_sepyz)), np.sqrt(np.mean(rad_sepxz))])

    return rms_rad


def get_ratio(pos, cent):

    # Get rms radii
    rms_rad = rms_radii(pos, cent)

    # Calculate ratio
    ratio = rms_rad.min() / rms_rad.max()

    return ratio


def get_data(snap, masslim=1e8):

    path = '/cosma/home/dp004/dc-rope1/FLARES/FLARES-1/G-EAGLE_' + reg + '/data'

    # Get particle IDs
    halo_part_inds = get_subgroup_part_inds(path, snap, part_type=4, all_parts=False)

    # Get halo IDs and halo data
    try:
        subgrp_ids = E.read_array('SUBFIND', path, snap, 'Subhalo/SubGroupNumber', numThreads=8)
        grp_ids = E.read_array('SUBFIND', path, snap, 'Subhalo/GroupNumber', numThreads=8)
        gal_ms = E.read_array('SUBFIND', path, snap, 'Subhalo/ApertureMeasurements/Mass/030kpc',
                              noH=False, physicalUnits=False, numThreads=8)[:, 4] * 10**10
        gal_cop = E.read_array('SUBFIND', path, snap, 'Subhalo/CentreOfPotential', numThreads=8)
        gal_hmr = E.read_array('SUBFIND', path, snap, 'Subhalo/HalfMassRad', numThreads=8)[:, 4]
        gal_coord = E.read_array('PARTDATA', path, snap, 'PartType4/Coordinates', noH=True,
                                 physicalUnits=True, numThreads=8)
    except ValueError:
        return [], [], []
    except OSError:
        return [], [], []
    except KeyError:
        return [], [], []

    # Remove particles not associated to a subgroup
    okinds = np.logical_and(subgrp_ids != 1073741824, gal_ms > masslim)
    grp_ids = grp_ids[okinds]
    subgrp_ids = subgrp_ids[okinds]
    gal_ms = gal_ms[okinds]
    gal_cop = gal_cop[okinds]
    gal_hmr = gal_hmr[okinds]
    halo_ids = np.zeros(grp_ids.size, dtype=float)
    for (ind, g), sg in zip(enumerate(grp_ids), subgrp_ids):
        halo_ids[ind] = float(str(int(g)) + '.%05d'%int(sg))

    print("There are", len(halo_ids), "galaxies in snapshot", snap)

    ratios = []
    masses = []
    hmrs = []
    for halo, cop, m, hmr in zip(halo_ids, gal_cop, gal_ms, gal_hmr):

        # Add stars from these galaxies
        part_inds = list(halo_part_inds[halo])
        ratios.append(get_ratio(gal_coord[part_inds, :], cop))
        masses.append(m)
        hmrs.append(hmr)

    return ratios, masses, hmrs



regions = []
for reg in range(0, 40):

    if reg < 10:
        regions.append('0' + str(reg))
    else:
        regions.append(str(reg))

snaps = ['003_z012p000', '004_z011p000', '005_z010p000',
         '006_z009p000', '007_z008p000', '008_z007p000',
         '009_z006p000', '010_z005p000', '011_z004p770']
axlims_x = []
axlims_y = []

# Define comoving softening length in kpc
csoft = 0.001802390 / 0.677
ratios_dict = {}
half_mass_rads_dict = {}
masses_dict = {}
for snap in snaps:
    ratios_dict[snap] = {}
    half_mass_rads_dict[snap] = {}
    masses_dict[snap] = {}

for reg in regions:

    for snap in snaps:

        print(reg, snap)
        try:
            ratios_dict[snap][reg], masses_dict[snap][reg], half_mass_rads_dict[snap][reg] = get_data(snap,
                                                                                                      masslim=1e8)
        except FileNotFoundError:
            continue


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

    rat = np.concatenate(list(ratios_dict[snap].values()))
    half_mass_rads_plt = np.concatenate(list(half_mass_rads_dict[snap].values()))

    rat_plt = rat[half_mass_rads_plt > 0]
    half_mass_rads_plt = half_mass_rads_plt[half_mass_rads_plt > 0]
    half_mass_rads_plt = half_mass_rads_plt[rat_plt > 0]
    rat_plt = rat_plt[rat_plt > 0]

    if len(rat_plt) > 0:
        cbar = ax.hexbin(half_mass_rads_plt / (csoft / (1 + z)), rat_plt, gridsize=100, mincnt=1, xscale='log',
                         norm=LogNorm(), linewidths=0.2, cmap='viridis')

    ax.text(0.1, 0.9, f'$z={z}$', bbox=dict(boxstyle="round,pad=0.3", fc='w', ec="k", lw=1, alpha=0.8),
            transform=ax.transAxes, horizontalalignment='right', fontsize=8)

    axlims_x.extend(ax.get_xlim())
    axlims_y.extend(ax.get_ylim())

    # Label axes
    if i == 2:
        ax.set_xlabel('$R_{1/2,\mathrm{\star}}/\epsilon$')
    if j == 0:
        ax.set_ylabel('$R_{rms, min}/ R_{rms, max}$')

for ax in [ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8, ax9]:
    ax.set_xlim(np.min(axlims_x), np.max(axlims_x))
    ax.set_ylim(np.min(axlims_y), np.max(axlims_y))

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

fig.savefig('plots/Axis_ratio_vs_HMRredshift.png',
            bbox_inches='tight')

plt.close(fig)

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

    rat = np.concatenate(list(ratios_dict[snap].values()))
    half_mass_rads_plt = np.concatenate(list(masses_dict[snap].values()))

    rat_plt = rat[half_mass_rads_plt > 1e8]
    half_mass_rads_plt = half_mass_rads_plt[half_mass_rads_plt > 1e8]
    half_mass_rads_plt = half_mass_rads_plt[rat_plt > 0]
    rat_plt = rat_plt[rat_plt > 0]

    if len(rat_plt) > 0:
        cbar = ax.hexbin(half_mass_rads_plt, rat_plt, gridsize=100, mincnt=1, xscale='log',
                         norm=LogNorm(), linewidths=0.2, cmap='viridis')

    ax.text(0.1, 0.9, f'$z={z}$', bbox=dict(boxstyle="round,pad=0.3", fc='w', ec="k", lw=1, alpha=0.8),
            transform=ax.transAxes, horizontalalignment='right', fontsize=8)

    axlims_x.extend(ax.get_xlim())
    axlims_y.extend(ax.get_ylim())

    # Label axes
    if i == 2:
        ax.set_xlabel('$M_{\star}/M_\odot$')
    if j == 0:
        ax.set_ylabel('$R_{rms, min}/ R_{rms, max}$')

for ax in [ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8, ax9]:
    ax.set_xlim(np.min(axlims_x), np.max(axlims_x))
    ax.set_ylim(np.min(axlims_y), np.max(axlims_y))

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

fig.savefig('plots/Axis_ratio_vs_mass_redshift.png',
            bbox_inches='tight')

plt.close(fig)
