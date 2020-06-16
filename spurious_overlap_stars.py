#!/cosma/home/dp004/dc-rope1/.conda/envs/flares-env/bin/python
import numpy as np
import eagle_IO.eagle_IO as E
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree
import seaborn as sns
from matplotlib.colors import LogNorm


sns.set_context("paper")
sns.set_style('whitegrid')


def get_phase_sep(cent1, cent2, vcent1, vcent2, r1, r2, vr1, vr2):

    # Compute the separation, sum of radii and overlap in real space
    sep = cent1 - cent2
    d = np.sqrt(sep[0] ** 2 + sep[1] ** 2 + sep[2] ** 2)
    extent = r1 + r2
    overlap = d / 0.03

    # Compute the separation, sum of radii and overlap in velocity space
    vsep = vcent1 - vcent2
    vd = np.sqrt(vsep[0] ** 2 + vsep[1] ** 2 + vsep[2] ** 2)
    vextent = vr1 + vr2
    voverlap = vd / vextent

    return overlap, voverlap


def rms_rad(pos, cent):

    # Get the seperation between particles and halo centre
    sep = pos - cent

    # Get radii squared
    rad_sep = sep[:, 0]**2 + sep[:, 1]**2 + sep[:, 2]**2

    return np.sqrt(5 / 3 * 1 / rad_sep.size * np.sum(rad_sep))


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


regions = []
for reg in range(0, 40):

    if reg < 10:
        regions.append('000' + str(reg))
    else:
        regions.append('00' + str(reg))

snaps = ['003_z012p000', '004_z011p000', '005_z010p000',
         '006_z009p000', '007_z008p000', '008_z007p000',
         '009_z006p000', '010_z005p000', '011_z004p770']

axlims_x = []
axlims_y = []

# Define comoving softening length in kpc
csoft = 0.001802390 / 0.677 * 1e3

p_lim = 2
s_lim = 1
v_lim = 2

overlaps, voverlaps = [], []

for reg in regions:

    for snap in snaps:

        print(reg, snap)

        path = '/cosma7/data/dp004/dc-love2/data/G-EAGLE/geagle_' + reg + '/data/'

        cops = E.read_array('SUBFIND', path, snap, 'Subhalo/CentreOfPotential', noH=True, physicalUnits=True,
                           verbose=False, numThreads=8)
        gal_app_ms = E.read_array('SUBFIND', path, snap, 'Subhalo/ApertureMeasurements/Mass/030kpc', noH=True,
                                  verbose=False, numThreads=8) * 10**10
        subgrp_ids = E.read_array('SUBFIND', path, snap, 'Subhalo/SubGroupNumber', verbose=False, numThreads=8)
        grp_ids = E.read_array('SUBFIND', path, snap, 'Subhalo/GroupNumber', verbose=False, numThreads=8)

        # Get the spurious halo IDs
        okinds = np.logical_and(subgrp_ids != 1073741824, gal_app_ms[:, 4] > 0)
        cops = cops[okinds]
        grp_ids = grp_ids[okinds]
        subgrp_ids = subgrp_ids[okinds]
        gal_app_ms = gal_app_ms[okinds]

        # Build a tree from the COPs
        tree = cKDTree(cops)

        print("There are", len(grp_ids), "halos")

        # Get the spurious halo IDs
        okinds = np.logical_and(subgrp_ids != 1073741824,
                                np.logical_and(gal_app_ms[:, 1] == 0, gal_app_ms[:, 4] > 0))

        sp_grp_ids = grp_ids[okinds]
        sp_subgrp_ids = subgrp_ids[okinds]
        sp_cops = cops[okinds, :]
        sp_app_ms = gal_app_ms[okinds, :]
        sp_halo_ids = np.zeros(sp_grp_ids.size, dtype=float)
        for (ind, g), sg in zip(enumerate(sp_grp_ids), sp_subgrp_ids):
            sp_halo_ids[ind] = float(str(int(g)) + '.%05d' % int(sg))

        print("Number of spurious with bh", len(sp_halo_ids[sp_app_ms[:, 5] > 0]), "of", len(sp_halo_ids))

        halo_ids = np.zeros(grp_ids.size, dtype=float)
        for (ind, g), sg in zip(enumerate(grp_ids), subgrp_ids):
            halo_ids[ind] = float(str(int(g)) + '.%05d' % int(sg))

        dd, parent_inds = tree.query(sp_cops, k=2, n_jobs=8)
        dists = dd[:, 1]
        parent_inds = parent_inds[:, 1]
        parents_ms = gal_app_ms[parent_inds, :]
        parent_grp_ids = grp_ids[parent_inds]
        parent_subgrp_ids = subgrp_ids[parent_inds]
        parent_cops = cops[parent_inds]

        print(len(dists), len(dists[dists < 30 / 1000]))

        poss = E.read_array('PARTDATA', path, snap, 'PartType4/Coordinates', noH=True,
                                 physicalUnits=True, verbose=False, numThreads=8)

        vels = E.read_array('PARTDATA', path, snap, 'PartType4/Velocity', noH=True,
                                 physicalUnits=True, verbose=False, numThreads=8)

        grp_ids = E.read_array('PARTDATA', path, snap, 'PartType4/GroupNumber', noH=True,
                                 physicalUnits=True, verbose=False, numThreads=8)

        subgrp_ids = E.read_array('PARTDATA', path, snap, 'PartType4/SubGroupNumber', noH=True,
                                 physicalUnits=True, verbose=False, numThreads=8)

        part_ids = E.read_array('PARTDATA', path, snap, 'PartType4/ParticleIDs', noH=True,
                                 physicalUnits=True, verbose=False, numThreads=8)

        # A copy of this array is needed for the extraction method
        group_part_ids = np.copy(part_ids)

        print("There are", len(subgrp_ids), "particles")

        # Remove particles not associated to a subgroup
        okinds = subgrp_ids != 1073741824
        grp_ids = grp_ids[okinds]
        subgrp_ids = subgrp_ids[okinds]
        part_ids = part_ids[okinds]
        group_part_ids = group_part_ids[okinds]
        poss = poss[okinds, :]
        vels = vels[okinds, :]

        print("There are", len(subgrp_ids), "particles")

        print(vels.shape)
        print(poss.shape)

        # Convert IDs to float(groupNumber.SubGroupNumber) format, i.e. group 1 subgroup 11 = 1.00011
        halo_ids = np.zeros(grp_ids.size, dtype=float)
        for (ind, g), sg in zip(enumerate(grp_ids), subgrp_ids):
            halo_ids[ind] = float(str(int(g)) + '.%05d' % int(sg))

        print("Got halo IDs")

        parts_in_groups, part_groups = get_part_inds(halo_ids, part_ids, group_part_ids, False)

        # Produce a dictionary containing the index of particles in each halo
        halo_part_inds = {}
        for ind, grp in zip(parts_in_groups, part_groups):
            halo_part_inds.setdefault(grp, set()).update({ind})

        # Now the dictionary is fully populated convert values from sets to arrays for indexing
        for key, val in halo_part_inds.items():
            halo_part_inds[key] = np.array(list(val))

        print("Got index dictionary")

        overlap, voverlap = np.zeros(sp_halo_ids.size), np.zeros(sp_halo_ids.size)

        for (ind, sp_g), sp_sg, sp_cop, prt_g, prt_sg, prt_cop in zip(enumerate(sp_grp_ids), sp_subgrp_ids, sp_cops,
                                                                      parent_grp_ids, parent_subgrp_ids, parent_cops):
            sp_id = float(str(int(sp_g)) + '.%05d' % int(sp_sg))
            prt_id = float(str(int(prt_g)) + '.%05d' % int(prt_sg))
            spinds = halo_part_inds[sp_id]
            prtinds = halo_part_inds[prt_id]
            # spinds = np.logical_and(subgrp_ids == sp_sg, grp_ids == sp_g)
            # prtinds = np.logical_and(subgrp_ids == prt_sg, grp_ids == prt_g)
            sp_vs = vels[spinds]
            prt_vs = vels[prtinds]
            sp_ps = poss[spinds]
            prt_ps = poss[prtinds]

            # Compute the overlaps
            sp_vcent = np.mean(sp_vs, axis=0)
            prt_vcent = np.mean(prt_vs, axis=0)
            sp_r = rms_rad(sp_ps, sp_cop)
            prt_r = rms_rad(prt_ps, prt_cop)
            sp_vr = rms_rad(sp_vs, sp_vcent)
            prt_vr = rms_rad(prt_vs, prt_vcent)

            overlap[ind], voverlap[ind] = get_phase_sep(prt_cop, sp_cop, prt_vcent, sp_vcent,
                                                        prt_r, sp_r, prt_vr, sp_vr)
        overlaps.extend(overlap)
        voverlaps.extend(voverlap)

overlap = np.array(overlaps)
voverlap = np.array(voverlaps)
print(overlap.size, voverlap.size)
okinds = np.logical_and(overlap != 0, voverlap != 0)
overlap = overlap[okinds]
voverlap = voverlap[okinds]
print(overlap.size, voverlap.size)

# Set up figure
fig1 = plt.figure()
ax2 = fig1.add_subplot(111)

# ax2.plot(np.full(1000, s_lim), np.logspace(-2, 2.8, 1000), color='k', linestyle='--')
# ax2.plot(np.logspace(-4, 2, 1000), np.full(1000, v_lim), color='k', linestyle='-.')

cbar2 = ax2.hexbin(overlap, voverlap, gridsize=50, mincnt=1, xscale='log', norm=LogNorm(),
                   yscale='log', linewidths=0.2, cmap='viridis', zorder=1)

ax2.set_xlabel(r'$|\langle\mathbf{r}\rangle_1-\langle\mathbf{r}\rangle_2| / (30 [\mathrm{kpc}])$')
ax2.set_ylabel(r'$|\langle\mathbf{v}\rangle_1-\langle\mathbf{v}\rangle_2|/ (\sigma_{v,1}+\sigma_{v,2})$')

cax2 = fig1.colorbar(cbar2, ax=ax2)
cax2.ax.set_ylabel(r'$N$')

# ax2.set_xlim(10**-4, 10**2)
# ax2.set_ylim(10**-2, 10**2.8)

fig1.savefig('plots/spurious_overlap_velvsreal_starsonly.png', bbox_inches='tight')

plt.close(fig1)






