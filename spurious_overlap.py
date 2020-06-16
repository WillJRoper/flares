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
    overlap = d / extent

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


regions = []
for reg in range(0, 1):

    if reg < 10:
        regions.append('000' + str(reg))
    else:
        regions.append('00' + str(reg))

# snaps = ['003_z012p000', '004_z011p000', '005_z010p000',
#          '006_z009p000', '007_z008p000', '008_z007p000',
#          '009_z006p000', '010_z005p000', '011_z004p770']
snaps = ['010_z005p000', ]
axlims_x = []
axlims_y = []

# Define comoving softening length in kpc
csoft = 0.001802390 / 0.677 * 1e3

p_lim = 1

half_mass_rads_dict = {}
xaxis_dict = {}
ms = {}
for snap in snaps:

    half_mass_rads_dict[snap] = {}
    xaxis_dict[snap] = {}
    ms[snap] = {}

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

        # Build a tree from the COPs
        tree = cKDTree(cops)

        print("There are", len(grp_ids), "halos")

        # Get the spurious halo IDs
        okinds = np.logical_and(subgrp_ids != 1073741824,
                                np.logical_and(gal_app_ms[:, 1] == 0,
                                               np.logical_or(gal_app_ms[:, 0] > 0, gal_app_ms[:, 4] > 0)))
        
        sp_grp_ids = grp_ids[okinds]
        sp_subgrp_ids = subgrp_ids[okinds]
        sp_cops = cops[okinds, :]
        sp_app_ms = gal_app_ms[okinds, :]
        sp_halo_ids = np.zeros(sp_grp_ids.size, dtype=float)
        for (ind, g), sg in zip(enumerate(sp_grp_ids), sp_subgrp_ids):
            sp_halo_ids[ind] = float(str(int(g)) + '.%05d' % int(sg))

        print("Number of spurious with bh", len(sp_halo_ids[sp_app_ms[:, 5] > 0]), "of", len(sp_halo_ids))
        print("Fraction of spurious with bh", len(sp_halo_ids[sp_app_ms[:, 5] > 0]) / len(sp_halo_ids))

        halo_ids = np.zeros(grp_ids.size, dtype=float)
        for (ind, g), sg in zip(enumerate(grp_ids), subgrp_ids):
            halo_ids[ind] = float(str(int(g)) + '.%05d' % int(sg))

        _, parent_inds = tree.query(sp_cops, k=2, n_jobs=8)
        parent_inds = parent_inds[:, 1]
        parents_ms = gal_app_ms[parent_inds, :]
        parent_grp_ids = grp_ids[parent_inds]
        parent_subgrp_ids = subgrp_ids[parent_inds]
        parent_cops = cops[parent_inds]

        gal_poss0 = E.read_array('PARTDATA', path, snap, 'PartType0/Coordinates', noH=True,
                                 physicalUnits=True, verbose=False, numThreads=8)
        gal_poss1 = E.read_array('PARTDATA', path, snap, 'PartType1/Coordinates', noH=True,
                                 physicalUnits=True, verbose=False, numThreads=8)
        gal_poss4 = E.read_array('PARTDATA', path, snap, 'PartType4/Coordinates', noH=True,
                                 physicalUnits=True, verbose=False, numThreads=8)

        poss = np.concatenate([gal_poss0, gal_poss1, gal_poss4])

        gal_vels0 = E.read_array('PARTDATA', path, snap, 'PartType0/Velocity', noH=True,
                                 physicalUnits=True, verbose=False, numThreads=8)
        gal_vels1 = E.read_array('PARTDATA', path, snap, 'PartType1/Velocity', noH=True,
                                 physicalUnits=True, verbose=False, numThreads=8)
        gal_vels4 = E.read_array('PARTDATA', path, snap, 'PartType4/Velocity', noH=True,
                                 physicalUnits=True, verbose=False, numThreads=8)

        vels = np.concatenate([gal_vels0, gal_vels1, gal_vels4])

        grp_id0 = E.read_array('PARTDATA', path, snap, 'PartType0/GroupNumber', noH=True,
                                 physicalUnits=True, verbose=False, numThreads=8)
        grp_id1 = E.read_array('PARTDATA', path, snap, 'PartType1/GroupNumber', noH=True,
                                 physicalUnits=True, verbose=False, numThreads=8)
        grp_id4 = E.read_array('PARTDATA', path, snap, 'PartType4/GroupNumber', noH=True,
                                 physicalUnits=True, verbose=False, numThreads=8)

        grp_ids = np.concatenate([grp_id0, grp_id1, grp_id4])

        subgrp_id0 = E.read_array('PARTDATA', path, snap, 'PartType0/SubGroupNumber', noH=True,
                                 physicalUnits=True, verbose=False, numThreads=8)
        subgrp_id1 = E.read_array('PARTDATA', path, snap, 'PartType1/SubGroupNumber', noH=True,
                                 physicalUnits=True, verbose=False, numThreads=8)
        subgrp_id4 = E.read_array('PARTDATA', path, snap, 'PartType4/SubGroupNumber', noH=True,
                                 physicalUnits=True, verbose=False, numThreads=8)

        subgrp_ids = np.concatenate([subgrp_id0, subgrp_id1, subgrp_id4])

        print("There are", len(subgrp_ids), "particles")

        # Remove particles not associated to a subgroup
        okinds = subgrp_ids != 1073741824
        grp_ids = grp_ids[okinds]
        subgrp_ids = subgrp_ids[okinds]
        poss = poss[okinds, :]
        vels = vels[okinds, :]

        print("There are", len(subgrp_ids), "particles")

        print(vels.shape)
        print(poss.shape)

        overlap, voverlap = np.zeros(sp_halo_ids.size), np.zeros(sp_halo_ids.size)

        for (ind, sp_g), sp_sg, sp_cop, prt_g, prt_sg, prt_cop in zip(enumerate(sp_grp_ids), sp_subgrp_ids, sp_cops,
                                                                      parent_grp_ids, parent_subgrp_ids, parent_cops):
            print(ind, sp_g, sp_sg, prt_g, prt_sg)
            spinds = np.logical_and(grp_ids == sp_g, subgrp_ids == sp_sg)
            prtinds = np.logical_and(grp_ids == prt_g, subgrp_ids == prt_sg)
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

# Set up figure
fig1 = plt.figure()
ax2 = fig1.add_subplot(111)

cbar2 = ax2.hexbin(overlap, voverlap, gridsize=50, mincnt=1, xscale='log', norm=LogNorm(),
                   yscale='log', linewidths=0.2, cmap='viridis', zorder=1)

sep_cut = p_lim - np.logspace(-6, 3, 1000)

# ax2.plot(np.logspace(-6, 3, 1000), sep_cut, color='w', linestyle='-')
ax2.fill_between(np.logspace(-6, 3, 1000), np.zeros(1000), sep_cut, color='c', alpha=0.2, zorder=2)
ax2.fill_between(np.logspace(-6, 3, 1000), np.full(1000, 10), sep_cut, color='r', alpha=0.2, zorder=2)

ax2.set_xlabel(r'$|\langle\mathbf{r}\rangle_1-\langle\mathbf{r}\rangle_2| / (\sigma_{R,1}+\sigma_{R,2})$')
ax2.set_ylabel(r'$|\langle\mathbf{v}\rangle_1-\langle\mathbf{v}\rangle_2|/ (\sigma_{v,1}+\sigma_{v,2})$')

cax2 = fig1.colorbar(cbar2, ax=ax2)
cax2.ax.set_ylabel(r'$N$')

fig1.savefig('plots/spurious_overlap_velvsreal.png', bbox_inches='tight')

plt.close(fig1)






