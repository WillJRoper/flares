#!/cosma/home/dp004/dc-rope1/.conda/envs/flares-env/bin/python
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import astropy.units as u
from scipy.stats import binned_statistic
from matplotlib.colors import LogNorm
import matplotlib.gridspec as gridspec
import eagle_IO.eagle_IO as E
import seaborn as sns
import pickle
import numba as nb
import itertools
matplotlib.use('Agg')

sns.set_style('whitegrid')


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


@nb.njit(nogil=True, parallel=True)
def calc_mass_rad(frac, rs, ms):

    # Sort the radii and masses
    sinds = np.argsort(rs)
    rs = rs[sinds]
    okinds = rs <= 0.03
    rs = rs[okinds]
    ms = ms[okinds]

    # Get the cumalative sum of masses
    m_profile = np.cumsum(ms)

    # Get the total mass and half the total mass
    tot_m = np.sum(ms)
    half_m = tot_m * frac

    # Get the half mass radius particle
    hmr_ind = np.argmin(np.abs(m_profile - half_m))
    hmr = rs[hmr_ind]

    return hmr, tot_m


def plot_median_stat(xs, ys, ax, lab, color, bins=None, ls='-'):

    if bins == None:
        bin = np.logspace(np.log10(xs.min()), np.log10(xs.max()), 20)
    else:
        bin = bins

    # Compute binned statistics
    y_stat, binedges, bin_ind = binned_statistic(xs, ys, statistic='median', bins=bin)

    # Compute bincentres
    bin_wid = binedges[1] - binedges[0]
    bin_cents = binedges[1:] - bin_wid / 2

    okinds = np.logical_and(~np.isnan(bin_cents), ~np.isnan(y_stat))

    ax.plot(bin_cents[okinds], y_stat[okinds], color=color, linestyle=ls, label=lab)


def plot_spread_stat(xs, ys, ax, color, bins=None):

    if bins == None:
        bin = np.logspace(np.log10(xs.min()), np.log10(xs.max()), 20)
    else:
        bin = bins

    # Compute binned statistics
    y_stat_16, binedges, bin_ind = binned_statistic(xs, ys, statistic=lambda y: np.percentile(y, 16), bins=bin)
    y_stat_84, binedges, bin_ind = binned_statistic(xs, ys, statistic=lambda y: np.percentile(y, 84), bins=bin)

    # Compute bincentres
    bin_wid = binedges[1] - binedges[0]
    bin_cents = binedges[1:] - bin_wid / 2

    okinds = np.logical_and(~np.isnan(bin_cents), np.logicak_and(~np.isnan(y_stat_16), ~np.isnan(y_stat_84)))

    ax.fill_between(bin_cents[okinds], y_stat_16[okinds], y_stat_84[okinds], color=color, alpha=0.6)


snaps = ['000_z015p000', '001_z014p000', '002_z013p000', '003_z012p000', '004_z011p000', '005_z010p000',
         '006_z009p000', '007_z008p000', '008_z007p000', '009_z006p000', '010_z005p000', '011_z004p770']

regions = []
for reg in range(0, 1):
    if reg < 10:
        regions.append('0' + str(reg))
    else:
        regions.append(str(reg))

reg_snaps = []
for reg in regions:

    for snap in snaps:

        reg_snaps.append((reg, snap))

axlims_x = []
axlims_y = []

# # Define comoving softening length in kpc
# csoft = 0.001802390/0.677*1e3

hmr0p05_dict = {}
hmr0p25_dict = {}
hmr0p5_dict = {}
hmr0p75_dict = {}
hmr0p90_dict = {}
ms_dict = {}

for reg, snap in reg_snaps:

    path = '/cosma/home/dp004/dc-rope1/FLARES/FLARES-1/G-EAGLE_' + reg + '/data'

    print(reg, snap)

    hmr0p05_dict[snap] = []
    hmr0p25_dict[snap] = []
    hmr0p5_dict[snap] = []
    hmr0p75_dict[snap] = []
    hmr0p90_dict[snap] = []
    ms_dict[snap] = []

    # Get the redshift
    z_str = snap.split('z')[1].split('p')
    z = float(z_str[0] + '.' + z_str[1])

    # Load all necessary arrays
    try:
        subfind_grp_ids = E.read_array('SUBFIND', path, snap, 'Subhalo/GroupNumber', numThreads=8)
        subfind_subgrp_ids = E.read_array('SUBFIND', path, snap, 'Subhalo/SubGroupNumber', numThreads=8)
        gal_cops = E.read_array('SUBFIND', path, snap, 'Subhalo/CentreOfPotential', noH=True,
                                physicalUnits=True, numThreads=8)
        all_gal_ns = E.read_array('SUBFIND', path, snap, 'Subhalo/SubLengthType', numThreads=8)
        all_gal_ms = E.read_array('SUBFIND', path, snap, 'Subhalo/ApertureMeasurements/Mass/030kpc',
                                  numThreads=8) * 10**10
    except ValueError:
        continue

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

    # Load data for luminosities
    try:
        all_poss = E.read_array('PARTDATA', path, snap, 'PartType4/Coordinates', noH=True,
                                physicalUnits=True, numThreads=8)
        part_masses = E.read_array('PARTDATA', path, snap, 'PartType4/Mass', noH=True, physicalUnits=True,
                                  numThreads=8) * 10 ** 10
        grp_ids = E.read_array('PARTDATA', path, snap, 'PartType4/GroupNumber', noH=True,
                               physicalUnits=True, verbose=False, numThreads=8)

        subgrp_ids = E.read_array('PARTDATA', path, snap, 'PartType4/SubGroupNumber', noH=True,
                                  physicalUnits=True, verbose=False, numThreads=8)

        part_ids = E.read_array('PARTDATA', path, snap, 'PartType4/ParticleIDs', noH=True,
                                physicalUnits=True, verbose=False, numThreads=8)
    except ValueError:
        continue

    # A copy of this array is needed for the extraction method
    group_part_ids = np.copy(part_ids)

    print("There are", len(subgrp_ids), "particles")

    # Convert IDs to float(groupNumber.SubGroupNumber) format, i.e. group 1 subgroup 11 = 1.00011
    part_halo_ids = np.zeros(grp_ids.size, dtype=float)
    for (ind, g), sg in zip(enumerate(grp_ids), subgrp_ids):
        part_halo_ids[ind] = float(str(int(g)) + '.%05d' % int(sg))

    print("There are", len(part_halo_ids), "particles")

    print("Got halo IDs")

    okinds = np.isin(part_halo_ids, star_halo_ids)
    part_halo_ids = part_halo_ids[okinds]
    part_ids = part_ids[okinds]
    group_part_ids = group_part_ids[okinds]
    all_poss = all_poss[okinds]
    part_masses = part_masses[okinds]

    parts_in_groups, part_groups = get_part_inds(part_halo_ids, part_ids, group_part_ids, False)

    # Produce a dictionary containing the index of particles in each halo
    halo_part_inds = {}
    for ind, grp in zip(parts_in_groups, part_groups):
        halo_part_inds.setdefault(grp, set()).update({ind})

    # Now the dictionary is fully populated convert values from sets to arrays for indexing
    for key, val in halo_part_inds.items():
        halo_part_inds[key] = np.array(list(val))

    # Get the position of each of these galaxies
    for id, cop in zip(star_halo_ids, gal_cops):
        mask = halo_part_inds[id]
        poss = all_poss[mask, :] - cop
        rs = calc_3drad(poss)
        mass = part_masses[mask]
        r0p05, m = calc_mass_rad(0.05, rs, mass)
        r0p25, _ = calc_mass_rad(0.25, rs, mass)
        r0p50, _ = calc_mass_rad(0.5, rs, mass)
        r0p75, _ = calc_mass_rad(0.75, rs, mass)
        r0p90, _ = calc_mass_rad(0.90, rs, mass)
        hmr0p05_dict[snap].append(r0p05)
        hmr0p25_dict[snap].append(r0p25)
        hmr0p5_dict[snap].append(r0p50)
        hmr0p75_dict[snap].append(r0p75)
        hmr0p90_dict[snap].append(r0p90)
        ms_dict[snap].append(m)

    print('There are', len(star_halo_ids), 'galaxies in ', reg, snap)
    print('There are', len(ms_dict[snap]), 'galaxies in ', snap)


def plot_relation(ms_dict, hmr_dict, savepath):

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

        if z <= 2.8:
            soft = 0.000474390 / 0.6777
        else:
            soft = 0.001802390 / (0.6777 * (1 + z))

        ms = np.array(ms_dict[snap])
        hmr = np.array(hmr_dict[snap])

        okinds =  np.logical_and(ms > 1e8, hmr > 0)
        ms = ms[okinds]
        hmr = hmr[okinds]

        try:
            cbar = ax.hexbin(ms, hmr / soft, gridsize=100, mincnt=1, xscale='log', yscale='log',
                             norm=LogNorm(), linewidths=0.2, cmap='viridis', alpha=0.7)
            plot_median_stat(ms, hmr / soft, ax, lab='REF', color='r')
        except ValueError:
            continue
        except OverflowError:
            continue

        ax.text(0.8, 0.9, f'$z={z}$', bbox=dict(boxstyle="round,pad=0.3", fc='w', ec="k", lw=1, alpha=0.8),
                transform=ax.transAxes, horizontalalignment='right', fontsize=8)

        axlims_x.extend(ax.get_xlim())
        axlims_y.extend(ax.get_ylim())

        # Label axes
        if i == 2:
            ax.set_xlabel(r'$M_{\star}/M_\odot$')
        if j == 0:
            ax.set_ylabel('$R_{1/2}/\epsilon$')

    for ax in [ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8, ax9]:

        ax.set_xlim(np.min(axlims_x), np.max(axlims_x))
        ax.set_ylim(np.min(axlims_y), np.max(axlims_y))
        for spine in ax.spines.values():
            spine.set_edgecolor('k')

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

    # handles, labels = ax1.get_legend_handles_labels()
    # ax1.legend(handles, labels, loc='upper left')

    fig.savefig(savepath, bbox_inches='tight')

    plt.close(fig)

plot_relation(ms_dict, hmr0p05_dict, savepath='plots/0p05MassRadii_all_snaps_REF.png')
plot_relation(ms_dict, hmr0p25_dict, savepath='plots/0p25MassRadii_all_snaps_REF.png')
plot_relation(ms_dict, hmr0p5_dict, savepath='plots/0p5MassRadii_all_snaps_REF.png')
plot_relation(ms_dict, hmr0p75_dict, savepath='plots/0p75MassRadii_all_snaps_REF.png')
plot_relation(ms_dict, hmr0p90_dict, savepath='plots/0p90MassRadii_all_snaps_REF.png')

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

    if z <= 2.8:
        soft = 0.000474390 / 0.6777
    else:
        soft = 0.001802390 / (0.6777 * (1 + z))

    ms = np.array(ms_dict[snap])
    hmr0p05 = np.array(hmr0p05_dict[snap])
    hmr0p25 = np.array(hmr0p25_dict[snap])
    hmr0p5 = np.array(hmr0p5_dict[snap])
    hmr0p75 = np.array(hmr0p75_dict[snap])
    hmr0p90 = np.array(hmr0p90_dict[snap])

    okinds =  np.logical_and(ms > 1e8, hmr0p05 > 0)
    ms = ms[okinds]
    hmr0p05 = hmr0p05[okinds]
    hmr0p25 = hmr0p25[okinds]
    hmr0p5 = hmr0p5[okinds]
    hmr0p75 = hmr0p75[okinds]
    hmr0p90 = hmr0p90[okinds]

    try:
        plot_spread_stat(ms, hmr0p05 / soft, ax, color='darkgrey')
        plot_median_stat(ms, hmr0p05 / soft, ax, lab='$\mathrm{frac}=0.05$', color='darkgrey')
        plot_spread_stat(ms, hmr0p25 / soft, ax, color='limegreen')
        plot_median_stat(ms, hmr0p25 / soft, ax, lab='$\mathrm{frac}=0.25$', color='limegreen')
        plot_spread_stat(ms, hmr0p5 / soft, ax, color='orangered')
        plot_median_stat(ms, hmr0p5 / soft, ax, lab='$\mathrm{frac}=0.50$', color='orangered')
        plot_spread_stat(ms, hmr0p75 / soft, ax, color='deepskyblue')
        plot_median_stat(ms, hmr0p75 / soft, ax, lab='$\mathrm{frac}=0.75$', color='deepskyblue')
        plot_spread_stat(ms, hmr0p90 / soft, ax, color='mediumorchid')
        plot_median_stat(ms, hmr0p90 / soft, ax, lab='$\mathrm{frac}=0.90$', color='mediumorchid')
    except ValueError:
        continue
    except OverflowError:
        continue

    ax.text(0.8, 0.9, f'$z={z}$', bbox=dict(boxstyle="round,pad=0.3", fc='w', ec="k", lw=1, alpha=0.8),
            transform=ax.transAxes, horizontalalignment='right', fontsize=8)

    axlims_x.extend(ax.get_xlim())
    axlims_y.extend(ax.get_ylim())

    # Label axes
    if i == 2:
        ax.set_xlabel(r'$M_{\star}/M_\odot$')
    if j == 0:
        ax.set_ylabel('$R_{\mathrm{frac}}/\epsilon$')

for ax in [ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8, ax9]:

    ax.set_xlim(np.min(axlims_x), np.max(axlims_x))
    ax.set_ylim(np.min(axlims_y), np.max(axlims_y))
    for spine in ax.spines.values():
        spine.set_edgecolor('k')

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

handles, labels = ax1.get_legend_handles_labels()
ax1.legend(handles, labels, loc='upper left')

fig.savefig('plots/MassRadii_all_snaps_REF.png', bbox_inches='tight')

plt.close(fig)
