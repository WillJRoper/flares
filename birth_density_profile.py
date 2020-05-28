#!/cosma/home/dp004/dc-rope1/.conda/envs/flares-env/bin/python
import matplotlib.pyplot as plt
import numpy as np
import pickle
import eagle_IO.eagle_IO as E
import matplotlib.gridspec as gridspec
from unyt import mh, cm, Gyr, g, Msun, Mpc
from matplotlib.colors import LogNorm
from scipy.stats import binned_statistic
import seaborn as sns


sns.set_style('whitegrid')


def plot_meidan_stat(xs, ys, ax, bins=None):

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

    ax.plot(bin_cents[okinds], y_stat[okinds], color='r', linestyle='-')


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

    return halo_part_inds


def get_data(masslim=1e8, load=False):

    regions = []
    for reg in range(0, 40):
        if reg < 10:
            regions.append('0' + str(reg))
        else:
            regions.append(str(reg))

    # Define snapshots
    snaps = ['003_z012p000', '004_z011p000', '005_z010p000',
             '006_z009p000', '007_z008p000', '008_z007p000',
             '009_z006p000', '010_z005p000', '011_z004p770']
    prog_snaps = ['002_z013p000', '003_z012p000', '004_z011p000',
                  '005_z010p000', '006_z009p000', '007_z008p000',
                  '008_z007p000', '009_z006p000', '010_z005p000']

    if load:

        with open('bd_profile.pck', 'rb') as pfile1:
            save_dict = pickle.load(pfile1)
        stellar_rad_dict = save_dict['met']
        stellar_bd_dict = save_dict['bd']

    else:

        stellar_rad_dict = {}
        stellar_bd_dict = {}

        for snap in snaps:

            stellar_rad_dict[snap] = {}
            stellar_bd_dict[snap] = {}

        for reg in regions:

            for snap, prog_snap in zip(snaps, prog_snaps):

                path = '/cosma/home/dp004/dc-rope1/FLARES/FLARES-1/G-EAGLE_' + reg + '/data'

                # Get particle IDs
                halo_part_inds = get_part_ids(path, snap, 4, all_parts=False)

                # Get halo IDs and halo data
                try:
                    subgrp_ids = E.read_array('SUBFIND', path, snap, 'Subhalo/SubGroupNumber', numThreads=8)
                    grp_ids = E.read_array('SUBFIND', path, snap, 'Subhalo/GroupNumber', numThreads=8)
                    gal_ms = E.read_array('SUBFIND', path, snap, 'Subhalo/ApertureMeasurements/Mass/030kpc',
                                          noH=True, physicalUnits=True, numThreads=8)[:, 4] * 10**10
                    gal_bd = E.read_array('PARTDATA', path, snap, 'PartType4/BirthDensity', noH=True,
                                            physicalUnits=True, numThreads=8)
                    gal_coord = E.read_array('PARTDATA', path, snap, 'PartType4/Coordinates', noH=True,
                                             physicalUnits=False, numThreads=8)
                    gal_cop = E.read_array('SUBFIND', path, snap, 'Subhalo/CentreOfPotential', noH=True,
                                           physicalUnits=False, numThreads=8)
                    gal_aborn = E.read_array('PARTDATA', path, snap, 'PartType4/StellarFormationTime', numThreads=8)
                except ValueError:
                    continue
                except OSError:
                    continue
                except KeyError:
                    continue

                z_str = snap.split('z')[1].split('p')
                z = float(z_str[0] + '.' + z_str[1])
                z_str = prog_snap.split('z')[1].split('p')
                prog_z = float(z_str[0] + '.' + z_str[1])

                # Remove particles not associated to a subgroup
                okinds = np.logical_and(subgrp_ids != 1073741824, gal_ms > masslim)
                grp_ids = grp_ids[okinds]
                subgrp_ids = subgrp_ids[okinds]
                gal_cop = gal_cop[okinds]
                halo_ids = np.zeros(grp_ids.size, dtype=float)
                for (ind, g), sg in zip(enumerate(grp_ids), subgrp_ids):
                    halo_ids[ind] = float(str(int(g)) + '.%05d'%int(sg))

                print("There are", len(halo_ids), "galaxies in snapshot", snap, "in region", reg)

                stellar_bd = []
                stellar_rad = []
                for halo, cop in zip(halo_ids, gal_cop):

                    # Add stars from these galaxies
                    part_inds = list(halo_part_inds[halo])
                    parts_bd = gal_bd[part_inds]
                    parts_rs = np.linalg.norm(gal_coord[part_inds, :] - cop, axis=1)
                    parts_aborn = gal_aborn[part_inds]
                    ok_inds = np.logical_and((1 / parts_aborn) - 1 < prog_z, parts_rs < 0.03)
                    stellar_bd.extend(parts_bd[ok_inds])
                    stellar_rad.extend(parts_rs[ok_inds] * 1e3)

                stellar_bd_dict[snap][reg] = stellar_bd
                stellar_rad_dict[snap][reg] = stellar_rad

        with open('bd_profile.pck', 'wb') as pfile1:
            pickle.dump({'bd': stellar_bd_dict, 'met': stellar_rad_dict}, pfile1)

    return stellar_bd_dict, stellar_rad_dict


snaps = ['003_z012p000', '004_z011p000', '005_z010p000',
         '006_z009p000', '007_z008p000', '008_z007p000',
         '009_z006p000', '010_z005p000', '011_z004p770']

# Define comoving softening length in kpc
csoft = 0.001802390 / 0.677 * 1e3

stellar_bd_dict, stellar_rad_dict = get_data(masslim=10**9.5, load=False)

axlims_x = []
axlims_y = []

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

    print("Plotting snapshot:", snap)

    z_str = snap.split('z')[1].split('p')
    z = float(z_str[0] + '.' + z_str[1])

    rads = np.concatenate(list(stellar_rad_dict[snap].values()))
    stellar_bd = (np.concatenate(list(stellar_bd_dict[snap].values()))
                  * 10**10 * Msun / Mpc ** 3 / mh).to(1 / cm ** 3).value
    okinds = np.logical_and(stellar_bd > 0, rads > 0)
    stellar_bd = stellar_bd[okinds]
    rads = rads[okinds]

    if len(stellar_bd) > 0:
        cbar = ax.hexbin(rads, stellar_bd, gridsize=100, mincnt=1, xscale='log', yscale='log',
                         norm=LogNorm(), linewidths=0.2, cmap='magma')
        plot_meidan_stat(rads, stellar_bd, ax)
        axlims_x.extend(ax.get_xlim())
        axlims_y.extend(ax.get_ylim())
    else:
        ax.loglog()
    ax.axvline(csoft, color='k', linestyle='--')

    ax.text(0.1, 0.9, f'$z={z}$', bbox=dict(boxstyle="round,pad=0.3", fc='w', ec="k", lw=1, alpha=0.8),
            transform=ax.transAxes, horizontalalignment='left', fontsize=8)

    # Label axes
    if i == 2:
        ax.set_xlabel("$R$ / [ckpc]")
    if j == 0:
        ax.set_ylabel("Stellar Birth Density [$n_H$ cm$^{-3}$]")

for ax in [ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8, ax9]:
    print(np.min(axlims_x), np.min(axlims_y))
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

fig.savefig('plots/birthdensity_profile.png', bbox_inches='tight')
