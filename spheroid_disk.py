import matplotlib.pyplot as plt
import numpy as np
import pickle
import eagle_IO.eagle_IO as E
import matplotlib.gridspec as gridspec
from unyt import mh, cm, Gyr, g, Msun, Mpc
from matplotlib.colors import LogNorm
import numba as nb
from flares import flares


fl = flares

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
    halo_part_inds = fl.get_subgroup_part_inds(path, snap, part_type=4, all_parts=False)

    # Get halo IDs and halo data
    try:
        subgrp_ids = E.read_array('SUBFIND', path, snap, 'Subhalo/SubGroupNumber', numThreads=8)
        grp_ids = E.read_array('SUBFIND', path, snap, 'Subhalo/GroupNumber', numThreads=8)
        gal_ms = E.read_array('SUBFIND', path, snap, 'Subhalo/ApertureMeasurements/Mass/030kpc',
                              noH=False, physicalUnits=False, numThreads=8)[:, 4] * 10**10
        gal_cop = E.read_array('SUBFIND', path, snap, 'Subhalo/CentreOfPotential', numThreads=8)
        gal_hmr = E.read_array('SUBFIND', path, snap, 'Subhalo/HalfMassRad', numThreads=8)
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

    ratios = []
    masses= []
    hmrs = []
    for halo, cop, m, hmr in zip(halo_ids, gal_cop, gal_ms, gal_hmr):

        # Add stars from these galaxies
        part_inds = list(halo_part_inds[halo])
        ratios.append(get_ratio(gal_coord[part_inds, :], cop))
        masses.append(m)
        hmrs.append(hmr)

    return ratios, masses, hmrs



regions = []
for reg in range(0, 2):

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
                         yscale='log', norm=LogNorm(), linewidths=0.2, cmap='viridis')

    ax.text(0.8, 0.1, f'$z={z}$', bbox=dict(boxstyle="round,pad=0.3", fc='w', ec="k", lw=1, alpha=0.8),
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
                         yscale='log', norm=LogNorm(), linewidths=0.2, cmap='viridis')

    ax.text(0.8, 0.1, f'$z={z}$', bbox=dict(boxstyle="round,pad=0.3", fc='w', ec="k", lw=1, alpha=0.8),
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
