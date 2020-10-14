#!/cosma/home/dp004/dc-rope1/.conda/envs/flares-env/bin/python
"""
Creates the plot of metallicity against birth density, with
the background coloured by f_th.
"""

import matplotlib.pyplot as plt
import numpy as np
import pickle
from matplotlib.colors import LogNorm
import matplotlib.gridspec as gridspec
import eagle_IO.eagle_IO as E
from scipy.stats import binned_statistic
from astropy.cosmology import Planck13 as cosmo
import astropy.units as u
import seaborn as sns

sns.set_style("whitegrid")


def calc_ages(z, a_born):

    # Convert scale factor into redshift
    z_borns = 1 / a_born - 1

    # Convert to time in Gyrs
    t = cosmo.age(z).value
    t_born = np.zeros(len(a_born))
    for ind, z_born in enumerate(z_borns):
        t_born[ind] = cosmo.age(z_born).value

    # Calculate the VR
    ages = (t - t_born)

    return ages


def plot_meidan_stat(xs, ys, ax, lab, color, bins=None, ls='-'):

    if bins == None:
        bin = np.logspace(np.log10(xs.min()), np.log10(xs.max()), 20)
    else:
        zs = np.float64(xs)

        uniz = np.unique(zs)
        bin_wids = uniz[1:] - uniz[:-1]
        low_bins = uniz[:-1] - (bin_wids / 2)
        high_bins = uniz[:-1] + (bin_wids / 2)
        low_bins = list(low_bins)
        high_bins = list(high_bins)
        low_bins.append(high_bins[-1])
        high_bins.append(uniz[-1] + 1)
        low_bins = np.array(low_bins)
        high_bins = np.array(high_bins)

        bin = np.zeros(uniz.size + 1)
        bin[:-1] = low_bins
        bin[1:] = high_bins

    # Compute binned statistics
    y_stat, binedges, bin_ind = binned_statistic(xs, ys, statistic='median',
                                                 bins=bin)

    # Compute bincentres
    bin_wid = binedges[1] - binedges[0]
    bin_cents = binedges[1:] - bin_wid / 2

    okinds = np.logical_and(~np.isnan(bin_cents), ~np.isnan(y_stat))

    ax.plot(bin_cents[okinds], y_stat[okinds], color=color, linestyle=ls,
            label=lab)


def plot_spread_stat(zs, ys, ax, color):
    zs = np.float64(zs)

    uniz = np.unique(zs)
    bin_wids = uniz[1:] - uniz[:-1]
    low_bins = uniz[:-1] - (bin_wids / 2)
    high_bins = uniz[:-1] + (bin_wids / 2)
    low_bins = list(low_bins)
    high_bins = list(high_bins)
    low_bins.append(high_bins[-1])
    high_bins.append(uniz[-1] + 1)
    low_bins = np.array(low_bins)
    high_bins = np.array(high_bins)

    bin = np.zeros(uniz.size + 1)
    bin[:-1] = low_bins
    bin[1:] = high_bins

    # Compute binned statistics
    y_stat_16, binedges, bin_ind = binned_statistic(zs, ys, statistic=lambda
        y: np.percentile(y, 16), bins=bin)
    y_stat_84, binedges, bin_ind = binned_statistic(zs, ys, statistic=lambda
        y: np.percentile(y, 84), bins=bin)

    # Compute bincentres
    bin_cents = uniz

    okinds = np.logical_and(~np.isnan(bin_cents),
                            np.logical_and(~np.isnan(y_stat_16),
                                           ~np.isnan(y_stat_84)))

    ax.fill_between(bin_cents[okinds], y_stat_16[okinds], y_stat_84[okinds],
                    alpha=0.3, color=color)


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


def get_data(masslim=1e8, eagle=False):

    if eagle:
        regions = ["EAGLE", ]
    else:
        regions = []
        for reg in range(0, 1):
            if reg < 10:
                regions.append('0' + str(reg))
            else:
                regions.append(str(reg))

    # Define snapshots
    if eagle:
        pre_snaps = ['003_z008p988', '006_z005p971', '009_z004p485',
                     '012_z003p017', '015_z002p012', '018_z001p259',
                     '021_z000p736', '027_z000p101', '001_z015p132',
                     '004_z008p075', '007_z005p487', '010_z003p984',
                     '013_z002p478', '019_z001p004', '028_z000p000',
                     '002_z009p993', '005_z007p050', '008_z005p037',
                     '011_z003p528', '017_z001p487', '020_z000p865',
                     '023_z000p503']

        snaps = np.zeros(29, dtype=object)
        for s in pre_snaps:
            ind = int(s.split('_')[0])
            snaps[ind] = s

        snaps = list(snaps)[1:]
        prog_snaps = snaps[:-1]
    else:
        snaps = ['001_z014p000', '002_z013p000', '003_z012p000',
                 '004_z011p000', '005_z010p000',
                 '006_z009p000', '007_z008p000', '008_z007p000', '009_z006p000',
                 '010_z005p000', '011_z004p770']
        prog_snaps = ['000_z015p000', '001_z014p000', '002_z013p000',
                      '003_z012p000', '004_z011p000', '005_z010p000',
                      '006_z009p000', '007_z008p000', '008_z007p000',
                      '009_z006p000', '010_z005p000']

    stellar_met = []
    stellar_bd = []
    stellar_form_radius = []
    hmrs = []
    stellar_current_radius = []
    stellar_bd_current = []
    stellar_met_current = []
    zs = []

    for reg in regions:

        for snap, prog_snap in zip(snaps, prog_snaps):
            
            if eagle:
                path = '/cosma7/data//Eagle/ScienceRuns/Planck1/L0050N0752/PE/AGNdT9/data/'
            else:
                path = '/cosma/home/dp004/dc-rope1/FLARES/FLARES-1/G-EAGLE_' + reg + '/data'

            # Get particle IDs
            try:
                halo_part_inds = get_part_ids(path, snap, 4, all_parts=False)
            except ValueError:
                continue
            except OSError:
                continue
            except KeyError:
                continue

            print(reg, snap)

            z_str = snap.split('z')[1].split('p')
            z = float(z_str[0] + '.' + z_str[1])
            z_str = prog_snap.split('z')[1].split('p')
            z_prog = float(z_str[0] + '.' + z_str[1])

            # Get halo IDs and halo data
            try:
                subgrp_ids = E.read_array('SUBFIND', path, snap,
                                          'Subhalo/SubGroupNumber',
                                          numThreads=8)
                grp_ids = E.read_array('SUBFIND', path, snap,
                                       'Subhalo/GroupNumber',
                                       numThreads=8)
                gal_ms = E.read_array('SUBFIND', path, snap,
                                      'Subhalo/ApertureMeasurements/Mass/030kpc',
                                      noH=True, physicalUnits=True,
                                      numThreads=8)[:, 4] * 10**10
                gal_hmr = E.read_array('SUBFIND', path, snap,
                                       'Subhalo/HalfMassRad',
                                       noH=True, physicalUnits=True,
                                       numThreads=8)[:, 4]
                gal_cop = E.read_array('SUBFIND', path, snap,
                                       'Subhalo/CentreOfPotential',
                                       noH=True, physicalUnits=True,
                                       numThreads=8)
                gal_bd = E.read_array('PARTDATA', path, snap,
                                      'PartType4/BirthDensity', noH=True,
                                        physicalUnits=True, numThreads=8)
                gal_met = E.read_array('PARTDATA', path, snap,
                                       'PartType4/Metallicity', noH=True,
                                       physicalUnits=True, numThreads=8)
                gal_aborn = E.read_array('PARTDATA', path, snap,
                                         'PartType4/StellarFormationTime',
                                         noH=True, physicalUnits=True,
                                         numThreads=8)
                gal_coords = E.read_array('PARTDATA', path, snap,
                                          'PartType4/Coordinates',
                                          noH=True, physicalUnits=True,
                                          numThreads=8)

            except ValueError:
                continue
            except OSError:
                continue
            except KeyError:
                continue

            # Remove particles not associated to a subgroup
            okinds = np.logical_and(subgrp_ids != 1073741824, gal_ms > masslim)
            grp_ids = grp_ids[okinds]
            subgrp_ids = subgrp_ids[okinds]
            gal_hmr = gal_hmr[okinds]
            gal_cop = gal_cop[okinds]
            halo_ids = np.zeros(grp_ids.size, dtype=float)
            for (ind, g), sg in zip(enumerate(grp_ids), subgrp_ids):
                halo_ids[ind] = float(str(int(g)) + '.%05d' % int(sg))

            for halo, hmr, cop in zip(halo_ids, gal_hmr, gal_cop):

                # Add stars from these galaxies
                part_inds = list(halo_part_inds[halo])
                pos = gal_coords[part_inds, :] - cop
                rs = np.linalg.norm(pos, axis=1)
                parts_bd = gal_bd[part_inds]
                parts_met = gal_met[part_inds]
                parts_aborn = gal_aborn[part_inds]

                okinds = (1 / parts_aborn) - 1 < z_prog

                stellar_form_radius.extend(rs[okinds] / hmr)
                hmrs.extend(np.full(len(parts_met[okinds]), hmr))
                stellar_bd.extend(parts_bd[okinds])
                stellar_met.extend(parts_met[okinds])

                if snap == '011_z004p770' or snap == '028_z000p000':
                    stellar_bd_current.extend(parts_bd)
                    stellar_met_current.extend(parts_met)
                    stellar_current_radius.extend(rs / hmr)

                zs.extend(np.full(len(parts_met[okinds]), z))

    return stellar_bd, stellar_met, stellar_form_radius, zs, stellar_bd_current, stellar_met_current, stellar_current_radius, hmrs

stellar_bd, stellar_met, stellar_form_radius, zs, stellar_bd_current, stellar_met_current, stellar_current_radius, hmrs = get_data(masslim=10**9.5)

eagle_stellar_bd, eagle_stellar_met, eagle_stellar_form_radius, eagle_zs, eagle_stellar_bd_current, eagle_stellar_met_current, eagle_stellar_current_radius, eagle_hmrs = get_data(masslim=10**9.5, eagle=True)

stellar_bd_all = np.concatenate((stellar_bd, eagle_stellar_bd)) * 10**10
stellar_met_all = np.concatenate((stellar_met, eagle_stellar_met))
stellar_formr_all = np.concatenate((stellar_form_radius,
                                    eagle_stellar_form_radius))
zs_all = np.concatenate((zs, eagle_zs))
hmrs_all = np.concatenate((hmrs, eagle_hmrs))

current_radius_all = np.concatenate((stellar_current_radius,
                                     eagle_stellar_current_radius))
current_stellar_bd_all = np.concatenate((stellar_bd_current, eagle_stellar_bd_current)) * 10**10
current_stellar_met_all = np.concatenate((stellar_met_current, eagle_stellar_met_current))

fig = plt.figure()
gs = gridspec.GridSpec(nrows=3, ncols=1)
gs.update(wspace=0.0, hspace=0.0)
ax1 = fig.add_subplot(gs[0, 0])
ax2 = fig.add_subplot(gs[1, 0])
ax3 = fig.add_subplot(gs[2, 0])

ax1.hexbin(stellar_formr_all, stellar_bd_all, gridsize=100, mincnt=1,
           xscale='log', yscale='log', norm=LogNorm(),
           linewidths=0.2, cmap='viridis', alpha=0.7)

ax2.hexbin(stellar_formr_all, stellar_met_all, gridsize=100, mincnt=1,
           xscale='log', norm=LogNorm(),
           linewidths=0.2, cmap='viridis', alpha=0.7)

ax3.hexbin(stellar_formr_all, zs_all, gridsize=100, mincnt=1,
           xscale='log', yscale='log', norm=LogNorm(),
           linewidths=0.2, cmap='viridis', alpha=0.7)

ax1.set_ylabel(r"$\rho_{\mathrm{birth}}$ / [$M_\odot$ Mpc$^{-3}$]")
ax2.set_ylabel(r"$Z$")
ax3.set_ylabel("$z_{\mathrm{form}}$")
ax3.set_xlabel("$R/R_{1/2}$")

fig.savefig("plots/stellar_formation_radius.png", bbox_inches="tight")

plt.close(fig)

bd_lims = []
met_lims = []

fig = plt.figure()
gs = gridspec.GridSpec(nrows=2, ncols=2)
gs.update(wspace=0.0, hspace=0.0)
ax1 = fig.add_subplot(gs[0, 0])
ax2 = fig.add_subplot(gs[1, 0])
ax3 = fig.add_subplot(gs[0, 1])
ax4 = fig.add_subplot(gs[1, 1])

ax1.hexbin(stellar_current_radius, stellar_bd_current * 10**10, gridsize=100, mincnt=1,
           xscale='log', yscale='log', norm=LogNorm(),
           linewidths=0.2, cmap='viridis', alpha=0.7)

ax1.text(0.8, 0.9, "FLARES",
        bbox=dict(boxstyle="round,pad=0.3", fc='w', ec="k", lw=1, alpha=0.8),
        transform=ax1.transAxes, horizontalalignment='right', fontsize=8)

ax2.hexbin(stellar_current_radius, stellar_met_current, gridsize=100, mincnt=1,
           xscale='log', norm=LogNorm(),
           linewidths=0.2, cmap='viridis', alpha=0.7)

ax3.hexbin(eagle_stellar_current_radius, eagle_stellar_bd_current * 10**10, gridsize=100, mincnt=1,
           xscale='log', yscale='log', norm=LogNorm(),
           linewidths=0.2, cmap='viridis', alpha=0.7)

ax3.text(0.8, 0.9, "EAGLE",
        bbox=dict(boxstyle="round,pad=0.3", fc='w', ec="k", lw=1, alpha=0.8),
        transform=ax3.transAxes, horizontalalignment='right', fontsize=8)

ax4.hexbin(eagle_stellar_current_radius, eagle_stellar_met_current, gridsize=100, mincnt=1,
           xscale='log', norm=LogNorm(),
           linewidths=0.2, cmap='viridis', alpha=0.7)

ax1.set_ylabel(r"$\rho_{\mathrm{birth}}$ / [$M_\odot$ Mpc$^{-3}$]")
ax2.set_ylabel(r"$Z$")
ax2.set_xlabel("$R/R_{1/2}$")
ax4.set_xlabel("$R/R_{1/2}$")

bd_lims.extend(ax1.get_ylim())
bd_lims.extend(ax3.get_ylim())
met_lims.extend(ax2.get_ylim())
met_lims.extend(ax4.get_ylim())

ax1.set_ylim(np.min(bd_lims), np.max(bd_lims))
ax2.set_ylim(np.min(met_lims), np.max(met_lims))
ax3.set_ylim(np.min(bd_lims), np.max(bd_lims))
ax4.set_ylim(np.min(met_lims), np.max(met_lims))

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

fig.savefig("plots/stellar_formation_radius_afterevo.png", bbox_inches="tight")

plt.close(fig)

# EAGLE parameters
parameters = {"f_th,min": 0.3,
              "f_th,max": 3,
              "n_Z": 1.0,
              "n_n": 1.0,
              "Z_pivot": 0.1 * 0.012,
              "n_pivot": 0.67}

star_formation_parameters = {"threshold_Z0": 0.002,
                             "threshold_n0": 0.1,
                             "slope": -0.64}


number_of_bins = 128

# Constants; these could be put in the parameter file but are rarely changed.
birth_density_bins = np.logspace(-3, 6.8, number_of_bins)
metal_mass_fraction_bins = np.logspace(-5.9, 0, number_of_bins)

# Now need to make background grid of f_th.
birth_density_grid, metal_mass_fraction_grid = np.meshgrid(
    0.5 * (birth_density_bins[1:] + birth_density_bins[:-1]),
    0.5 * (metal_mass_fraction_bins[1:] + metal_mass_fraction_bins[:-1]))

f_th_grid = parameters["f_th,min"] + (parameters["f_th,max"] - parameters["f_th,min"]) / (
    1.0
    + (metal_mass_fraction_grid / parameters["Z_pivot"]) ** parameters["n_Z"]
    * (birth_density_grid / parameters["n_pivot"]) ** (-parameters["n_n"])
)

# Set up plot
fig = plt.figure()
ax = fig.add_subplot(111)

ax.loglog()

mappable = ax.pcolormesh(birth_density_bins, metal_mass_fraction_bins,
                         f_th_grid, vmin=0.3, vmax=3)

# Add colorbars
cax1 = ax.inset_axes([0.65, 0.1, 0.3, 0.03])
cbar1 = fig.colorbar(mappable, cax=cax1, orientation="horizontal")
cbar1.ax.set_xlabel(r'$f_{th}$', labelpad=1.5, fontsize=9)
cbar1.ax.xaxis.set_label_position('top')
cbar1.ax.tick_params(axis='x', labelsize=8)

H, _, _ = np.histogram2d(stellar_bd_all, stellar_met_all,
                         bins=[birth_density_bins, metal_mass_fraction_bins])

ax.contour(birth_density_grid, metal_mass_fraction_grid, H.T, levels=6, cmap="magma")

# Add line showing SF law
sf_threshold_density = star_formation_parameters["threshold_n0"] * \
                       (metal_mass_fraction_bins
                        / star_formation_parameters["threshold_Z0"]) ** (star_formation_parameters["slope"])
ax.plot(sf_threshold_density, metal_mass_fraction_bins, linestyle="dashed", label="SF threshold")


# Label axes
ax.set_xlabel(r"$\rho_{\mathrm{birth}}$ / [$M_\odot$ Mpc$^{-3}$]")
ax.set_ylabel(r"$Z$")

legend = ax.legend(markerfirst=True, loc="lower left", fontsize=8)
plt.setp(legend.get_texts())

try:
    fontsize = legend.get_texts()[0].get_fontsize()
except:
    fontsize = 6


ax.text(0.975, 0.025, "\n".join([f"${k.replace('_', '_{') + '}'}$: ${v:.4g}$" for k, v in parameters.items()]),
         color="k", transform=ax9.transAxes, ha="right", va="bottom", fontsize=fontsize)

ax.text(0.975, 0.975, "Contour lines \n linearly spaced", color="k", transform=ax3.transAxes, ha="right", va="top",
         fontsize=fontsize)

fig.savefig('plots/birthdensity_metallicity_EAGLE+FLARES.png', bbox_inches='tight')
