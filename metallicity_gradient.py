#!/cosma/home/dp004/dc-rope1/.conda/envs/flares-env/bin/python
import matplotlib.pyplot as plt
import numpy as np
import pickle
from matplotlib.colors import LogNorm
import matplotlib.gridspec as gridspec
import eagle_IO.eagle_IO as E
from scipy.stats import binned_statistic
from scipy.optimize import curve_fit
import astropy.units as u
from astropy.cosmology import Planck13 as cosmo
from astropy.cosmology import z_at_value
from unyt import mh, cm, Gyr, g, Msun, Mpc
import seaborn as sns

sns.set_style("whitegrid")

def plot_meidan_stat(xs, ys, ax, lab, color, bins=None, ls='-', xy=True):

    if bins == "mass":
        bin_lims = np.linspace(10**8, 10**12, 20)

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
        bin_lims = bin

    bin_cents = []
    y_stat = []
    for low, up in zip(bin_lims[:-1], bin_lims[1:]):

        okinds = np.logical_and(xs > low, xs <= up)

        bin_cents.append(np.median(xs[okinds]))
        y_stat.append(np.nanmedian(ys[okinds]))

    bin_cents = np.array(bin_cents)
    y_stat = np.array(y_stat)

    okinds = np.logical_and(~np.isnan(bin_cents), ~np.isnan(y_stat))

    if xy:
        ax.plot(bin_cents[okinds], y_stat[okinds], color=color, linestyle=ls,
                label=lab)
    else:
        sinds = np.argsort(bin_cents[okinds])
        ax.plot(y_stat[okinds][sinds], bin_cents[okinds][sinds], color=color,
                linestyle=ls,
                label=lab)


def plot_meidan_statyx(xs, ys, ax, lab, color, ls='-'):

    zs_binlims = np.linspace(0, 15, 31)

    zs_plt = np.zeros(50)
    rs_plt = np.zeros(50)
    for ind, (low, up) in enumerate(zip(zs_binlims[:-1], zs_binlims[1:])):
        okinds = np.logical_and(ys >= low, ys < up)
        rs_plt[ind] = np.median(xs[okinds])
        zs_plt[ind] = np.median(ys[okinds])

    ax.plot(rs_plt, zs_plt, color=color, linestyle=ls, label=lab)


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
        part_ids = E.read_array('SNAP', sim, snapshot, 'PartType' + str(part_type) + '/ParticleIDs', numThreads=16)
    else:
        part_ids = E.read_array('PARTDATA', sim, snapshot, 'PartType' + str(part_type) + '/ParticleIDs',
                                numThreads=16)

    # Extract the halo IDs (group names/keys) contained within this snapshot
    group_part_ids = E.read_array('PARTDATA', sim, snapshot, 'PartType' + str(part_type) + '/ParticleIDs',
                                  numThreads=16)
    grp_ids = E.read_array('PARTDATA', sim, snapshot, 'PartType' + str(part_type) + '/GroupNumber',
                           numThreads=16)
    subgrp_ids = E.read_array('PARTDATA', sim, snapshot, 'PartType' + str(part_type) + '/SubGroupNumber',
                              numThreads=16)

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


def strt_fit(x, m, c):
    return m * x + c


def get_data(masslim=1e8, eagle=False, ref=False):

    if eagle or ref:
        regions = ["EAGLE", ]
    else:
        regions = []
        for reg in range(0, 40):
            if reg < 10:
                regions.append('0' + str(reg))
            else:
                regions.append(str(reg))

    # Define snapshots
    if eagle or ref:
        pre_snaps = ['000_z0100p000', '003_z008p988', '006_z005p971',
                     '009_z004p485', '012_z003p017', '015_z002p012',
                     '018_z001p259', '021_z000p736', '024_z000p366',
                     '027_z000p101', '001_z015p132', '004_z008p075',
                     '007_z005p487', '010_z003p984', '013_z002p478',
                     '016_z001p737', '019_z001p004', '022_z000p615',
                     '025_z000p271', '028_z000p000', '002_z009p993',
                     '005_z007p050', '008_z005p037', '011_z003p528',
                     '014_z002p237', '017_z001p487', '020_z000p865',
                     '023_z000p503', '026_z000p183']

        ini_snaps = np.zeros(29, dtype=object)
        for s in pre_snaps:
            ind = int(s.split('_')[0])
            ini_snaps[ind] = s

        ini_snaps = ini_snaps[ini_snaps != 0.0]

        snaps = list(ini_snaps)[1:]
        prog_snaps = list(ini_snaps)[:-1]

    else:
        snaps = ['001_z014p000', '002_z013p000', '003_z012p000',
                 '004_z011p000', '005_z010p000',
                 '006_z009p000', '007_z008p000', '008_z007p000', '009_z006p000',
                 '010_z005p000', '011_z004p770']
        prog_snaps = ['000_z0100p000', '001_z014p000', '002_z013p000',
                      '003_z012p000', '004_z011p000', '005_z010p000',
                      '006_z009p000', '007_z008p000', '008_z007p000',
                      '009_z006p000', '010_z005p000']

    met_grads = []
    recent_met_grads = []
    zs = []
    recent_zs = []
    masses = []
    recent_masses = []

    for reg in regions:

        for snap, prog_snap in zip(snaps, prog_snaps):
            
            if eagle:
                path = "/cosma7/data//Eagle/ScienceRuns/Planck1/" \
                       "L0050N0752/PE/AGNdT9/data/"
            elif ref:
                path = "/cosma7/data//Eagle/ScienceRuns/Planck1/" \
                       "L0100N1504/PE/REFERENCE/data"
            else:
                path = "/cosma/home/dp004/dc-rope1/FLARES/FLARES-1/" \
                       "G-EAGLE_" + reg + "/data"

            print(reg, snap)

            z_str = snap.split('z')[1].split('p')
            z = float(z_str[0] + '.' + z_str[1])
            z_str = prog_snap.split('z')[1].split('p')
            z_prog = float(z_str[0] + '.' + z_str[1])

            z100Myr = z_at_value(cosmo.age, (cosmo.age(z).value - 0.1) * u.Gyr)

            # Get particle IDs
            try:
                halo_part_inds = get_part_ids(path, snap, 4, all_parts=False)
            except ValueError:
                print(reg, snap, "No data")
                continue
            except OSError:
                print(reg, snap, "No data")
                continue
            except KeyError:
                print(reg, snap, "No data")
                continue

            # Get halo IDs and halo data
            try:
                grp_ids = E.read_array('SUBFIND', path, snap,
                                       'Subhalo/GroupNumber',
                                       numThreads=16)
                subgrp_ids = E.read_array('SUBFIND', path, snap,
                                          'Subhalo/SubGroupNumber',
                                          numThreads=16)
                gal_ms = E.read_array('SUBFIND', path, snap,
                                      'Subhalo/ApertureMeasurements/Mass/030kpc',
                                      noH=True, physicalUnits=True,
                                      numThreads=16)[:, 4] * 10**10
                gal_cop = E.read_array('SUBFIND', path, snap,
                                       'Subhalo/CentreOfPotential',
                                       noH=True, physicalUnits=True,
                                       numThreads=16)
                gal_hmr = E.read_array('SUBFIND', path, snap,
                                       'Subhalo/HalfMassRad',
                                       noH=True, physicalUnits=True,
                                       numThreads=16)[:, 4] * 1e3

                # gal_bd = E.read_array('PARTDATA', path, snap,
                #                       'PartType4/BirthDensity', noH=True,
                #                         physicalUnits=True, numThreads=16)
                gal_ox = E.read_array('PARTDATA', path, snap,
                                       'PartType4/SmoothedElementAbundance/'
                                       'Oxygen', noH=True,
                                       physicalUnits=True, numThreads=16)
                gal_hy = E.read_array('PARTDATA', path, snap,
                                       'PartType4/SmoothedElementAbundance/'
                                       'Hydrogen', noH=True,
                                       physicalUnits=True, numThreads=16)
                gal_coords = E.read_array('PARTDATA', path, snap,
                                          'PartType4/Coordinates',
                                          noH=True, physicalUnits=True,
                                          numThreads=16)
                gal_aborn = E.read_array('PARTDATA', path, snap,
                                         'PartType4/StellarFormationTime',
                                         noH=True, physicalUnits=True,
                                         numThreads=16)

            except ValueError:

                print(reg, snap, "No data")

                continue

            except OSError:

                print(reg, snap, "No data")

                continue

            except KeyError:

                print(reg, snap, "No data")

                continue

            # Remove particles not associated to a subgroup
            okinds = np.logical_and(subgrp_ids != 1073741824, gal_ms > masslim)
            grp_ids = grp_ids[okinds]
            subgrp_ids = subgrp_ids[okinds]
            gal_cop = gal_cop[okinds]
            gal_ms = gal_ms[okinds]
            halo_ids = np.zeros(grp_ids.size, dtype=float)
            for (ind, g), sg in zip(enumerate(grp_ids), subgrp_ids):
                halo_ids[ind] = float(str(int(g)) + '.%05d' % int(sg))

            for halo, cop, m, hmr in zip(halo_ids, gal_cop, gal_ms, gal_hmr):

                if hmr == 0.0:
                    continue

                # Add stars from these galaxies
                part_inds = list(halo_part_inds[halo])
                pos = gal_coords[part_inds, :] - cop
                rs = np.linalg.norm(pos, axis=1) * 10**3
                # parts_bd = (gal_bd[part_inds] * 10**10
                #             * Msun / Mpc ** 3 / mh).to(1 / cm ** 3).value
                okinds = gal_ox[part_inds] > 0
                rs = rs[okinds]
                parts_met = 12 + np.log10(gal_ox[part_inds][okinds]
                                          / gal_hy[part_inds][okinds])
                parts_aborn = gal_aborn[part_inds][okinds]

                # okinds = np.logical_and(rs <= 1,
                #                         (1 / parts_aborn) - 1 < z_prog)

                # okinds = np.logical_and(rs <= hmr * 2, rs > hmr * 0.5)
                okinds = rs < 30
                prof_parts_met = parts_met[okinds]
                prof_rs = rs[okinds]

                if len(prof_rs) < 10:
                    continue
                
                popt, pcov = curve_fit(strt_fit, prof_rs,
                                       prof_parts_met,
                                       p0=(-0.5, 0))

                # fig = plt.figure()
                # ax = fig.add_subplot(111)
                #
                # ax.hexbin(rs, parts_met,
                #           gridsize=100, mincnt=1,
                #           norm=LogNorm(), linewidths=0.2, cmap='viridis')
                # xs = np.linspace(0, 30, 100)
                # ax.plot(xs, strt_fit(xs, popt[0], popt[1]), linestyle="--",
                #         color="darkorange")
                #
                # ax.set_xlabel("$R / [\mathrm{pkpc}]$")
                # ax.set_ylabel("$12 + \log_{10}(O/H)$")
                #
                # fig.savefig("plots/metprof_%.1f.png" % np.log10(m),
                #             bbox_inches="tight")
                #
                # plt.close(fig)

                met_grads.append(popt[0])

                zs.append(z)

                masses.append(m)

                okinds = np.logical_and((1 / parts_aborn) - 1 >= z100Myr,
                                        rs < 30)
                recent_parts_met = parts_met[okinds]
                recent_rs = rs[okinds]

                if len(recent_rs) < 10:
                    continue

                popt, pcov = curve_fit(strt_fit, recent_rs,
                                       recent_parts_met,
                                       p0=(-0.5, 0))

                recent_met_grads.append(popt[0])

                recent_zs.append(z)

                recent_masses.append(m)

    return np.array(met_grads), np.array(zs), np.array(masses),\
           np.array(recent_met_grads), np.array(recent_zs), \
           np.array(recent_masses)

met_grads, zs, masses, recent_met_grads, recent_zs, recent_masses = get_data(masslim=10**8)

agndt9_met_grads, agndt9_zs, agndt9_masses, agndt9_recent_met_grads, agndt9_recent_zs, agndt9_recent_masses = get_data(masslim=10**8, eagle=True)

ref_met_grads, ref_zs, ref_masses, ref_recent_met_grads, ref_recent_zs, ref_recent_masses = get_data(masslim=10**8, ref=True)

met_grads_all = np.concatenate((met_grads,
                               agndt9_met_grads,
                               ref_met_grads))

zs_all = np.concatenate((zs,
                         agndt9_zs,
                         ref_zs))

mass_all = np.concatenate((masses,
                           agndt9_masses,
                           ref_masses))

recent_met_grads_all = np.concatenate((recent_met_grads,
                                       agndt9_recent_met_grads,
                                       ref_recent_met_grads))

recent_zs_all = np.concatenate((recent_zs,
                                agndt9_recent_zs,
                                ref_recent_zs))

recent_mass_all = np.concatenate((recent_masses,
                                  agndt9_recent_masses,
                                  ref_recent_masses))

fig = plt.figure()
ax = fig.add_subplot(111)

# ax.hexbin(zs_all, met_grads_all,
#           gridsize=100, mincnt=1,
#           norm=LogNorm(), linewidths=0.2,
#           cmap='Greys', alpha=0.4)

okinds1 = np.logical_and(mass_all > 10**8, mass_all <= 10**9)
okinds2 = np.logical_and(mass_all > 10**9, mass_all <= 10**9.5)
okinds3 = np.logical_and(mass_all > 10**9.5, mass_all <= 10**10)
okinds4 = mass_all > 10**10

plot_meidan_stat(zs_all[okinds1], met_grads_all[okinds1],
                 ax, lab="$10^8 < M_\star/M_\odot \leq 10^9$",
                 color='darkorange', bins=1)

plot_meidan_stat(zs_all[okinds2], met_grads_all[okinds2],
                 ax, lab="$10^9 < M_\star/M_\odot \leq 10^{9.5}$",
                 color='royalblue', bins=1)

plot_meidan_stat(zs_all[okinds3], met_grads_all[okinds3],
                 ax, lab="$10^{9.5} < M_\star/M_\odot \leq 10^{10}$",
                 color='limegreen', bins=1)

plot_meidan_stat(zs_all[okinds4], met_grads_all[okinds4],
                 ax, lab="$10^{10} < M_\star/M_\odot$",
                 color='magenta', bins=1)

ax.set_xlabel("$z$")
ax.set_ylabel(r"$\nabla_{O/H}$")

handles, labels = ax.get_legend_handles_labels()
ax.legend(handles, labels)

# ax.set_ylim(-1, 1)

fig.savefig("plots/stellar_met_grad_evo.png", bbox_inches="tight")

plt.close(fig)

fig = plt.figure()
gs = gridspec.GridSpec(nrows=4, ncols=1)
gs.update(wspace=0.0, hspace=0.0)
ax1 = fig.add_subplot(gs[0, 0])
ax2 = fig.add_subplot(gs[1, 0])
ax3 = fig.add_subplot(gs[2, 0])
ax4 = fig.add_subplot(gs[3, 0])

okinds1 = np.logical_and(zs_all > 0, zs_all <= 1)
okinds2 = np.logical_and(zs_all > 1, zs_all <= 3)
okinds3 = np.logical_and(zs_all > 3, zs_all <= 5)
okinds4 = zs_all > 5

ax1.hexbin(mass_all[okinds1], met_grads_all[okinds1],
          gridsize=100, mincnt=1, xscale="log",
          norm=LogNorm(), linewidths=0.2,
          cmap='plasma')

ax1.text(0.8, 0.9, "$0 < z \leq 1$",
        bbox=dict(boxstyle="round,pad=0.3", fc='w', ec="k", lw=1, alpha=0.8),
        transform=ax1.transAxes, horizontalalignment='right', fontsize=8)

ax2.hexbin(mass_all[okinds2], met_grads_all[okinds2],
          gridsize=100, mincnt=1, xscale="log",
          norm=LogNorm(), linewidths=0.2,
          cmap='plasma')

ax2.text(0.8, 0.9, "$1 < z \leq 3$",
        bbox=dict(boxstyle="round,pad=0.3", fc='w', ec="k", lw=1, alpha=0.8),
        transform=ax2.transAxes, horizontalalignment='right', fontsize=8)

ax3.hexbin(mass_all[okinds3], met_grads_all[okinds3],
          gridsize=100, mincnt=1, xscale="log",
          norm=LogNorm(), linewidths=0.2,
          cmap='plasma')

ax3.text(0.8, 0.9, "$3 < z \leq 5$",
        bbox=dict(boxstyle="round,pad=0.3", fc='w', ec="k", lw=1, alpha=0.8),
        transform=ax3.transAxes, horizontalalignment='right', fontsize=8)

ax4.hexbin(mass_all[okinds4], met_grads_all[okinds4],
          gridsize=100, mincnt=1, xscale="log",
          norm=LogNorm(), linewidths=0.2,
          cmap='plasma')

ax4.text(0.8, 0.9, "$5 < z$",
        bbox=dict(boxstyle="round,pad=0.3", fc='w', ec="k", lw=1, alpha=0.8),
        transform=ax4.transAxes, horizontalalignment='right', fontsize=8)

ax4.set_xlabel("$M_\star/M_\odot$")
for ax in [ax1, ax2, ax3, ax4]:
    ax.set_ylabel(r"$\nabla_{O/H}$")
    ax.axhline(0, linestyle="--", color="k")
    if ax != ax4:
        ax.tick_params(axis='x', top=False, bottom=False, labeltop=False,
                       labelbottom=False)

fig.savefig("plots/stellar_met_grad_mass.png", bbox_inches="tight")

plt.close(fig)

fig = plt.figure()
ax = fig.add_subplot(111)

# ax.hexbin(recent_zs_all, recent_met_grads_all,
#           gridsize=100, mincnt=1,
#           norm=LogNorm(), linewidths=0.2,
#           cmap='Greys', alpha=0.4)

okinds1 = np.logical_and(recent_mass_all > 10**8, recent_mass_all <= 10**9)
okinds2 = np.logical_and(recent_mass_all > 10**9, recent_mass_all <= 10**9.5)
okinds3 = np.logical_and(recent_mass_all > 10**9.5, recent_mass_all <= 10**10)
okinds4 = recent_mass_all > 10**10

plot_meidan_stat(recent_zs_all[okinds1], recent_met_grads_all[okinds1],
                 ax, lab="$10^8 < M_\star/M_\odot \leq 10^9$",
                 color='darkorange', bins=1)

plot_meidan_stat(recent_zs_all[okinds2], recent_met_grads_all[okinds2],
                 ax, lab="$10^9 < M_\star/M_\odot \leq 10^{9.5}$",
                 color='royalblue', bins=1)

plot_meidan_stat(recent_zs_all[okinds3], recent_met_grads_all[okinds3],
                 ax, lab="$10^{9.5} < M_\star/M_\odot \leq 10^{10}$",
                 color='limegreen', bins=1)

plot_meidan_stat(recent_zs_all[okinds4], recent_met_grads_all[okinds4],
                 ax, lab="$10^{10} < M_\star/M_\odot$",
                 color='magenta', bins=1)

ax.set_xlabel("$z$")
ax.set_ylabel(r"$\nabla_{O/H}$")

handles, labels = ax.get_legend_handles_labels()
ax.legend(handles, labels)

# ax.set_ylim(-1, 1)

fig.savefig("plots/stellar_recent_met_grad_evo.png", bbox_inches="tight")

plt.close(fig)

fig = plt.figure()
gs = gridspec.GridSpec(nrows=4, ncols=1)
gs.update(wspace=0.0, hspace=0.0)
ax1 = fig.add_subplot(gs[0, 0])
ax2 = fig.add_subplot(gs[1, 0])
ax3 = fig.add_subplot(gs[2, 0])
ax4 = fig.add_subplot(gs[3, 0])

okinds1 = np.logical_and(zs_all > 0, zs_all <= 1)
okinds2 = np.logical_and(zs_all > 1, zs_all <= 3)
okinds3 = np.logical_and(zs_all > 3, zs_all <= 5)
okinds4 = zs_all > 5

ax1.hexbin(recent_mass_all[okinds1], recent_met_grads_all[okinds1],
          gridsize=100, mincnt=1, xscale="log",
          norm=LogNorm(), linewidths=0.2,
          cmap='plasma')

ax1.text(0.8, 0.9, "$0 < z \leq 1$",
        bbox=dict(boxstyle="round,pad=0.3", fc='w', ec="k", lw=1, alpha=0.8),
        transform=ax1.transAxes, horizontalalignment='right', fontsize=8)

ax2.hexbin(recent_mass_all[okinds2], recent_met_grads_all[okinds2],
          gridsize=100, mincnt=1, xscale="log",
          norm=LogNorm(), linewidths=0.2,
          cmap='plasma')

ax2.text(0.8, 0.9, "$1 < z \leq 3$",
        bbox=dict(boxstyle="round,pad=0.3", fc='w', ec="k", lw=1, alpha=0.8),
        transform=ax2.transAxes, horizontalalignment='right', fontsize=8)

ax3.hexbin(recent_mass_all[okinds3], recent_met_grads_all[okinds3],
          gridsize=100, mincnt=1, xscale="log",
          norm=LogNorm(), linewidths=0.2,
          cmap='plasma')

ax3.text(0.8, 0.9, "$3 < z \leq 5$",
        bbox=dict(boxstyle="round,pad=0.3", fc='w', ec="k", lw=1, alpha=0.8),
        transform=ax3.transAxes, horizontalalignment='right', fontsize=8)

ax4.hexbin(recent_mass_all[okinds4], recent_met_grads_all[okinds4],
          gridsize=100, mincnt=1, xscale="log",
          norm=LogNorm(), linewidths=0.2,
          cmap='plasma')

ax4.text(0.8, 0.9, "$5 < z$",
        bbox=dict(boxstyle="round,pad=0.3", fc='w', ec="k", lw=1, alpha=0.8),
        transform=ax4.transAxes, horizontalalignment='right', fontsize=8)

ax4.set_xlabel("$M_\star/M_\odot$")
for ax in [ax1, ax2, ax3, ax4]:
    ax.set_ylabel(r"$\nabla_{O/H}$")
    ax.axhline(0, linestyle="--", color="k")
    if ax != ax4:
        ax.tick_params(axis='x', top=False, bottom=False, labeltop=False,
                       labelbottom=False)

fig.savefig("plots/stellar_recent_met_grad_mass.png", bbox_inches="tight")

plt.close(fig)
