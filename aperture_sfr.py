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
from unyt import mh, cm, Gyr, g, Msun, Mpc
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


def plot_meidan_stat(xs, ys, ax, lab, color, bins=None, ls='-', xy=True):

    zs = np.float64(xs)

    uniz = np.unique(zs)
    print(uniz)
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

    # Compute bin centres
    bin_wid = binedges[1] - binedges[0]
    bin_cents = binedges[1:] - bin_wid / 2

    print(bin_cents)
    print(y_stat)

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


def get_data(masslim=1e8, eagle=False, ref=False):

    if eagle or ref:
        regions = ["EAGLE", ]
    else:
        regions = []
        for reg in range(0, 1):
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
        prog_snaps = ['000_z0100p000', '001_z014p000', '002_z013p000',
                      '003_z012p000', '004_z011p000', '005_z010p000',
                      '006_z009p000', '007_z008p000', '008_z007p000',
                      '009_z006p000', '010_z005p000']

    sfr_in = []
    sfr_out = []
    zs = []
    mass = []

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

            # Get halo IDs and halo data
            try:
                subgrp_ids = E.read_array('SUBFIND', path, snap,
                                          'Subhalo/SubGroupNumber',
                                          numThreads=8)
                gal_ms = E.read_array('SUBFIND', path, snap,
                                      'Subhalo/ApertureMeasurements/Mass/030kpc',
                                      noH=True, physicalUnits=True,
                                      numThreads=8)[:, 4] * 10**10
                sfr_1kpc = E.read_array('SUBFIND', path, snap,
                                       'Subhalo/ApertureMeasurements/SFR/001kpc',
                                       noH=True, physicalUnits=True,
                                       numThreads=8)
                sfr_30kpc = E.read_array('SUBFIND', path, snap,
                                         'Subhalo/ApertureMeasurements/SFR/030kpc',
                                         noH=True, physicalUnits=True,
                                         numThreads=8)

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
            sfr_1kpc = sfr_1kpc[okinds]
            sfr_30kpc = sfr_30kpc[okinds]
            gal_ms = gal_ms[okinds]

            out = sfr_30kpc - sfr_1kpc
            out[out <= 0] = 0

            sfr_out.extend(out)
            sfr_in.extend(sfr_1kpc)
            zs.extend(np.full(len(gal_ms), z))
            mass.extend(gal_ms)

    return np.array(sfr_in), np.array(sfr_out), np.array(zs), np.array(mass)

sfr_in, sfr_out, zs, masses = get_data(masslim=10**8)

agndt9_sfr_in, agndt9_sfr_out, agndt9_zs, agndt9_masses = get_data(masslim=10**8, eagle=True)

ref_sfr_in, ref_sfr_out, ref_zs, ref_masses = get_data(masslim=10**8, ref=True)

sfrin_all = np.concatenate((sfr_in,
                            agndt9_sfr_in,
                            ref_sfr_in))
sfrout_all = np.concatenate((sfr_out,
                             agndt9_sfr_out,
                             ref_sfr_out))
zs_all = np.concatenate((zs, agndt9_zs, ref_zs))

mass_all = np.concatenate((masses, agndt9_masses, ref_masses))

fig = plt.figure(figsize=(5, 9))
ax = fig.add_subplot(111)

okinds = np.logical_and(sfrout_all > 0, sfrin_all > 0)

ax.hexbin(zs_all[okinds], sfrout_all[okinds] / sfrin_all[okinds],
          gridsize=100, mincnt=1, yscale="log",
          norm=LogNorm(), linewidths=0.2,
          cmap='Greys', alpha=0.4)

ax.axhline(1)

okinds1 = np.logical_and(mass_all > 10**8, mass_all <= 10**9)
okinds2 = np.logical_and(mass_all > 10**9, mass_all <= 10**9.5)
okinds3 = np.logical_and(mass_all > 10**9.5, mass_all <= 10**10)
okinds4 = mass_all > 10**10

plot_meidan_stat(zs_all[okinds1], sfrout_all[okinds1] / sfrin_all[okinds1],
                 ax, lab="$10^8 < M/M_\odot \leq 10^9$",
                 color='darkorange', bins=1)

plot_meidan_stat(zs_all[okinds2], sfrout_all[okinds2] / sfrin_all[okinds2],
                 ax, lab="$10^9 < M/M_\odot \leq 10^{9.5}$",
                 color='royalblue', bins=1, ls="dashed")

plot_meidan_stat(zs_all[okinds3], sfrout_all[okinds3] / sfrin_all[okinds3],
                 ax, lab="$10^{9.5} < M/M_\odot \leq 10^{10}$",
                 color='limegreen', bins=1, ls="dashdot")

plot_meidan_stat(zs_all[okinds4], sfrout_all[okinds4] / sfrin_all[okinds4],
                 ax, lab="$10^{10} < M/M_\odot$",
                 color='magenta', bins=1, ls="dotted")

ax.set_xlabel("$z$")
ax.set_ylabel("SFR (Extended) / SFR (Core)")

handles, labels = ax.get_legend_handles_labels()
ax.legend(handles, labels)

fig.savefig("plots/aperture_sfr_evolution.png", bbox_inches="tight")

plt.close(fig)

fig = plt.figure(figsize=(5, 9))
ax = fig.add_subplot(111)

combo_zs = np.concatenate((zs_all, zs_all))
combo_sfr = np.concatenate((sfrin_all, sfrout_all))

okinds = combo_sfr > 0

ax.hexbin(combo_zs[okinds], combo_sfr[okinds],
          gridsize=100, mincnt=1, yscale="log",
          norm=LogNorm(), linewidths=0.2,
          cmap='Greys', alpha=0.4)

okinds1 = np.logical_and(mass_all > 10**8, mass_all <= 10**9)
okinds2 = np.logical_and(mass_all > 10**9, mass_all <= 10**9.5)
okinds3 = np.logical_and(mass_all > 10**9.5, mass_all <= 10**10)
okinds4 = mass_all > 10**10

plot_meidan_stat(zs_all[okinds1], sfrout_all[okinds1],
                 ax, lab="Out: $10^8 < M/M_\odot \leq 10^9$",
                 color='darkorange', bins=1, ls="dashed")

plot_meidan_stat(zs_all[okinds2], sfrout_all[okinds2],
                 ax, lab="Out: $10^9 < M/M_\odot \leq 10^{9.5}$",
                 color='royalblue', bins=1, ls="dashed")

plot_meidan_stat(zs_all[okinds3], sfrout_all[okinds3],
                 ax, lab="Out: $10^{9.5} < M/M_\odot \leq 10^{10}$",
                 color='limegreen', bins=1, ls="dashed")

plot_meidan_stat(zs_all[okinds4], sfrout_all[okinds4],
                 ax, lab="Out: $10^{10} < M/M_\odot$",
                 color='magenta', bins=1, ls="dashed")

plot_meidan_stat(zs_all[okinds1], sfrin_all[okinds1],
                 ax, lab="In: $10^8 < M/M_\odot \leq 10^9$",
                 color='darkorange', bins=1, ls="dashed")

plot_meidan_stat(zs_all[okinds2], sfrin_all[okinds2],
                 ax, lab="In: $10^9 < M/M_\odot \leq 10^{9.5}$",
                 color='royalblue', bins=1, ls="dashed")

plot_meidan_stat(zs_all[okinds3], sfrin_all[okinds3],
                 ax, lab="In: $10^{9.5} < M/M_\odot \leq 10^{10}$",
                 color='limegreen', bins=1, ls="dashed")

plot_meidan_stat(zs_all[okinds4], sfrin_all[okinds4],
                 ax, lab="In: $10^{10} < M/M_\odot$",
                 color='magenta', bins=1, ls="dashed")

ax.set_xlabel("$z$")
ax.set_ylabel("SFR")

handles, labels = ax.get_legend_handles_labels()
ax.legend(handles, labels)

fig.savefig("plots/aperture_sfr_evolution_split.png", bbox_inches="tight")

plt.close(fig)
