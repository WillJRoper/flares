#!/cosma/home/dp004/dc-rope1/.conda/envs/flares-env/bin/python
"""
Creates the plot of metallicity against birth density, with
the background coloured by f_th.
"""

import matplotlib.pyplot as plt
import numpy as np
import pickle
import eagle_IO.eagle_IO as E
from scipy.stats import binned_statistic
from astropy.cosmology import Planck13 as cosmo
from astropy.cosmology import z_at_value
from matplotlib.colors import LogNorm
from unyt import mh, cm, Gyr, g, Msun, Mpc
import matplotlib.gridspec as gridspec
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


def plot_meidan_stat(xs, ys, ax, lab, color, bins=None, ls="-"):

    if bins == None:
        bin = np.logspace(np.log10(xs.min()),
                          np.log10(xs.max()), 40)
    elif bins == "lin":
        bin = np.linspace(xs.min(),
                          xs.max(), 40)
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
    y_stat, binedges, bin_ind = binned_statistic(xs, ys, statistic="median",
                                                 bins=bin)

    # Compute bincentres
    bin_wid = binedges[1] - binedges[0]
    bin_cents = binedges[1:] - bin_wid / 2

    okinds = np.logical_and(~np.isnan(bin_cents), ~np.isnan(y_stat))
    if lab != None:
        ax.plot(bin_cents[okinds], y_stat[okinds], color=color, linestyle=ls,
                label=lab)
    else:
        ax.plot(bin_cents[okinds], y_stat[okinds], color=color, linestyle=ls)


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


def get_data(eagle=False, ref=False):

    if eagle or ref:
        regions = ["EAGLE", ]
    else:
        regions = []
        for reg in range(20, 21):
            if reg < 10:
                regions.append("0" + str(reg))
            else:
                regions.append(str(reg))

    # Define snapshots
    if eagle or ref:
        pre_snaps = ['000_z020p000', '003_z008p988', '006_z005p971',
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

        snaps = list(snaps[1:])
        prog_snaps = snaps[:-1]
    else:
        # Define snapshots
        snaps = ['003_z012p000', '004_z011p000', '005_z010p000',
                 '006_z009p000', '007_z008p000', '008_z007p000',
                 '009_z006p000', '010_z005p000', '011_z004p770']
        prog_snaps = ['002_z013p000', '003_z012p000', '004_z011p000',
                      '005_z010p000', '006_z009p000', '007_z008p000',
                      '008_z007p000', '009_z006p000', '010_z005p000']

    stellar_formz = []
    stellar_formr = []
    ovden = []

    for reg, ovd in zip(regions, ovds):

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
            # Get particle IDs
            halo_part_inds = get_part_ids(path, snap, 4, all_parts=False)

            # Get halo IDs and halo data
            try:
                subgrp_ids = E.read_array('SUBFIND', path, snap,
                                          'Subhalo/SubGroupNumber',
                                          numThreads=8)
                grp_ids = E.read_array('SUBFIND', path, snap,
                                       'Subhalo/GroupNumber', numThreads=8)
                gal_ms = E.read_array('SUBFIND', path, snap,
                                      'Subhalo/ApertureMeasurements/Mass/030kpc',
                                      noH=True, physicalUnits=True,
                                      numThreads=8)[:, 4] * 10 ** 10
                gal_coord = E.read_array('PARTDATA', path, snap,
                                         'PartType4/Coordinates', noH=True,
                                         physicalUnits=True, numThreads=8)
                gal_cop = E.read_array('SUBFIND', path, snap,
                                       'Subhalo/CentreOfPotential', noH=True,
                                       physicalUnits=True, numThreads=8)
                gal_aborn = E.read_array('PARTDATA', path, snap,
                                         'PartType4/StellarFormationTime',
                                         numThreads=8)
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
            okinds = np.logical_and(subgrp_ids != 1073741824, gal_ms > 10**9)
            grp_ids = grp_ids[okinds]
            subgrp_ids = subgrp_ids[okinds]
            gal_cop = gal_cop[okinds]
            halo_ids = np.zeros(grp_ids.size, dtype=float)
            for (ind, g), sg in zip(enumerate(grp_ids), subgrp_ids):
                halo_ids[ind] = float(str(int(g)) + '.%05d' % int(sg))

            print("There are", len(halo_ids), "galaxies in snapshot", snap,
                  "in region", reg)

            for halo, cop in zip(halo_ids, gal_cop):

                # Add stars from these galaxies
                part_inds = list(halo_part_inds[halo])
                parts_rs = np.linalg.norm(gal_coord[part_inds, :] - cop,
                                          axis=1)
                parts_aborn = gal_aborn[part_inds]
                z_born = (1 / parts_aborn) - 1
                ok_inds = np.logical_and(z_born < prog_z,
                                         parts_rs < 0.03)
                stellar_formz.extend(z_born [ok_inds])
                stellar_formr.extend(parts_rs[ok_inds] * 1e3)

    return np.array(stellar_formr), np.array(stellar_formz),


log1pdelta = ovds = np.loadtxt("region_overdensity.txt", dtype=float)

stellar_formr, stellar_formz = get_data()
stellar_formr_ref, stellar_formz_ref = get_data(ref=True)
stellar_formr_agndT9, stellar_formz_agndT9 = get_data(eagle=True)

zs_all = np.concatenate((stellar_formz, stellar_formz_ref, stellar_formz_agndT9))
stellar_formr_all = np.concatenate((stellar_formr, stellar_formr_ref,
                                 stellar_formr_agndT9))

fig = plt.figure()
ax = fig.add_subplot(111)

ax.hexbin(zs_all, stellar_formr_all, gridsize=100, mincnt=1, yscale="log",
          norm=LogNorm(), linewidths=0.2, cmap="Greys", alpha=0.4)

plot_meidan_stat(np.array(stellar_formz_agndT9), np.array(stellar_formr_agndT9), ax,
                 lab="AGNdT9: L0050N0752", color="royalblue", bins=None,
                 ls="dashdot")
plot_meidan_stat(np.array(stellar_formz_ref), np.array(stellar_formr_ref),
                 ax, lab="REFERENCE: L0100N1504", color="limegreen",
                 bins=None, ls="--")
plot_meidan_stat(np.array(stellar_formz), np.array(stellar_formr),
                 ax, lab="FLARES", color="m",
                 bins=None, ls="--")

ax.set_xlim(-0.1, 27)

ax.set_xlabel("$z$")
ax.set_ylabel(r"$R_{\star, \mathrm{form}}$ / [pkpc]")

handles, labels = ax.get_legend_handles_labels()
ax.legend(handles, labels, loc="lower right")

ax.set_yscale("log")

fig.savefig("plots/stellarform_r_evolution.png", bbox_inches="tight")

plt.close(fig)
