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


def get_data(eagle=False, ref=False):

    if eagle or ref:
        regions = ["EAGLE", ]
    else:
        regions = []
        for reg in range(0, 40):
            if reg < 10:
                regions.append("0" + str(reg))
            else:
                regions.append(str(reg))

    # Define snapshots
    if eagle or ref:
        snaps = ["027_z000p101", ]
    else:
        snaps = ["011_z004p770", ]

    stellar_met = []
    stellar_bd = []
    ovden = []
    zs = []
    
    if eagle or ref:
        ovds = [1, ]
    else:
        ovds = np.loadtxt("region_overdensity.txt", dtype=float)

    for reg, ovd in zip(regions, ovds):

        for snap in snaps:
            
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

            z_str = snap.split("z")[1].split("p")
            z = float(z_str[0] + "." + z_str[1])

            # Get halo IDs and halo data
            try:
                # parts_subgroup = E.read_array("PARTDATA", path, snap,
                #                               "PartType4/SubGroupNumber",
                #                               noH=True,
                #                               physicalUnits=True,
                #                               numThreads=8)
                parts_bd = E.read_array("PARTDATA", path, snap,
                                      "PartType4/BirthDensity", noH=True,
                                        physicalUnits=True, numThreads=8)
                parts_met = E.read_array("PARTDATA", path, snap,
                                       "PartType4/Metallicity", noH=True,
                                       physicalUnits=True, numThreads=8)
                parts_aborn = E.read_array("PARTDATA", path, snap,
                                         "PartType4/StellarFormationTime",
                                         noH=True, physicalUnits=True,
                                         numThreads=8)
                # gal_ms = E.read_array("SUBFIND", path, snap,
                #                       "Subhalo/ApertureMeasurements/"
                #                       "Mass/030kpc",
                #                       noH=True, physicalUnits=True,
                #                       numThreads=8)[:, 4] * 10 ** 10
            except ValueError:
                print(reg, snap, "No data")
                continue
            except OSError:
                print(reg, snap, "No data")
                continue
            except KeyError:
                print(reg, snap, "No data")
                continue

            # okinds = parts_subgroup != 2**30
            # parts_subgroup = parts_subgroup[okinds]
            # parts_bd = parts_bd[okinds]
            # parts_met = parts_met[okinds]
            # parts_aborn = parts_aborn[okinds]
            #
            # part_sub_mass = gal_ms[parts_subgroup]
            #
            # okinds = part_sub_mass > 10**9
            #
            # # Add stars from these galaxies
            # stellar_bd.extend((parts_bd[okinds] * 10**10
            #                    * Msun / Mpc ** 3 / mh).to(1 / cm ** 3).value)
            # stellar_met.extend(parts_met[okinds])
            # zs.extend((1 / parts_aborn[okinds]) - 1)
            # ovden.extend(np.full_like(parts_bd[okinds], ovd))

            # Add stars from these galaxies
            stellar_bd.extend((parts_bd * 10**10
                               * Msun / Mpc ** 3 / mh).to(1 / cm ** 3).value)
            stellar_met.extend(parts_met)
            zs.extend((1 / parts_aborn) - 1)
            ovden.extend(np.full_like(parts_bd, ovd))

    return np.array(stellar_bd), np.array(stellar_met), \
           np.array(zs), np.array(ovden)


log1pdelta = ovds = np.loadtxt("region_overdensity.txt", dtype=float)

stellar_bd, stellar_met, zs, ovdens = get_data()

agndt9_stellar_bd, agndt9_stellar_met, agndt9_zs, _ = get_data(eagle=True)

ref_stellar_bd, ref_stellar_met, ref_zs, _ = get_data(ref=True)

zs_all = np.concatenate((zs, ref_zs, agndt9_zs))
stellar_bd_all = np.concatenate((stellar_bd, ref_stellar_bd,
                                 agndt9_stellar_bd))
stellar_met_all = np.concatenate((stellar_met, ref_stellar_met,
                                  agndt9_stellar_met))

dbinLims = [-0.3, -0.15, -0.04, 0.04, 0.12, 0.22, 0.3]
dbins = dbinLims[:-1] + np.diff(dbinLims)/2
dbins = np.array(["%.2f" % db for db in dbins]).astype(float)

N_regions = np.histogram(log1pdelta[::-1], dbinLims)[0]
bin_labels = ["[%.2f - %.2f] (%i)" % (db1, db2, _N)
              for db1, db2, _N in zip(dbinLims[:-1],
                                      dbinLims[1:], N_regions)]


dselect = np.digitize(log1pdelta, dbinLims) - 1
dindex = np.arange(0, np.max(dselect)+1)

ticks = np.linspace(0.05, .95, len(dindex))
_cmap = plt.cm.get_cmap("plasma", len(ticks))
# colors = [ cm.plasma(i) for i in ticks ]

fig = plt.figure()
ax = fig.add_subplot(111)

ax.hexbin(zs_all, stellar_bd_all, gridsize=100, mincnt=1, yscale="log", 
          norm=LogNorm(), linewidths=0.2, cmap="Greys", alpha=0.4)

plot_meidan_stat(np.array(agndt9_zs), np.array(agndt9_stellar_bd), ax,
                 lab="AGNdT9: L0050N0752", color="royalblue", bins=None,
                 ls="dashdot")
plot_meidan_stat(np.array(ref_zs), np.array(ref_stellar_bd),
                 ax, lab="REFERENCE: L0100N1504", color="limegreen",
                 bins=None, ls="--")

ax.plot((40, 90), (10**1, 10**3), color="k", linestyle="-", label="FLARES")

for low, up, c in zip(dbinLims[:-1], dbinLims[1:], _cmap.colors):

    print(low, up, c)

    okinds = np.logical_and(ovdens >= low, ovdens < up)

    plot_meidan_stat(np.array(zs)[okinds], np.array(stellar_bd)[okinds],
                     ax, lab=None, color=c,
                     bins=None, ls="-")

ax.set_xlim(-0.1, 27)

sm = plt.cm.ScalarMappable(cmap=_cmap, norm=plt.Normalize(vmin=0., vmax=1.))
sm._A = []  # # fake up the array of the scalar mappable
cbaxes = ax.inset_axes([0.7, 0.625, 0.03, 0.35])
cbar = plt.colorbar(sm, ticks=ticks, cax=cbaxes)
cbar.ax.set_yticklabels(bin_labels, fontsize=8)
cbar.ax.set_ylabel("$[\mathrm{log_{10}}(1 \,+\,\delta)] \; "
                   "(N_{\mathrm{regions}})$", size=9, rotation=90)

ax.set_xlabel("$z$")
ax.set_ylabel(r"$<\rho_{\mathrm{birth}}>$ / [cm$^{-3}$]")

handles, labels = ax.get_legend_handles_labels()
ax.legend(handles, labels, loc="lower right")

ax.set_yscale("log")

fig.savefig("plots/stellarbd_z_evolution_od.png", bbox_inches="tight")

plt.close(fig)

fig = plt.figure()
ax = fig.add_subplot(111)

plot_meidan_stat(np.array(agndt9_zs), np.array(agndt9_stellar_bd), ax,
                 lab="AGNdT9: L0050N0752", color="royalblue", bins=None,
                 ls="dashdot")
plot_meidan_stat(np.array(ref_zs), np.array(ref_stellar_bd),
                 ax, lab="REFERENCE: L0100N1504", color="limegreen",
                 bins=None, ls="--")

ax.plot((40, 90), (10**1, 10**3), color="k", linestyle="-", label="FLARES")

for low, up, c in zip(dbinLims[:-1], dbinLims[1:], _cmap.colors):

    print(low, up, c)

    okinds = np.logical_and(ovdens >= low, ovdens < up)

    plot_meidan_stat(np.array(zs)[okinds], np.array(stellar_bd)[okinds],
                     ax, lab=None, color=c,
                     bins=None, ls="-")

ax.set_xlim(None, 32)

sm = plt.cm.ScalarMappable(cmap=_cmap, norm=plt.Normalize(vmin=0., vmax=1.))
sm._A = []  # # fake up the array of the scalar mappable
cbaxes = ax.inset_axes([0.7, 0.625, 0.03, 0.35])
cbar = plt.colorbar(sm, ticks=ticks, cax=cbaxes)
cbar.ax.set_yticklabels(bin_labels, fontsize=8)
cbar.ax.set_ylabel("$[\mathrm{log_{10}}(1 \,+\,\delta)] \; "
                   "(N_{\mathrm{regions}})$", size=9, rotation=90)

ax.set_xlabel("$z$")
ax.set_ylabel(r"$<\rho_{\mathrm{birth}}>$ / [cm$^{-3}$]")

handles, labels = ax.get_legend_handles_labels()
ax.legend(handles, labels, loc="lower right")

ax.set_yscale("log")

fig.savefig("plots/stellarbd_z_evolution_od_nohex.png", bbox_inches="tight")

plt.close(fig)

fig = plt.figure()
ax = fig.add_subplot(111)

ax.hexbin(stellar_bd_all, stellar_met_all, gridsize=100, mincnt=1,
          xscale="log", norm=LogNorm(), linewidths=0.2, cmap="Greys")

ax.set_xlabel(r"$<\rho_{\mathrm{birth}}>$ / [cm$^{-3}$]")
ax.set_ylabel(r"$<Z>$")

fig.savefig("plots/stellarbd_vs_stellarz.png", bbox_inches="tight")

plt.close(fig)

fig = plt.figure()
ax = fig.add_subplot(111)

ax.hexbin(zs_all, stellar_met_all, gridsize=100, mincnt=1,
          norm=LogNorm(), linewidths=0.2, cmap="Greys", alpha=0.4)

plot_meidan_stat(np.array(agndt9_zs), np.array(agndt9_stellar_met), ax,
                 lab="AGNdT9: L0050N0752", color="royalblue", bins=None,
                 ls="dashdot")
plot_meidan_stat(np.array(ref_zs), np.array(ref_stellar_met),
                 ax, lab="REFERENCE: L0100N1504", color="limegreen",
                 bins=None, ls="--")

ax.plot((40, 90), (0.2, 0.5), color="k", linestyle="-", label="FLARES")

for low, up, c in zip(dbins[:-1], dbins[1:], _cmap.colors):

    okinds = np.logical_and(ovdens >= low, ovdens < up)

    plot_meidan_stat(np.array(zs)[okinds], np.array(stellar_met)[okinds],
                     ax, lab=None, color=c,
                     bins=None, ls="-")

ax.set_xlim(None, 27)

sm = plt.cm.ScalarMappable(cmap=_cmap, norm=plt.Normalize(vmin=0., vmax=1.))
sm._A = []  # # fake up the array of the scalar mappable
cbaxes = ax.inset_axes([0.7, 0.625, 0.03, 0.35])
cbar = plt.colorbar(sm, ticks=ticks, cax=cbaxes)
cbar.ax.set_yticklabels(bin_labels, fontsize=8)
cbar.ax.set_ylabel("$[\mathrm{log_{10}}(1 \,+\,\delta)] \; "
                   "(N_{\mathrm{regions}})$", size=9, rotation=90)

ax.set_xlabel("$z$")
ax.set_ylabel(r"$<Z>$")

handles, labels = ax.get_legend_handles_labels()
ax.legend(handles, labels, loc="upper left")

fig.savefig("plots/stellarmet_z_evolution_od.png", bbox_inches="tight")

plt.close(fig)

fig = plt.figure()
ax = fig.add_subplot(111)

plot_meidan_stat(np.array(agndt9_zs), np.array(agndt9_stellar_met), ax,
                 lab="AGNdT9: L0050N0752", color="royalblue", bins=None,
                 ls="dashdot")
plot_meidan_stat(np.array(ref_zs), np.array(ref_stellar_met),
                 ax, lab="REFERENCE: L0100N1504", color="limegreen",
                 bins=None, ls="--")

ax.plot((40, 90), (0.0001, 0.00002), color="k", linestyle="-", label="FLARES")

for low, up, c in zip(dbins[:-1], dbins[1:], _cmap.colors):

    okinds = np.logical_and(ovdens >= low, ovdens < up)

    plot_meidan_stat(np.array(zs)[okinds], np.array(stellar_met)[okinds],
                     ax, lab=None, color=c,
                     bins=None, ls="-")

ax.set_xlim(None, 27)

sm = plt.cm.ScalarMappable(cmap=_cmap, norm=plt.Normalize(vmin=0., vmax=1.))
sm._A = []  # # fake up the array of the scalar mappable
cbaxes = ax.inset_axes([0.7, 0.625, 0.03, 0.35])
cbar = plt.colorbar(sm, ticks=ticks, cax=cbaxes)
cbar.ax.set_yticklabels(bin_labels, fontsize=8)
cbar.ax.set_ylabel("$[\mathrm{log_{10}}(1 \,+\,\delta)] \; "
                   "(N_{\mathrm{regions}})$", size=9, rotation=90)

ax.set_xlabel("$z$")
ax.set_ylabel(r"$<Z>$")

handles, labels = ax.get_legend_handles_labels()
ax.legend(handles, labels, loc="upper left")

fig.savefig("plots/stellarmet_z_evolution_od_nohex.png", bbox_inches="tight")
