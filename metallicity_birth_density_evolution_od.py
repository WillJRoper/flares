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


def get_data(eagle=False, ref=False):

    if eagle or ref:
        regions = ["EAGLE", ]
    else:
        regions = []
        for reg in range(0, 1):
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
    fths = []
    masses = []
    
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
                parts_subgroup = E.read_array("PARTDATA", path, snap,
                                              "PartType4/SubGroupNumber",
                                              noH=True,
                                              physicalUnits=True,
                                              numThreads=8)
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
                parts_fth = E.read_array("PARTDATA", path, snap,
                                         "PartType4/Feedback_EnergyFraction",
                                         noH=True, physicalUnits=True,
                                         numThreads=8)
                gal_ms = E.read_array("SUBFIND", path, snap,
                                      "Subhalo/ApertureMeasurements/"
                                      "Mass/030kpc",
                                      noH=True, physicalUnits=True,
                                      numThreads=8)[:, 4] * 10 ** 10
            except ValueError:
                print(reg, snap, "No data")
                continue
            except OSError:
                print(reg, snap, "No data")
                continue
            except KeyError:
                print(reg, snap, "No data")
                continue

            okinds = parts_subgroup != 2**30
            parts_subgroup = parts_subgroup[okinds]
            parts_bd = parts_bd[okinds]
            parts_met = parts_met[okinds]
            parts_aborn = parts_aborn[okinds]
            parts_fth = parts_fth[okinds]

            part_sub_mass = gal_ms[parts_subgroup]

            # okinds = part_sub_mass > 10**9

            # # Add stars from these galaxies
            # stellar_bd.extend((parts_bd[okinds] * 10**10
            #                    * Msun / Mpc ** 3 / mh).to(1 / cm ** 3).value)
            # stellar_met.extend(parts_met[okinds])
            # zs.extend((1 / parts_aborn[okinds]) - 1)
            # ovden.extend(np.full_like(parts_bd[okinds], ovd))
            # fths.extend(parts_fth[okinds])

            # Add stars from these galaxies
            stellar_bd.extend((parts_bd * 10**10
                               * Msun / Mpc ** 3 / mh).to(1 / cm ** 3).value)
            stellar_met.extend(parts_met)
            zs.extend((1 / parts_aborn) - 1)
            ovden.extend(np.full_like(parts_bd, ovd))
            fths.extend(parts_fth)
            masses.extend(part_sub_mass)

    return np.array(stellar_bd), np.array(stellar_met), \
           np.array(zs), np.array(ovden), np.array(fths), np.array(masses)


log1pdelta = ovds = np.loadtxt("region_overdensity.txt", dtype=float)

stellar_bd, stellar_met, zs, ovdens, fth, masses = get_data()

agndt9_stellar_bd, agndt9_stellar_met, agndt9_zs, _, agndt9_fth, agndt9_masses = get_data(eagle=True)

ref_stellar_bd, ref_stellar_met, ref_zs, _, ref_fth, ref_masses = get_data(ref=True)

agndt9_ovdens = np.zeros_like(agndt9_stellar_met)
ref_ovdens = np.zeros_like(ref_stellar_met)

zs_all = np.concatenate((zs, ref_zs, agndt9_zs))
stellar_bd_all = np.concatenate((stellar_bd, ref_stellar_bd,
                                 agndt9_stellar_bd))
stellar_met_all = np.concatenate((stellar_met, ref_stellar_met,
                                  agndt9_stellar_met))

ovdens_all = np.concatenate((ovdens, agndt9_ovdens, ref_ovdens))
fth_all = np.concatenate((fth, agndt9_fth, ref_fth))

masses_all = np.concatenate((masses, agndt9_masses, ref_masses))

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

# fig = plt.figure()
# ax = fig.add_subplot(111)
#
# ax.hexbin(zs_all, stellar_bd_all, gridsize=100, mincnt=1, yscale="log",
#           norm=LogNorm(), linewidths=0.2, cmap="Greys", alpha=0.4)
#
# plot_meidan_stat(np.array(agndt9_zs), np.array(agndt9_stellar_bd), ax,
#                  lab="AGNdT9: L0050N0752", color="royalblue", bins=None,
#                  ls="dashdot")
# plot_meidan_stat(np.array(ref_zs), np.array(ref_stellar_bd),
#                  ax, lab="REFERENCE: L0100N1504", color="limegreen",
#                  bins=None, ls="--")
#
# ax.plot((40, 90), (10**1, 10**3), color="k", linestyle="-", label="FLARES")
#
# for low, up, c in zip(dbinLims[:-1], dbinLims[1:], _cmap.colors):
#
#     print(low, up, c)
#
#     okinds = np.logical_and(ovdens >= low, ovdens < up)
#
#     plot_meidan_stat(np.array(zs)[okinds], np.array(stellar_bd)[okinds],
#                      ax, lab=None, color=c,
#                      bins=None, ls="-")
#
# ax.set_xlim(-0.1, 27)
#
# sm = plt.cm.ScalarMappable(cmap=_cmap, norm=plt.Normalize(vmin=0., vmax=1.))
# sm._A = []  # # fake up the array of the scalar mappable
# cbaxes = ax.inset_axes([0.7, 0.625, 0.03, 0.35])
# cbar = plt.colorbar(sm, ticks=ticks, cax=cbaxes)
# cbar.ax.set_yticklabels(bin_labels, fontsize=8)
# cbar.ax.set_ylabel("$[\mathrm{log_{10}}(1 \,+\,\delta)] \; "
#                    "(N_{\mathrm{regions}})$", size=9, rotation=90)
#
# ax.set_xlabel("$z$")
# ax.set_ylabel(r"$\rho_{\mathrm{birth}}$ / [cm$^{-3}$]")
#
# handles, labels = ax.get_legend_handles_labels()
# ax.legend(handles, labels, loc="lower right")
#
# ax.set_yscale("log")
#
# fig.savefig("plots/stellarbd_z_evolution_od.png", bbox_inches="tight")
#
# plt.close(fig)
#
# fig = plt.figure(figsize=(6, 6))
# gs = gridspec.GridSpec(nrows=2, ncols=2)
# gs.update(wspace=0.0, hspace=0.0)
# ax1 = fig.add_subplot(gs[0, 0])
# ax2 = fig.add_subplot(gs[1, 0])
# ax3 = fig.add_subplot(gs[0, 1])
# ax4 = fig.add_subplot(gs[1, 1])
#
# okinds = np.logical_and(masses_all > 10**8, masses_all <= 10**9)
#
# ax1.hexbin(zs_all[okinds], stellar_bd_all[okinds], gridsize=100, mincnt=1,
#            yscale="log",  norm=LogNorm(),
#            linewidths=0.2, cmap="Greys")
#
# ax1.text(0.8, 0.9, "$10^8 < M/M_\odot \leq 10^9$",
#         bbox=dict(boxstyle="round,pad=0.3", fc='w', ec="k", lw=1, alpha=0.8),
#         transform=ax1.transAxes, horizontalalignment='right', fontsize=8)
#
# okinds = np.logical_and(masses_all > 10**9, masses_all <= 10**9.5)
#
# ax2.hexbin(zs_all[okinds], stellar_bd_all[okinds], gridsize=100, mincnt=1,
#            yscale="log",  norm=LogNorm(),
#            linewidths=0.2, cmap="Greys")
#
# ax2.text(0.8, 0.9, "$10^9 < M/M_\odot \leq 10^{9.5}$",
#         bbox=dict(boxstyle="round,pad=0.3", fc='w', ec="k", lw=1, alpha=0.8),
#         transform=ax2.transAxes, horizontalalignment='right', fontsize=8)
#
# okinds = np.logical_and(masses_all > 10**9.5, masses_all <= 10**10)
#
# ax3.hexbin(zs_all[okinds], stellar_bd_all[okinds], gridsize=100, mincnt=1,
#            yscale="log",  norm=LogNorm(),
#            linewidths=0.2, cmap="Greys")
#
# ax3.text(0.8, 0.9, "$10^{9.5} < M/M_\odot \leq 10^{10}$",
#         bbox=dict(boxstyle="round,pad=0.3", fc='w', ec="k", lw=1, alpha=0.8),
#         transform=ax3.transAxes, horizontalalignment='right', fontsize=8)
#
# okinds = masses_all > 10**10
#
# ax4.hexbin(zs_all[okinds], stellar_bd_all[okinds], gridsize=100, mincnt=1,
#            yscale="log",  norm=LogNorm(),
#            linewidths=0.2, cmap="Greys")
#
# ax4.text(0.8, 0.9, "$10^{10} < M/M_\odot$",
#         bbox=dict(boxstyle="round,pad=0.3", fc='w', ec="k", lw=1, alpha=0.8),
#         transform=ax4.transAxes, horizontalalignment='right', fontsize=8)
#
# ax2.set_xlabel(r"$z$")
# ax4.set_xlabel(r"$z$")
# ax1.set_ylabel(r"$\rho_{\mathrm{birth}}$ / [cm$^{-3}$]")
# ax2.set_ylabel(r"$\rho_{\mathrm{birth}}$ / [cm$^{-3}$]")
#
# xlims = []
# ylims = []
# for ax in [ax1, ax2, ax3, ax4]:
#     xlims.extend(ax.get_xlim())
#     ylims.extend(ax.get_ylim())
#
# for ax in [ax1, ax2, ax3, ax4]:
#     ax.set_xlim(0, 30)
#     ax.set_ylim(np.min(ylims) - 0.1 * np.min(ylims),
#                 np.max(ylims) + 0.1 * np.max(ylims))
#
# # Remove axis labels
# ax1.tick_params(axis='x', top=False, bottom=False, labeltop=False,
#                 labelbottom=False)
# ax3.tick_params(axis='both', left=False, top=False, right=False, bottom=False,
#                 labelleft=False, labeltop=False,
#                 labelright=False, labelbottom=False)
# ax4.tick_params(axis='y', left=False, right=False, labelleft=False,
#                 labelright=False)
#
# fig.savefig("plots/stellarbd_z_evolution_mass.png", bbox_inches="tight")
#
# plt.close(fig)
#
# fig = plt.figure(figsize=(6, 6))
# gs = gridspec.GridSpec(nrows=2, ncols=2)
# gs.update(wspace=0.0, hspace=0.0)
# ax1 = fig.add_subplot(gs[0, 0])
# ax2 = fig.add_subplot(gs[1, 0])
# ax3 = fig.add_subplot(gs[0, 1])
# ax4 = fig.add_subplot(gs[1, 1])
#
# okinds = np.logical_and(masses > 10**8, masses <= 10**9)
#
# ax1.hexbin(zs[okinds], stellar_bd[okinds], gridsize=100, mincnt=1,
#            yscale="log",  norm=LogNorm(),
#            linewidths=0.2, cmap="Greys")
#
# ax1.text(0.8, 0.9, "$10^8 < M/M_\odot \leq 10^9$",
#         bbox=dict(boxstyle="round,pad=0.3", fc='w', ec="k", lw=1, alpha=0.8),
#         transform=ax1.transAxes, horizontalalignment='right', fontsize=8)
#
# okinds = np.logical_and(masses > 10**9, masses <= 10**9.5)
#
# ax2.hexbin(zs[okinds], stellar_bd[okinds], gridsize=100, mincnt=1,
#            yscale="log",  norm=LogNorm(),
#            linewidths=0.2, cmap="Greys")
#
# ax2.text(0.8, 0.9, "$10^9 < M/M_\odot \leq 10^{9.5}$",
#         bbox=dict(boxstyle="round,pad=0.3", fc='w', ec="k", lw=1, alpha=0.8),
#         transform=ax2.transAxes, horizontalalignment='right', fontsize=8)
#
# okinds = np.logical_and(masses > 10**9.5, masses <= 10**10)
#
# ax3.hexbin(zs[okinds], stellar_bd[okinds], gridsize=100, mincnt=1,
#            yscale="log",  norm=LogNorm(),
#            linewidths=0.2, cmap="Greys")
#
# ax3.text(0.8, 0.9, "$10^{9.5} < M/M_\odot \leq 10^{10}$",
#         bbox=dict(boxstyle="round,pad=0.3", fc='w', ec="k", lw=1, alpha=0.8),
#         transform=ax3.transAxes, horizontalalignment='right', fontsize=8)
#
# okinds = masses > 10**10
#
# ax4.hexbin(zs[okinds], stellar_bd[okinds], gridsize=100, mincnt=1,
#            yscale="log",  norm=LogNorm(),
#            linewidths=0.2, cmap="Greys")
#
# ax4.text(0.8, 0.9, "$10^{10} < M/M_\odot$",
#         bbox=dict(boxstyle="round,pad=0.3", fc='w', ec="k", lw=1, alpha=0.8),
#         transform=ax4.transAxes, horizontalalignment='right', fontsize=8)
#
# ax2.set_xlabel(r"$z$")
# ax4.set_xlabel(r"$z$")
# ax1.set_ylabel(r"$\rho_{\mathrm{birth}}$ / [cm$^{-3}$]")
# ax2.set_ylabel(r"$\rho_{\mathrm{birth}}$ / [cm$^{-3}$]")
#
# xlims = []
# ylims = []
# for ax in [ax1, ax2, ax3, ax4]:
#     xlims.extend(ax.get_xlim())
#     ylims.extend(ax.get_ylim())
#
# for ax in [ax1, ax2, ax3, ax4]:
#     ax.set_xlim(0, 30)
#     ax.set_ylim(np.min(ylims) - 0.1 * np.min(ylims),
#                 np.max(ylims) + 0.1 * np.max(ylims))
#
# # Remove axis labels
# ax1.tick_params(axis='x', top=False, bottom=False, labeltop=False,
#                 labelbottom=False)
# ax3.tick_params(axis='both', left=False, top=False, right=False, bottom=False,
#                 labelleft=False, labeltop=False,
#                 labelright=False, labelbottom=False)
# ax4.tick_params(axis='y', left=False, right=False, labelleft=False,
#                 labelright=False)
#
# fig.savefig("plots/stellarbd_z_evolution_mass_FLARES.png", bbox_inches="tight")
#
# plt.close(fig)
#
# fig = plt.figure()
# ax = fig.add_subplot(111)
#
# okinds = stellar_met_all == 0
#
# ax.hexbin(zs_all[okinds], stellar_bd_all[okinds], gridsize=100, mincnt=1,
#           yscale="log", norm=LogNorm(), linewidths=0.2, cmap="Greys",
#           alpha=0.4)
#
# okinds = agndt9_stellar_met == 0
#
# plot_meidan_stat(np.array(agndt9_zs)[okinds],
#                  np.array(agndt9_stellar_bd)[okinds], ax,
#                  lab="AGNdT9: L0050N0752", color="royalblue", bins=None,
#                  ls="dashdot")
#
# okinds = ref_stellar_met == 0
#
# plot_meidan_stat(np.array(ref_zs)[okinds], np.array(ref_stellar_bd)[okinds],
#                  ax, lab="REFERENCE: L0100N1504", color="limegreen",
#                  bins=None, ls="--")
#
# ax.plot((40, 90), (10**1, 10**3), color="k", linestyle="-", label="FLARES")
#
# for low, up, c in zip(dbinLims[:-1], dbinLims[1:], _cmap.colors):
#
#     print(low, up, c)
#
#     okinds = np.logical_and(ovdens >= low, np.logical_and(ovdens < up,
#                                                           stellar_met == 0))
#
#     plot_meidan_stat(np.array(zs)[okinds], np.array(stellar_bd)[okinds],
#                      ax, lab=None, color=c,
#                      bins=None, ls="-")
#
# ax.set_xlim(-0.1, 27)
#
# sm = plt.cm.ScalarMappable(cmap=_cmap, norm=plt.Normalize(vmin=0., vmax=1.))
# sm._A = []  # # fake up the array of the scalar mappable
# cbaxes = ax.inset_axes([0.7, 0.625, 0.03, 0.35])
# cbar = plt.colorbar(sm, ticks=ticks, cax=cbaxes)
# cbar.ax.set_yticklabels(bin_labels, fontsize=8)
# cbar.ax.set_ylabel("$[\mathrm{log_{10}}(1 \,+\,\delta)] \; "
#                    "(N_{\mathrm{regions}})$", size=9, rotation=90)
#
# ax.set_xlabel("$z$")
# ax.set_ylabel(r"$\rho_{\mathrm{birth}}$ / [cm$^{-3}$]")
#
# handles, labels = ax.get_legend_handles_labels()
# ax.legend(handles, labels, loc="upper left")
#
# ax.set_yscale("log")
#
# fig.savefig("plots/stellarbd_z_evolution_zeromet.png", bbox_inches="tight")
#
# plt.close(fig)
#
# fig = plt.figure()
# ax = fig.add_subplot(111)
#
# okinds = stellar_met_all > 0
#
# ax.hexbin(zs_all[okinds], stellar_bd_all[okinds], gridsize=100, mincnt=1,
#           yscale="log", norm=LogNorm(), linewidths=0.2, cmap="Greys",
#           alpha=0.4)
#
# okinds = agndt9_stellar_met > 0
#
# plot_meidan_stat(np.array(agndt9_zs)[okinds],
#                  np.array(agndt9_stellar_bd)[okinds], ax,
#                  lab="AGNdT9: L0050N0752", color="royalblue", bins=None,
#                  ls="dashdot")
#
# okinds = ref_stellar_met > 0
#
# plot_meidan_stat(np.array(ref_zs)[okinds], np.array(ref_stellar_bd)[okinds],
#                  ax, lab="REFERENCE: L0100N1504", color="limegreen",
#                  bins=None, ls="--")
#
# ax.plot((40, 90), (10**1, 10**3), color="k", linestyle="-", label="FLARES")
#
# for low, up, c in zip(dbinLims[:-1], dbinLims[1:], _cmap.colors):
#
#     print(low, up, c)
#
#     okinds = np.logical_and(ovdens >= low, np.logical_and(ovdens < up,
#                                                           stellar_met > 0))
#
#     plot_meidan_stat(np.array(zs)[okinds], np.array(stellar_bd)[okinds],
#                      ax, lab=None, color=c,
#                      bins=None, ls="-")
#
# ax.set_xlim(-0.1, 27)
#
# sm = plt.cm.ScalarMappable(cmap=_cmap, norm=plt.Normalize(vmin=0., vmax=1.))
# sm._A = []  # # fake up the array of the scalar mappable
# cbaxes = ax.inset_axes([0.7, 0.625, 0.03, 0.35])
# cbar = plt.colorbar(sm, ticks=ticks, cax=cbaxes)
# cbar.ax.set_yticklabels(bin_labels, fontsize=8)
# cbar.ax.set_ylabel("$[\mathrm{log_{10}}(1 \,+\,\delta)] \; "
#                    "(N_{\mathrm{regions}})$", size=9, rotation=90)
#
# ax.set_xlabel("$z$")
# ax.set_ylabel(r"$\rho_{\mathrm{birth}}$ / [cm$^{-3}$]")
#
# handles, labels = ax.get_legend_handles_labels()
# ax.legend(handles, labels)
#
# ax.set_yscale("log")
#
# fig.savefig("plots/stellarbd_z_evolution_nonzeromet.png", bbox_inches="tight")
#
# plt.close(fig)
#
# fig = plt.figure()
# ax = fig.add_subplot(111)
#
# okinds = stellar_met_all > 0.02
#
# ax.hexbin(zs_all[okinds], stellar_bd_all[okinds], gridsize=100, mincnt=1,
#           yscale="log", norm=LogNorm(), linewidths=0.2, cmap="Greys",
#           alpha=0.4)
#
# okinds = agndt9_stellar_met > 0.02
#
# plot_meidan_stat(np.array(agndt9_zs)[okinds],
#                  np.array(agndt9_stellar_bd)[okinds], ax,
#                  lab="AGNdT9: L0050N0752", color="royalblue", bins=None,
#                  ls="dashdot")
#
# okinds = ref_stellar_met > 0.02
#
# plot_meidan_stat(np.array(ref_zs)[okinds], np.array(ref_stellar_bd)[okinds],
#                  ax, lab="REFERENCE: L0100N1504", color="limegreen",
#                  bins=None, ls="--")
#
# ax.plot((40, 90), (10**1, 10**3), color="k", linestyle="-", label="FLARES")
#
# for low, up, c in zip(dbinLims[:-1], dbinLims[1:], _cmap.colors):
#
#     print(low, up, c)
#
#     okinds = np.logical_and(ovdens >= low, np.logical_and(ovdens < up,
#                                                           stellar_met > 0.02))
#
#     plot_meidan_stat(np.array(zs)[okinds], np.array(stellar_bd)[okinds],
#                      ax, lab=None, color=c,
#                      bins=None, ls="-")
#
# ax.set_xlim(-0.1, 27)
#
# sm = plt.cm.ScalarMappable(cmap=_cmap, norm=plt.Normalize(vmin=0., vmax=1.))
# sm._A = []  # # fake up the array of the scalar mappable
# cbaxes = ax.inset_axes([0.7, 0.625, 0.03, 0.35])
# cbar = plt.colorbar(sm, ticks=ticks, cax=cbaxes)
# cbar.ax.set_yticklabels(bin_labels, fontsize=8)
# cbar.ax.set_ylabel("$[\mathrm{log_{10}}(1 \,+\,\delta)] \; "
#                    "(N_{\mathrm{regions}})$", size=9, rotation=90)
#
# ax.set_xlabel("$z$")
# ax.set_ylabel(r"$\rho_{\mathrm{birth}}$ / [cm$^{-3}$]")
#
# handles, labels = ax.get_legend_handles_labels()
# ax.legend(handles, labels)
#
# ax.set_yscale("log")
#
# fig.savefig("plots/stellarbd_z_evolution_0p02.png", bbox_inches="tight")
#
# plt.close(fig)
#
# fig = plt.figure()
# ax = fig.add_subplot(111)
#
# okinds = stellar_met_all > 0.04
#
# ax.hexbin(zs_all[okinds], stellar_bd_all[okinds], gridsize=100, mincnt=1,
#           yscale="log", norm=LogNorm(), linewidths=0.2, cmap="Greys",
#           alpha=0.4)
#
# okinds = agndt9_stellar_met > 0.04
#
# plot_meidan_stat(np.array(agndt9_zs)[okinds],
#                  np.array(agndt9_stellar_bd)[okinds], ax,
#                  lab="AGNdT9: L0050N0752", color="royalblue", bins=None,
#                  ls="dashdot")
#
# okinds = ref_stellar_met > 0.04
#
# plot_meidan_stat(np.array(ref_zs)[okinds], np.array(ref_stellar_bd)[okinds],
#                  ax, lab="REFERENCE: L0100N1504", color="limegreen",
#                  bins=None, ls="--")
#
# ax.plot((40, 90), (10**1, 10**3), color="k", linestyle="-", label="FLARES")
#
# for low, up, c in zip(dbinLims[:-1], dbinLims[1:], _cmap.colors):
#
#     print(low, up, c)
#
#     okinds = np.logical_and(ovdens >= low, np.logical_and(ovdens < up,
#                                                           stellar_met > 0.04))
#
#     plot_meidan_stat(np.array(zs)[okinds], np.array(stellar_bd)[okinds],
#                      ax, lab=None, color=c,
#                      bins=None, ls="-")
#
# ax.set_xlim(-0.1, 27)
#
# sm = plt.cm.ScalarMappable(cmap=_cmap, norm=plt.Normalize(vmin=0., vmax=1.))
# sm._A = []  # # fake up the array of the scalar mappable
# cbaxes = ax.inset_axes([0.7, 0.625, 0.03, 0.35])
# cbar = plt.colorbar(sm, ticks=ticks, cax=cbaxes)
# cbar.ax.set_yticklabels(bin_labels, fontsize=8)
# cbar.ax.set_ylabel("$[\mathrm{log_{10}}(1 \,+\,\delta)] \; "
#                    "(N_{\mathrm{regions}})$", size=9, rotation=90)
#
# ax.set_xlabel("$z$")
# ax.set_ylabel(r"$\rho_{\mathrm{birth}}$ / [cm$^{-3}$]")
#
# handles, labels = ax.get_legend_handles_labels()
# ax.legend(handles, labels)
#
# ax.set_yscale("log")
#
# fig.savefig("plots/stellarbd_z_evolution_0p04.png", bbox_inches="tight")
#
# plt.close(fig)
#
# fig = plt.figure()
# ax = fig.add_subplot(111)
#
# okinds = stellar_met_all > 0.01
#
# ax.hexbin(zs_all[okinds], stellar_bd_all[okinds], gridsize=100, mincnt=1,
#           yscale="log", norm=LogNorm(), linewidths=0.2, cmap="Greys",
#           alpha=0.4)
#
# okinds = agndt9_stellar_met > 0.01
#
# plot_meidan_stat(np.array(agndt9_zs)[okinds],
#                  np.array(agndt9_stellar_bd)[okinds], ax,
#                  lab="AGNdT9: L0050N0752", color="royalblue", bins=None,
#                  ls="dashdot")
#
# okinds = ref_stellar_met > 0.01
#
# plot_meidan_stat(np.array(ref_zs)[okinds], np.array(ref_stellar_bd)[okinds],
#                  ax, lab="REFERENCE: L0100N1504", color="limegreen",
#                  bins=None, ls="--")
#
# ax.plot((40, 90), (10**1, 10**3), color="k", linestyle="-", label="FLARES")
#
# for low, up, c in zip(dbinLims[:-1], dbinLims[1:], _cmap.colors):
#
#     print(low, up, c)
#
#     okinds = np.logical_and(ovdens >= low, np.logical_and(ovdens < up,
#                                                           stellar_met > 0.01))
#
#     plot_meidan_stat(np.array(zs)[okinds], np.array(stellar_bd)[okinds],
#                      ax, lab=None, color=c,
#                      bins=None, ls="-")
#
# ax.set_xlim(-0.1, 27)
#
# sm = plt.cm.ScalarMappable(cmap=_cmap, norm=plt.Normalize(vmin=0., vmax=1.))
# sm._A = []  # # fake up the array of the scalar mappable
# cbaxes = ax.inset_axes([0.7, 0.625, 0.03, 0.35])
# cbar = plt.colorbar(sm, ticks=ticks, cax=cbaxes)
# cbar.ax.set_yticklabels(bin_labels, fontsize=8)
# cbar.ax.set_ylabel("$[\mathrm{log_{10}}(1 \,+\,\delta)] \; "
#                    "(N_{\mathrm{regions}})$", size=9, rotation=90)
#
# ax.set_xlabel("$z$")
# ax.set_ylabel(r"$\rho_{\mathrm{birth}}$ / [cm$^{-3}$]")
#
# handles, labels = ax.get_legend_handles_labels()
# ax.legend(handles, labels)
#
# ax.set_yscale("log")
#
# fig.savefig("plots/stellarbd_z_evolution_0p01.png", bbox_inches="tight")
#
# plt.close(fig)
#
# fig = plt.figure()
# ax = fig.add_subplot(111)
#
# okinds = stellar_met_all > np.median(stellar_met)
#
# ax.hexbin(zs_all[okinds], stellar_bd_all[okinds], gridsize=100, mincnt=1,
#           yscale="log", norm=LogNorm(), linewidths=0.2, cmap="Greys",
#           alpha=0.4)
#
# okinds = agndt9_stellar_met > np.median(stellar_met)
#
# plot_meidan_stat(np.array(agndt9_zs)[okinds],
#                  np.array(agndt9_stellar_bd)[okinds], ax,
#                  lab="AGNdT9: L0050N0752", color="royalblue", bins=None,
#                  ls="dashdot")
#
# okinds = ref_stellar_met > np.median(stellar_met)
#
# plot_meidan_stat(np.array(ref_zs)[okinds], np.array(ref_stellar_bd)[okinds],
#                  ax, lab="REFERENCE: L0100N1504", color="limegreen",
#                  bins=None, ls="--")
#
# ax.plot((40, 90), (10**1, 10**3), color="k", linestyle="-", label="FLARES")
#
# for low, up, c in zip(dbinLims[:-1], dbinLims[1:], _cmap.colors):
#
#     print(low, up, c)
#
#     okinds = np.logical_and(ovdens >= low, np.logical_and(ovdens < up,
#                                                           stellar_met >
#                                                           np.median(stellar_met)))
#
#     plot_meidan_stat(np.array(zs)[okinds], np.array(stellar_bd)[okinds],
#                      ax, lab=None, color=c,
#                      bins=None, ls="-")
#
# ax.set_xlim(-0.1, 27)
#
# sm = plt.cm.ScalarMappable(cmap=_cmap, norm=plt.Normalize(vmin=0., vmax=1.))
# sm._A = []  # # fake up the array of the scalar mappable
# cbaxes = ax.inset_axes([0.7, 0.625, 0.03, 0.35])
# cbar = plt.colorbar(sm, ticks=ticks, cax=cbaxes)
# cbar.ax.set_yticklabels(bin_labels, fontsize=8)
# cbar.ax.set_ylabel("$[\mathrm{log_{10}}(1 \,+\,\delta)] \; "
#                    "(N_{\mathrm{regions}})$", size=9, rotation=90)
#
# ax.set_xlabel("$z$")
# ax.set_ylabel(r"$\rho_{\mathrm{birth}}$ / [cm$^{-3}$]")
#
# handles, labels = ax.get_legend_handles_labels()
# ax.legend(handles, labels)
#
# ax.set_yscale("log")
#
# fig.savefig("plots/stellarbd_z_evolution_med.png", bbox_inches="tight")
#
# plt.close(fig)
#
# fig = plt.figure(figsize=(6, 6))
# gs = gridspec.GridSpec(nrows=2, ncols=2)
# gs.update(wspace=0.0, hspace=0.0)
# ax1 = fig.add_subplot(gs[0, 0])
# ax2 = fig.add_subplot(gs[1, 0])
# ax3 = fig.add_subplot(gs[0, 1])
# ax4 = fig.add_subplot(gs[1, 1])
#
# ax1.hexbin(zs_all, stellar_bd_all, gridsize=100, mincnt=1, yscale="log",
#            norm=LogNorm(), linewidths=0.2, cmap="Greys", alpha=0.8)
#
# ax1.text(0.8, 0.9, "All",
#         bbox=dict(boxstyle="round,pad=0.3", fc='w', ec="k", lw=1, alpha=0.8),
#         transform=ax1.transAxes, horizontalalignment='right', fontsize=8)
#
# ax2.hexbin(zs, stellar_bd, gridsize=100, mincnt=1, yscale="log",
#           norm=LogNorm(), linewidths=0.2, cmap="Greys", alpha=0.8)
#
# ax2.text(0.8, 0.9, "FLARES",
#         bbox=dict(boxstyle="round,pad=0.3", fc='w', ec="k", lw=1, alpha=0.8),
#         transform=ax2.transAxes, horizontalalignment='right', fontsize=8)
#
# ax3.hexbin(agndt9_zs, agndt9_stellar_bd, gridsize=100, mincnt=1, yscale="log",
#           norm=LogNorm(), linewidths=0.2, cmap="Greys", alpha=0.8)
#
# ax3.text(0.8, 0.9, "EAGLE-AGNdT9",
#         bbox=dict(boxstyle="round,pad=0.3", fc='w', ec="k", lw=1, alpha=0.8),
#         transform=ax3.transAxes, horizontalalignment='right', fontsize=8)
#
# ax4.hexbin(ref_zs, ref_stellar_bd, gridsize=100, mincnt=1, yscale="log",
#           norm=LogNorm(), linewidths=0.2, cmap="Greys", alpha=0.8)
#
# ax4.text(0.8, 0.9, "EAGLE-REF",
#         bbox=dict(boxstyle="round,pad=0.3", fc='w', ec="k", lw=1, alpha=0.8),
#         transform=ax4.transAxes, horizontalalignment='right', fontsize=8)
#
# ax2.set_xlabel(r"$z$")
# ax4.set_xlabel(r"$z$")
# ax1.set_ylabel(r"$\rho_{\mathrm{birth}}$ / [cm$^{-3}$]")
# ax2.set_ylabel(r"$\rho_{\mathrm{birth}}$ / [cm$^{-3}$]")
#
# xlims = []
# ylims = []
# for ax in [ax1, ax2, ax3, ax4]:
#     xlims.extend(ax.get_xlim())
#     ylims.extend(ax.get_ylim())
#
# for ax in [ax1, ax2, ax3, ax4]:
#     ax.set_xlim(0, 30)
#     ax.set_ylim(np.min(ylims) - 0.1 * np.min(ylims),
#                 np.max(ylims) + 0.1 * np.max(ylims))
#
# # Remove axis labels
# ax1.tick_params(axis='x', top=False, bottom=False, labeltop=False,
#                 labelbottom=False)
# ax3.tick_params(axis='both', left=False, top=False, right=False, bottom=False,
#                 labelleft=False, labeltop=False,
#                 labelright=False, labelbottom=False)
# ax4.tick_params(axis='y', left=False, right=False, labelleft=False,
#                 labelright=False)
#
# fig.savefig("plots/stellarbd_evolution_split.png", bbox_inches="tight")
#
# plt.close(fig)
#
# fig = plt.figure()
# ax = fig.add_subplot(111)
#
# ax.hexbin(zs_all, fth_all, gridsize=100, mincnt=1,
#           norm=LogNorm(), linewidths=0.2, cmap="Greys", alpha=0.4)
#
# plot_meidan_stat(np.array(agndt9_zs), np.array(agndt9_fth), ax,
#                  lab="AGNdT9: L0050N0752", color="royalblue", bins=None,
#                  ls="dashdot")
# plot_meidan_stat(np.array(ref_zs), np.array(ref_fth),
#                  ax, lab="REFERENCE: L0100N1504", color="limegreen",
#                  bins=None, ls="--")
#
# ax.plot((40, 90), (1, 1.1), color="k", linestyle="-", label="FLARES")
#
# for low, up, c in zip(dbinLims[:-1], dbinLims[1:], _cmap.colors):
#
#     print(low, up, c)
#
#     okinds = np.logical_and(ovdens >= low, ovdens < up)
#
#     plot_meidan_stat(np.array(zs)[okinds], np.array(fth)[okinds],
#                      ax, lab=None, color=c,
#                      bins=None, ls="-")
#
# ax.set_xlim(-0.1, 27)
#
# sm = plt.cm.ScalarMappable(cmap=_cmap, norm=plt.Normalize(vmin=0., vmax=1.))
# sm._A = []  # # fake up the array of the scalar mappable
# cbaxes = ax.inset_axes([0.7, 0.625, 0.03, 0.35])
# cbar = plt.colorbar(sm, ticks=ticks, cax=cbaxes)
# cbar.ax.set_yticklabels(bin_labels, fontsize=8)
# cbar.ax.set_ylabel("$[\mathrm{log_{10}}(1 \,+\,\delta)] \; "
#                    "(N_{\mathrm{regions}})$", size=9, rotation=90)
#
# ax.set_xlabel("$z$")
# ax.set_ylabel(r"$\rho_{\mathrm{birth}}$ / [cm$^{-3}$]")
#
# handles, labels = ax.get_legend_handles_labels()
# ax.legend(handles, labels, loc="lower right")
#
# fig.savefig("plots/stellarfth_z_evolution_od.png", bbox_inches="tight")
#
# plt.close(fig)
#
# fig = plt.figure()
# ax = fig.add_subplot(111)
#
# im = ax.hexbin(zs_all, stellar_bd_all, C=stellar_met_all, gridsize=100,
#                mincnt=1, yscale="log", reduce_C_function=np.mean,
#                linewidths=0.2, cmap="plasma")
#
# ax.set_xlabel("$z$")
# ax.set_ylabel(r"$\rho_{\mathrm{birth}}$ / [cm$^{-3}$]")
#
# cbar = plt.colorbar(im)
# cbar.ax.set_ylabel("$Z$")
#
# fig.savefig("plots/stellarbdmet_z_evolution.png", bbox_inches="tight")
#
# plt.close(fig)
#
# fig = plt.figure()
# ax = fig.add_subplot(111)
#
# im = ax.hexbin(zs_all, stellar_bd_all, C=ovdens_all, gridsize=100,
#                mincnt=1, yscale="log", reduce_C_function=np.mean,
#                linewidths=0.2, cmap="plasma")
#
# ax.set_xlabel("$z$")
# ax.set_ylabel(r"$\rho_{\mathrm{birth}}$ / [cm$^{-3}$]")
#
# cbar = plt.colorbar(im)
# cbar.ax.set_ylabel("$\Delta$")
#
# fig.savefig("plots/stellarbdovden_z_evolution.png", bbox_inches="tight")
#
# plt.close(fig)
#
# fig = plt.figure()
# ax = fig.add_subplot(111)
#
# plot_meidan_stat(np.array(agndt9_zs), np.array(agndt9_stellar_bd), ax,
#                  lab="AGNdT9: L0050N0752", color="royalblue", bins=None,
#                  ls="dashdot")
# plot_meidan_stat(np.array(ref_zs), np.array(ref_stellar_bd),
#                  ax, lab="REFERENCE: L0100N1504", color="limegreen",
#                  bins=None, ls="--")
#
# ax.plot((40, 90), (10**1, 10**3), color="k", linestyle="-", label="FLARES")
#
# for low, up, c in zip(dbinLims[:-1], dbinLims[1:], _cmap.colors):
#
#     print(low, up, c)
#
#     okinds = np.logical_and(ovdens >= low, ovdens < up)
#
#     plot_meidan_stat(np.array(zs)[okinds], np.array(stellar_bd)[okinds],
#                      ax, lab=None, color=c,
#                      bins=None, ls="-")
#
# ax.set_xlim(None, 32)
#
# sm = plt.cm.ScalarMappable(cmap=_cmap, norm=plt.Normalize(vmin=0., vmax=1.))
# sm._A = []  # # fake up the array of the scalar mappable
# cbaxes = ax.inset_axes([0.7, 0.625, 0.03, 0.35])
# cbar = plt.colorbar(sm, ticks=ticks, cax=cbaxes)
# cbar.ax.set_yticklabels(bin_labels, fontsize=8)
# cbar.ax.set_ylabel("$[\mathrm{log_{10}}(1 \,+\,\delta)] \; "
#                    "(N_{\mathrm{regions}})$", size=9, rotation=90)
#
# ax.set_xlabel("$z$")
# ax.set_ylabel(r"$\rho_{\mathrm{birth}}$ / [cm$^{-3}$]")
#
# handles, labels = ax.get_legend_handles_labels()
# ax.legend(handles, labels, loc="lower right")
#
# ax.set_yscale("log")
#
# fig.savefig("plots/stellarbd_z_evolution_od_nohex.png", bbox_inches="tight")
#
# plt.close(fig)
#
# fig = plt.figure(figsize=(6, 6))
# gs = gridspec.GridSpec(nrows=2, ncols=2)
# gs.update(wspace=0.0, hspace=0.0)
# ax1 = fig.add_subplot(gs[0, 0])
# ax2 = fig.add_subplot(gs[1, 0])
# ax3 = fig.add_subplot(gs[0, 1])
# ax4 = fig.add_subplot(gs[1, 1])
#
# ax1.hexbin(stellar_bd_all, stellar_met_all, gridsize=100, mincnt=1,
#            xscale="log", norm=LogNorm(), linewidths=0.2, cmap="Greys")
#
# ax1.text(0.8, 0.9, "All",
#         bbox=dict(boxstyle="round,pad=0.3", fc='w', ec="k", lw=1, alpha=0.8),
#         transform=ax1.transAxes, horizontalalignment='right', fontsize=8)
#
# ax2.hexbin(stellar_bd, stellar_met, gridsize=100, mincnt=1,
#            xscale="log", norm=LogNorm(), linewidths=0.2, cmap="Greys")
#
# ax2.text(0.8, 0.9, "FLARES",
#         bbox=dict(boxstyle="round,pad=0.3", fc='w', ec="k", lw=1, alpha=0.8),
#         transform=ax2.transAxes, horizontalalignment='right', fontsize=8)
#
# ax3.hexbin(agndt9_stellar_bd, agndt9_stellar_met, gridsize=100, mincnt=1,
#            xscale="log", norm=LogNorm(), linewidths=0.2, cmap="Greys")
#
# ax3.text(0.8, 0.9, "EAGLE-AGNdT9",
#         bbox=dict(boxstyle="round,pad=0.3", fc='w', ec="k", lw=1, alpha=0.8),
#         transform=ax3.transAxes, horizontalalignment='right', fontsize=8)
#
# ax4.hexbin(ref_stellar_bd, ref_stellar_met, gridsize=100, mincnt=1,
#            xscale="log", norm=LogNorm(), linewidths=0.2, cmap="Greys")
#
# ax4.text(0.8, 0.9, "EAGLE-REF",
#         bbox=dict(boxstyle="round,pad=0.3", fc='w', ec="k", lw=1, alpha=0.8),
#         transform=ax4.transAxes, horizontalalignment='right', fontsize=8)
#
# ax2.set_xlabel(r"$\rho_{\mathrm{birth}}$ / [cm$^{-3}$]")
# ax4.set_xlabel(r"$\rho_{\mathrm{birth}}$ / [cm$^{-3}$]")
# ax1.set_ylabel(r"$Z$")
# ax2.set_ylabel(r"$Z$")
#
# xlims = []
# ylims = []
# for ax in [ax1, ax2, ax3, ax4]:
#     xlims.extend(ax.get_xlim())
#     ylims.extend(ax.get_ylim())
#
# for ax in [ax1, ax2, ax3, ax4]:
#     ax.set_xlim(np.min(xlims) - 0.1 * np.min(xlims),
#                 np.max(xlims) + 0.1 * np.max(xlims))
#     ax.set_ylim(np.min(ylims) - 0.1 * np.min(ylims),
#                 np.max(ylims) + 0.1 * np.max(ylims))
#
#
# # Remove axis labels
# ax1.tick_params(axis='x', top=False, bottom=False, labeltop=False,
#                 labelbottom=False)
# ax3.tick_params(axis='both', left=False, top=False, right=False, bottom=False,
#                 labelleft=False, labeltop=False,
#                 labelright=False, labelbottom=False)
# ax4.tick_params(axis='y', left=False, right=False, labelleft=False,
#                 labelright=False)
#
# fig.savefig("plots/stellarbd_vs_stellarz.png", bbox_inches="tight")
#
# plt.close(fig)
#
# fig = plt.figure()
# ax = fig.add_subplot(111)
#
# ax.hexbin(zs_all, stellar_met_all, gridsize=100, mincnt=1,
#           norm=LogNorm(), linewidths=0.2, cmap="Greys", alpha=0.4)
#
# plot_meidan_stat(np.array(agndt9_zs), np.array(agndt9_stellar_met), ax,
#                  lab="AGNdT9: L0050N0752", color="royalblue", bins=None,
#                  ls="dashdot")
# plot_meidan_stat(np.array(ref_zs), np.array(ref_stellar_met),
#                  ax, lab="REFERENCE: L0100N1504", color="limegreen",
#                  bins=None, ls="--")
#
# ax.plot((40, 90), (0.0001, 0.00002), color="k", linestyle="-", label="FLARES")
#
# for low, up, c in zip(dbinLims[:-1], dbinLims[1:], _cmap.colors):
#
#     okinds = np.logical_and(ovdens >= low, ovdens < up)
#
#     plot_meidan_stat(np.array(zs)[okinds],  np.array(stellar_met)[okinds],
#                      ax, lab=None, color=c,
#                      bins=None, ls="-")
#
# ax.set_xlim(None, 27)
#
# sm = plt.cm.ScalarMappable(cmap=_cmap, norm=plt.Normalize(vmin=0., vmax=1.))
# sm._A = []  # # fake up the array of the scalar mappable
# cbaxes = ax.inset_axes([0.7, 0.625, 0.03, 0.35])
# cbar = plt.colorbar(sm, ticks=ticks, cax=cbaxes)
# cbar.ax.set_yticklabels(bin_labels, fontsize=8)
# cbar.ax.set_ylabel("$[\mathrm{log_{10}}(1 \,+\,\delta)] \; "
#                    "(N_{\mathrm{regions}})$", size=9, rotation=90)
#
# ax.set_xlabel("$z$")
# ax.set_ylabel(r"$Z$")
#
# handles, labels = ax.get_legend_handles_labels()
# ax.legend(handles, labels, loc="upper left")
#
# fig.savefig("plots/stellarmet_z_evolution_od.png", bbox_inches="tight")
#
# plt.close(fig)
#
# fig = plt.figure()
# ax = fig.add_subplot(111)
#
# plot_meidan_stat(np.array(agndt9_zs), np.array(agndt9_stellar_met), ax,
#                  lab="AGNdT9: L0050N0752", color="royalblue", bins=None,
#                  ls="dashdot")
# plot_meidan_stat(np.array(ref_zs), np.array(ref_stellar_met),
#                  ax, lab="REFERENCE: L0100N1504", color="limegreen",
#                  bins=None, ls="--")
#
# ax.plot((40, 90), (0.0001, 0.00002), color="k", linestyle="-", label="FLARES")
#
# for low, up, c in zip(dbinLims[:-1], dbinLims[1:], _cmap.colors):
#
#     okinds = np.logical_and(ovdens >= low, ovdens < up)
#
#     plot_meidan_stat(np.array(zs)[okinds], np.array(stellar_met)[okinds],
#                      ax, lab=None, color=c,
#                      bins=None, ls="-")
#
# ax.set_xlim(None, 27)
#
# sm = plt.cm.ScalarMappable(cmap=_cmap, norm=plt.Normalize(vmin=0., vmax=1.))
# sm._A = []  # # fake up the array of the scalar mappable
# cbaxes = ax.inset_axes([0.7, 0.625, 0.03, 0.35])
# cbar = plt.colorbar(sm, ticks=ticks, cax=cbaxes)
# cbar.ax.set_yticklabels(bin_labels, fontsize=8)
# cbar.ax.set_ylabel("$[\mathrm{log_{10}}(1 \,+\,\delta)] \; "
#                    "(N_{\mathrm{regions}})$", size=9, rotation=90)
#
# ax.set_xlabel("$z$")
# ax.set_ylabel(r"$Z$")
#
# handles, labels = ax.get_legend_handles_labels()
# ax.legend(handles, labels, loc="upper left")
#
# fig.savefig("plots/stellarmet_z_evolution_od_nohex.png", bbox_inches="tight")


# ============================ Evolution Gradients ============================

bins_age = np.arange(cosmo.age(27.5).value, cosmo.age(0).value,
                     (1000 * u.Myr).to(u.Gyr).value)
print(bins_age)
bins = np.array([z_at_value(lambda x: cosmo.age(x).value,
                            a, zmin=0, zmax=28) for a in bins_age])
bins[-1] = 0
print(bins)

okinds = np.logical_and(~np.isnan(zs_all),
                        np.logical_and(~np.isnan(stellar_bd_all),
                                       ~np.isnan(stellar_met_all)))
zs_all = zs_all[okinds]
stellar_bd_all = stellar_bd_all[okinds]
stellar_met_all = stellar_met_all[okinds]

print(np.min(zs_all), np.max(zs_all))
print(np.min(stellar_bd_all), np.max(stellar_bd_all))
print(np.min(stellar_met_all), np.max(stellar_met_all))

# Compute binned statistics
stellar_bd_binned, binedges, bin_ind = binned_statistic(zs_all, stellar_bd_all,
                                                        statistic="median",
                                                        bins=bins)
stellar_met_binned, binedges, bin_ind = binned_statistic(zs_all, stellar_met_all,
                                                        statistic="median",
                                                        bins=bins)

bin_width = bins[1] - bins[0]
bin_cents = bins[1:] - (bin_width / 2)
# bin_cents_z = np.array([z_at_value(cosmo.age, a, zmin=0, zmax=28)
#                         for a in bin_cents])

growth = lambda x1, x2, t1, t2: np.arctan((t2 + t1) * (x2 - x1) 
                                          / ((t2 - t1) * (x2 + x1))) \
                                / (np.pi / 2)

met_grad = []
bd_grad = []
plt_zs = []
for i in range(len(stellar_bd_binned) - 1):
    bd_grad.append(growth(stellar_bd_binned[i], stellar_bd_binned[i + 1],
                          cosmo.age(bin_cents[i]).value, cosmo.age(bin_cents[i + 1]).value))
    met_grad.append(growth(stellar_met_binned[i], stellar_met_binned[i + 1],
                           cosmo.age(bin_cents[i]).value, cosmo.age(bin_cents[i + 1]).value))
    plt_zs.append(bin_cents[i + 1] - (bin_width / 2))

met_grad = np.array(met_grad)
bd_grad = np.array(bd_grad)
plt_zs = np.array(plt_zs)

fig = plt.figure()
ax = fig.add_subplot(111)

ax.plot(plt_zs, bd_grad, linestyle="-", color="r", label="Birth Density")
ax.plot(plt_zs, met_grad, linestyle="-", color="r", label="Birth Metallicity")

ax.set_xlabel("$z$")
ax.set_ylabel(r"$\beta$")

fig.savefig("plots/stellarbdmet_z_evolution_gradient.png", bbox_inches="tight")

