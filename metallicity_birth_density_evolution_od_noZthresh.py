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
    fths = []
    masses = []
    
    if eagle or ref:
        ovds = [1, ]
    else:
        ovds = np.loadtxt("region_overdensity.txt", dtype=float)

    for reg, ovd in zip(regions, ovds):

        for snap in snaps:
            
            if eagle:
                path = "/cosma7/data/Eagle/ScienceRuns/Planck1/L0025N0376/" \
                       "PE/EagleVariation_NoZDEPSFthresh"

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


stellar_bd_all, stellar_met_all, zs_all, ovdens_all, fth_all, masses_all = get_data(eagle=True)

fig = plt.figure()
ax = fig.add_subplot(111)

ax.hexbin(zs_all, stellar_bd_all, gridsize=100, mincnt=1, yscale="log", 
          norm=LogNorm(), linewidths=0.2, cmap="Greys", alpha=0.4)

plot_meidan_stat(np.array(zs_all), np.array(stellar_bd_all), ax,
                 lab="NoZDEPSFthresh: L0025N0376", color="royalblue", bins=None,
                 ls="dashdot")

ax.set_xlabel("$z$")
ax.set_ylabel(r"$\rho_{\mathrm{birth}}$ / [cm$^{-3}$]")

handles, labels = ax.get_legend_handles_labels()
ax.legend(handles, labels, loc="lower right")

ax.set_yscale("log")

fig.savefig("plots/stellarbd_z_evolution_od_noZthresh.png", bbox_inches="tight")

plt.close(fig)

fig = plt.figure(figsize=(6, 6))
gs = gridspec.GridSpec(nrows=2, ncols=2)
gs.update(wspace=0.0, hspace=0.0)
ax1 = fig.add_subplot(gs[0, 0])
ax2 = fig.add_subplot(gs[1, 0])
ax3 = fig.add_subplot(gs[0, 1])
ax4 = fig.add_subplot(gs[1, 1])

okinds = np.logical_and(masses_all > 10**8, masses_all <= 10**9)

ax1.hexbin(zs_all[okinds], stellar_bd_all[okinds], gridsize=100, mincnt=1,
           yscale="log",  norm=LogNorm(),
           linewidths=0.2, cmap="Greys")

ax1.text(0.8, 0.9, "$10^8 < M/M_\odot \leq 10^9$",
        bbox=dict(boxstyle="round,pad=0.3", fc='w', ec="k", lw=1, alpha=0.8),
        transform=ax1.transAxes, horizontalalignment='right', fontsize=8)

okinds = np.logical_and(masses_all > 10**9, masses_all <= 10**9.5)

ax2.hexbin(zs_all[okinds], stellar_bd_all[okinds], gridsize=100, mincnt=1,
           yscale="log",  norm=LogNorm(),
           linewidths=0.2, cmap="Greys")

ax2.text(0.8, 0.9, "$10^9 < M/M_\odot \leq 10^{9.5}$",
        bbox=dict(boxstyle="round,pad=0.3", fc='w', ec="k", lw=1, alpha=0.8),
        transform=ax2.transAxes, horizontalalignment='right', fontsize=8)

okinds = np.logical_and(masses_all > 10**9.5, masses_all <= 10**10)

ax3.hexbin(zs_all[okinds], stellar_bd_all[okinds], gridsize=100, mincnt=1,
           yscale="log",  norm=LogNorm(),
           linewidths=0.2, cmap="Greys")

ax3.text(0.8, 0.9, "$10^{9.5} < M/M_\odot \leq 10^{10}$",
        bbox=dict(boxstyle="round,pad=0.3", fc='w', ec="k", lw=1, alpha=0.8),
        transform=ax3.transAxes, horizontalalignment='right', fontsize=8)

okinds = masses_all > 10**10

ax4.hexbin(zs_all[okinds], stellar_bd_all[okinds], gridsize=100, mincnt=1,
           yscale="log",  norm=LogNorm(),
           linewidths=0.2, cmap="Greys")

ax4.text(0.8, 0.9, "$10^{10} < M/M_\odot$",
        bbox=dict(boxstyle="round,pad=0.3", fc='w', ec="k", lw=1, alpha=0.8),
        transform=ax4.transAxes, horizontalalignment='right', fontsize=8)

ax2.set_xlabel(r"$z$")
ax4.set_xlabel(r"$z$")
ax1.set_ylabel(r"$\rho_{\mathrm{birth}}$ / [cm$^{-3}$]")
ax2.set_ylabel(r"$\rho_{\mathrm{birth}}$ / [cm$^{-3}$]")

xlims = []
ylims = []
for ax in [ax1, ax2, ax3, ax4]:
    xlims.extend(ax.get_xlim())
    ylims.extend(ax.get_ylim())

for ax in [ax1, ax2, ax3, ax4]:
    ax.set_xlim(0, 30)
    ax.set_ylim(np.min(ylims) - 0.1 * np.min(ylims),
                np.max(ylims) + 0.1 * np.max(ylims))

# Remove axis labels
ax1.tick_params(axis='x', top=False, bottom=False, labeltop=False,
                labelbottom=False)
ax3.tick_params(axis='both', left=False, top=False, right=False, bottom=False,
                labelleft=False, labeltop=False,
                labelright=False, labelbottom=False)
ax4.tick_params(axis='y', left=False, right=False, labelleft=False,
                labelright=False)

fig.savefig("plots/stellarbd_z_evolution_mass_noZthresh.png", bbox_inches="tight")

plt.close(fig)

fig = plt.figure()
ax = fig.add_subplot(111)

okinds = stellar_met_all == 0

ax.hexbin(zs_all[okinds], stellar_bd_all[okinds], gridsize=100, mincnt=1, yscale="log",
          norm=LogNorm(), linewidths=0.2, cmap="Greys", alpha=0.4)

plot_meidan_stat(np.array(zs_all)[okinds], np.array(stellar_bd_all)[okinds], ax,
                 lab="NoZDEPSFthresh: L0025N0376", color="royalblue", bins=None,
                 ls="dashdot")

ax.set_xlabel("$z$")
ax.set_ylabel(r"$\rho_{\mathrm{birth}}$ / [cm$^{-3}$]")

handles, labels = ax.get_legend_handles_labels()
ax.legend(handles, labels, loc="lower right")

ax.set_yscale("log")

fig.savefig("plots/stellarbd_z_evolution_zeromet_noZthresh.png", bbox_inches="tight")

plt.close(fig)

fig = plt.figure()
ax = fig.add_subplot(111)

okinds = stellar_met_all > 0.04

ax.hexbin(zs_all[okinds], stellar_bd_all[okinds], gridsize=100, mincnt=1, yscale="log",
          norm=LogNorm(), linewidths=0.2, cmap="Greys", alpha=0.4)

plot_meidan_stat(np.array(zs_all)[okinds], np.array(stellar_bd_all)[okinds], ax,
                 lab="NoZDEPSFthresh: L0025N0376", color="royalblue", bins=None,
                 ls="dashdot")

ax.set_xlabel("$z$")
ax.set_ylabel(r"$\rho_{\mathrm{birth}}$ / [cm$^{-3}$]")

handles, labels = ax.get_legend_handles_labels()
ax.legend(handles, labels, loc="lower right")

ax.set_yscale("log")

fig.savefig("plots/stellarbd_z_evolution_0p04_noZthresh.png", bbox_inches="tight")

plt.close(fig)

fig = plt.figure()
ax = fig.add_subplot(111)

okinds = stellar_met_all > 0.02

ax.hexbin(zs_all[okinds], stellar_bd_all[okinds], gridsize=100, mincnt=1, yscale="log",
          norm=LogNorm(), linewidths=0.2, cmap="Greys", alpha=0.4)

plot_meidan_stat(np.array(zs_all)[okinds], np.array(stellar_bd_all)[okinds], ax,
                 lab="NoZDEPSFthresh: L0025N0376", color="royalblue", bins=None,
                 ls="dashdot")

ax.set_xlabel("$z$")
ax.set_ylabel(r"$\rho_{\mathrm{birth}}$ / [cm$^{-3}$]")

handles, labels = ax.get_legend_handles_labels()
ax.legend(handles, labels, loc="lower right")

ax.set_yscale("log")

fig.savefig("plots/stellarbd_z_evolution_0p02_noZthresh.png", bbox_inches="tight")

plt.close(fig)

fig = plt.figure()
ax = fig.add_subplot(111)

ax.hexbin(zs_all, fth_all, gridsize=100, mincnt=1, yscale="log", 
          norm=LogNorm(), linewidths=0.2, cmap="Greys", alpha=0.4)

plot_meidan_stat(np.array(zs_all), np.array(fth_all), ax,
                 lab="NoZDEPSFthresh: L0025N0376", color="royalblue", bins=None,
                 ls="dashdot")

ax.set_xlabel("$z$")
ax.set_ylabel(r"$f_{th}$")

handles, labels = ax.get_legend_handles_labels()
ax.legend(handles, labels, loc="lower right")

ax.set_yscale("log")

fig.savefig("plots/stellarfth_z_evolution_od_noZthresh.png", bbox_inches="tight")

plt.close(fig)

fig = plt.figure()
ax = fig.add_subplot(111)

ax.hexbin(zs_all, stellar_met_all, gridsize=100, mincnt=1,
          norm=LogNorm(), linewidths=0.2, cmap="Greys", alpha=0.4)

plot_meidan_stat(np.array(zs_all), np.array(stellar_met_all), ax,
                 lab="NoZDEPSFthresh: L0025N0376", color="royalblue", bins=None,
                 ls="dashdot")

ax.set_xlabel("$z$")
ax.set_ylabel(r"$Z$")

handles, labels = ax.get_legend_handles_labels()
ax.legend(handles, labels, loc="lower right")

ax.set_yscale("log")

fig.savefig("plots/stellarmet_z_evolution_od_noZthresh.png", bbox_inches="tight")

plt.close(fig)
