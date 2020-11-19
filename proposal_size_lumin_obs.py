import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.lines import Line2D
from astropy.cosmology import Planck13 as cosmo
from scipy.stats import binned_statistic
import astropy.units as u
import FLARE.plt
import h5py
import FLARE.photom as photconv
import seaborn as sns


def m_to_M(m, cosmo, z):
    flux = photconv.m_to_flux(m)
    lum = photconv.flux_to_L(flux, cosmo, z)
    M = photconv.lum_to_M(lum)
    return M

def M_to_m(M, cosmo, z):
    lum = photconv.M_to_lum(M)
    flux = photconv.lum_to_flux(lum, cosmo, z)
    m = photconv.flux_to_m(flux)
    return m


def plot_meidan_stat(xs, ys, ax, lab, color, bins=None, ls='-'):

    # Compute binned statistics
    y_stat, binedges, bin_ind = binned_statistic(xs, ys, statistic='median',
                                                 bins=bins)

    # Compute bincentres
    bin_wid = binedges[1] - binedges[0]
    bin_cents = binedges[1:] - bin_wid / 2

    okinds = np.logical_and(~np.isnan(bin_cents), ~np.isnan(y_stat))

    ax.plot(bin_cents[okinds], y_stat[okinds], color=color, linestyle=ls,
            label=lab)


# Define Kawamata17 fit and parameters
kawa_params = {'beta': {6: 0.46, 7: 0.46, 8: 0.38, 9: 0.56},
               'r_0': {6: 0.94, 7: 0.94, 8: 0.81, 9: 1.2}}
kawa_up_params = {'beta': {6: 0.46 + 0.08, 7: 0.46 + 0.08,
                           8: 0.38 + 0.28, 9: 0.56 + 1.01},
                  'r_0': {6: 0.94 + 0.2, 7: 0.94 + 0.2,
                          8: 0.81 + 5.28, 9: 1.2 + 367.64}}
kawa_low_params = {'beta': {6: 0.46 - 0.09, 7: 0.46 - 0.09,
                            8: 0.38 - 0.78, 9: 0.56 - 0.27},
                   'r_0': {6: 0.94 - 0.15, 7: 0.94 - 0.15,
                           8: 0.81 - 0.26, 9: 1.2 - 0.74}}
kawa_fit = lambda l, r0, b: r0 * (l / photconv.M_to_lum(-21)) ** b

# snaps = ['003_z012p000', '004_z011p000',
#          '005_z010p000', '006_z009p000']
#
# f = 'FAKE.TH.FUV'
#
# Type = "Total"
# orientation = "sim"
#
# # Get FLARES results
# regions = []
# for reg in range(0, 40):
#     if reg < 10:
#         regions.append('0' + str(reg))
#     else:
#         regions.append(str(reg))
#
# reg_snaps = []
# for reg in reversed(regions):
#
#     for snap in snaps:
#         reg_snaps.append((reg, snap))
#
# # hlr_dict = {}
# # hlr_app_dict = {}
# hlr = []
# lumin = []
# mass = []
# F_zs = []
#
# for reg, snap in reg_snaps:
#
#     z_str = snap.split('z')[1].split('p')
#     z = float(z_str[0] + '.' + z_str[1])
#
#     hdf = h5py.File("../flares-sizes-obs/data/flares_sizes_{}_{}.hdf5".format(reg, snap), "r")
#     type_group = hdf[Type]
#     orientation_group = type_group[orientation]
#
#     # hlr_dict[z].extend(orientation_group[f]["HLR_0.5"][...])
#     # hlr_app_dict[z].extend(
#     #     orientation_group[f]["HLR_Aperture_0.5"][...])
#     hlr.extend( orientation_group[f]["HLR_0.5"][...])
#     lumin.extend(orientation_group[f]["Luminosity"][...])
#     mass.extend(orientation_group[f]["Mass"][...])
#     F_zs.extend(np.full(orientation_group[f]["Mass"][...].size, z))
#     hdf.close()
#
# hlr = np.array(hlr)
# lumin = np.array(lumin)
# mass = np.array(mass)
# F_zs = np.array(F_zs)
#
# okinds = mass > 10**9
#
# print("There are", len(F_zs), "FLARES galaxies")
#
# hlr = hlr[okinds]
# lumin = lumin[okinds]
# mass = mass[okinds]
# F_zs = F_zs[okinds]

df = pd.read_csv("HighzSizes/All.csv")

papers = df["Paper"].values
mags = df["Magnitude"].values
r_es_arcs = df["R_e"].values
r_es_type = df["R_e (Unit)"].values
mag_type = df["Magnitude Type"].values
zs = df["Redshift"].values

okinds = zs >= 7
papers = papers[okinds]
mags = mags[okinds]
r_es_arcs = r_es_arcs[okinds]
r_es_type = r_es_type[okinds]
mag_type = mag_type[okinds]
zs = zs[okinds]

# Define pixel resolutions
wfc3 = 0.13
nircam_short = 0.031
nircam_long = 0.063

# Convert to physical kiloparsecs
r_es = np.zeros(len(papers))
for (ind, r), z in zip(enumerate(r_es_arcs), zs):
    if r_es_type[ind] == "kpc":
        r_es[ind] = r
    else:
        r_es[ind] = r / cosmo.arcsec_per_kpc_proper(z).value
    if mags[ind] < 0:
        mags[ind] = M_to_m(mags[ind], cosmo, z)

labels = {"G11": "Grazian+2011 (z=7)",
          "G12": "Grazian+2012 (z=7)",
          "C16": "Calvi+2016 (z=10-10.5)",
          "K18": "Kawamata+2018 (z=9)",
          "M18": "FIRE-2 intrinsic (Ma+2018, z=9-10)",
          "F20": "FLARES particle (Roper+[in prep])",
          "MO18": "Morishita+2018 (z=9-10)",
          "B19": "Bridge+2019 (z=7-8",
          "O16": "Oesch+2016 (z=11.1)",
          "S18": "Salmon+2018 (z=9.9)",
          "H20": "Holwerda+2020 (z=9-10.2)"}
markers = {"G11": "s", "G12": "v", "C16": "D",
           "K18": "o", "M18": "H", "F20": ".", "MO18": "o",
           "B19": "^", "O16": "P", "S18": "D", "H20": "*"}
colors = {"G11": "darkred", "G12": "darkred", "C16": "darkred",
           "K18": "darkred", "M18": "green", "F20": ".", "MO18": "darkred",
           "B19": "darkred", "O16": "darkred", "S18": "darkred", "H20": "darkred"}

z9_conv = cosmo.arcsec_per_kpc_proper(9).value

cmap = mpl.cm.get_cmap("cividis")
norm = plt.Normalize(vmin=7, vmax=10.5)

fit_lumins = np.logspace(np.log10(photconv.M_to_lum(-21.6)),
                         np.log10(photconv.M_to_lum(-18)),
                         1000)

# # ============================================================================
#
# fig = plt.figure()
# ax = fig.add_subplot(111)
# ax1 = ax.twinx()
#
# # Plot pixels
# ax1.axhline(5 * nircam_short, label="5 NIRCam-Short Pixels",
#             linestyle="dashed", color="k", alpha=0.2)
# ax1.axhline(5 * nircam_long, label="5 NIRCam-Long Pixels",
#             linestyle="dashdot", color="k", alpha=0.2)
# ax1.axhline(3 * wfc3, label="3 WFC3-IR Pixels",
#             linestyle="dotted", color="k", alpha=0.2)
#
# # ax.plot([26, 26.1], [100, 1000], color="k", alpha=0.8, label=labels["K18"])
# #
# # for z in [7, 8, 9]:
# #     ax.plot(lum_to_M(fit_lumins) + cosmo.distmod(z).value,
# #             kawa_fit(fit_lumins,
# #                      kawa_params['r_0'][int(z)],
# #                      kawa_params['beta'][int(z)]),
# #             color=cmap(norm(z)), alpha=0.8)
#
# for p in ["G11", "G12", "C16", "K18"]:
#
#     ax.scatter(mags[papers == p],
#                r_es[papers == p],
#                marker=markers[p], label=labels[p], s=16,
#                color=cmap(norm(zs[papers == p])))
#     ax1.scatter(mags[papers == p],
#                 r_es[papers == p] * z9_conv,
#                marker=markers[p], label=labels[p], s=0,
#                 color=cmap(norm(zs[papers == p])))
#
# ax.set_xlabel("$m_{i}$")
# ax.set_ylabel("$R_e / \mathrm{pkpc}$")
# ax1.set_ylabel("$R_e(z=9) / \mathrm{arcseconds}$")
#
# ax.set_ylim(0, 3.0)
# ax1.set_ylim(0, 3.0 * z9_conv)
# ax.set_xlim(24.1, 32.5)
# ax1.set_xlim(24.1, 32.5)
#
# handles2, _ = ax1.get_legend_handles_labels()
#
# legend_elements = [Line2D([0], [0], marker=markers[p], color='w',
#                           label=labels[p], markerfacecolor='k', markersize=8)
#                    for p in ["G11", "G12", "C16", "K18"]]
# legend_elements.extend(handles2[:3])
#
# ax.legend(handles=legend_elements)
#
# sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
# sm._A = []  # # fake up the array of the scalar mappable
# cbaxes = ax.inset_axes([0.0, 1.0, 1.0, 0.04])
# cbar = plt.colorbar(sm, cax=cbaxes, orientation="horizontal")
# cbaxes.xaxis.set_ticks_position("top")
# cbar.ax.set_xlabel("$z$", labelpad=-30)
#
# # plt.show()
# fig.savefig("plots/proposal_obs_size_mag.png", bbox_inches="tight")
#
# plt.close(fig)
#
# # ============================================================================

fig = plt.figure()
ax = fig.add_subplot(111)
ax1 = ax.twinx()
ax2 = ax.twiny()

# Add some extra space for the second axis at the bottom
fig.subplots_adjust(bottom=0.2)

ax.semilogy()
ax1.semilogy()

# Plot pixels
ax1.axhline(5 * nircam_short, label="5 NIRCam-Short Pixels",
            linestyle="dashed", color="k", alpha=0.2)
ax1.axhline(5 * nircam_long, label="5 NIRCam-Long Pixels",
            linestyle="dashdot", color="k", alpha=0.2)
ax1.axhline(3 * wfc3, label="3 WFC3-IR Pixels",
            linestyle="dotted", color="k", alpha=0.2)

ax.plot([26, 26.1], [100, 1000], color="k", alpha=0.8, label=labels["K18"])

for z in [7, 8, 9]:

    if z == 7:
        low_lim = -12.2
    elif z == 8:
        low_lim = -16.8
    else:
        low_lim = -15.4
    fit_lumins = np.logspace(np.log10(photconv.M_to_lum(-21.6)),
                             np.log10(photconv.M_to_lum(low_lim)),
                             1000)

    ax.plot(photconv.lum_to_M(fit_lumins),
            kawa_fit(fit_lumins,
                     kawa_params['r_0'][int(z)],
                     kawa_params['beta'][int(z)]),
            color=cmap(norm(z)), alpha=0.8)

for p in ["G11", "G12", "C16", "M18", "F20"]:

    if p != "F20":

        M = m_to_M(mags[papers == p], cosmo, zs[papers == p])

        ax.scatter(M, r_es[papers == p],
                   marker=markers[p], label=labels[p], s=10,
                   color=cmap(norm(zs[papers == p])), alpha=0.7)
        ax1.scatter(M, r_es[papers == p] * z9_conv,
                    marker=markers[p], label=labels[p], s=0,
                    color=cmap(norm(zs[papers == p])))
        ax2.scatter(M, r_es[papers == p],
                    marker=markers[p], label=labels[p], s=0,
                    color=cmap(norm(zs[papers == p])))


ax.set_xlabel("$M_{i}$")
ax.set_ylabel("$R_e / \mathrm{pkpc}$")
ax1.set_ylabel("$R_e(z=9) / \mathrm{arcseconds}$")

new_tick_locations = np.arange(m_to_M(24, cosmo, 9), m_to_M(29.5, cosmo, 9), 2)
bins = np.arange(-24, -10, 1)

# # Plot median
# plot_meidan_stat(m_to_M(mags, cosmo, zs), r_es, ax,
#                  lab="Median", color="darkorange", bins=bins,
#                  ls='-')

def tick_function(M):
    m = M_to_m(M, cosmo, 9)
    return ["%.2f" % im for im in m]

# Move twinned axis ticks and label from top to bottom
ax2.xaxis.set_ticks_position("bottom")
ax2.xaxis.set_label_position("bottom")

# Offset the twin axis below the host
ax2.spines["bottom"].set_position(("axes", -0.15))

# Turn on the frame for the twin axis, but then hide all
# but the bottom spine
ax2.set_frame_on(True)
ax2.patch.set_visible(False)

# as @ali14 pointed out, for python3, use this
for sp in ax2.spines.values():
    sp.set_visible(False)
ax2.spines["bottom"].set_visible(True)

ax2.set_xticks(new_tick_locations)
ax2.set_xticklabels(tick_function(new_tick_locations))
ax2.set_xlabel(r"$m_i(z=9)$")

handles2, _ = ax1.get_legend_handles_labels()

legend_elements = [Line2D([0], [0], marker=markers[p], color='w',
                          label=labels[p], markerfacecolor='k', markersize=6)
                   for p in ["G11", "G12", "C16", "M18"]]
legend_elements.insert(0, Line2D([0], [0], linestyle="-", color='k',
                                 label=labels["K18"], markersize=6))
# legend_elements.insert(0, Line2D([0], [0], linestyle="-", color='darkorange',
#                                  label="Median", markersize=8))
legend_elements.extend(handles2[:3])

ax.legend(handles=legend_elements)

sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
sm._A = []  # # fake up the array of the scalar mappable
cbaxes = ax.inset_axes([0.0, 1.0, 1.0, 0.04])
cbar = plt.colorbar(sm, cax=cbaxes, orientation="horizontal")
cbaxes.xaxis.set_ticks_position("top")
cbar.ax.set_xlabel("$z$", labelpad=-30)

ax.set_xlim(-24, np.max(new_tick_locations))
ax.set_ylim(0.005, 3)
ax1.set_ylim(0.005 * z9_conv, 3 * z9_conv)
ax2.set_xlim(-24, np.max(new_tick_locations))

# plt.show()
fig.savefig("plots/proposal_obs_size_abmag.png", bbox_inches="tight")

plt.close(fig)

# ============================================================================

cmap = mpl.cm.get_cmap("cividis")
norm = plt.Normalize(vmin=9, vmax=10.5)

fig = plt.figure(figsize=(6, 6))
ax = fig.add_subplot(111)
ax1 = ax.twinx()
ax2 = ax.twiny()

# Add some extra space for the second axis at the bottom
fig.subplots_adjust(bottom=0.2)

ax.semilogy()
ax1.semilogy()

# Plot pixels
ax1.axhline(5 * nircam_short, label="5 NIRCam-Short Pixels",
            linestyle="dashed", color="k", alpha=0.2)
ax1.axhline(5 * nircam_long, label="5 NIRCam-Long Pixels",
            linestyle="dashdot", color="k", alpha=0.2)
ax1.axhline(3 * wfc3, label="3 WFC3-IR Pixels",
            linestyle="dotted", color="k", alpha=0.2)
ax1.axhline(nircam_short, label="1 NIRCam-Short Pixels",
            linestyle="dashed", color="darkgray", alpha=0.2)
ax1.axhline(nircam_long, label="1 NIRCam-Long Pixels",
            linestyle="dashdot", color="darkgray", alpha=0.2)

ax.plot([26, 26.1], [100, 1000], color="k", alpha=0.8, label=labels["K18"])

for z in [9, ]:

    if z == 7:
        low_lim = -12.2
    elif z == 8:
        low_lim = -16.8
    else:
        low_lim = -15.4
    fit_lumins = np.logspace(np.log10(photconv.M_to_lum(-21.6)),
                             np.log10(photconv.M_to_lum(low_lim)),
                             1000)

    ax.plot(photconv.lum_to_M(fit_lumins),
            kawa_fit(fit_lumins,
                     kawa_params['r_0'][int(z)],
                     kawa_params['beta'][int(z)]),
            color=colors["K18"], alpha=0.8)

legend_elements = []

for p in markers.keys():

    if p != "F20" and p != "K18":

        okinds = zs[papers == p] >= 9

        if zs[papers == p][okinds].size == 0:
            continue

        M = m_to_M(mags[papers == p][okinds], cosmo, zs[papers == p][okinds])

        # if np.max(M) > -18:
        #     continue

        legend_elements.append(Line2D([0], [0], marker=markers[p], color='w',
                                  label=labels[p], markerfacecolor=colors[p],
                                  markersize=8, alpha=0.7))

        ax.scatter(M, r_es[papers == p][okinds],
                   marker=markers[p], label=labels[p], s=25,
                   color=colors[p], alpha=0.7)
        ax1.scatter(M, r_es[papers == p][okinds] * z9_conv,
                    marker=markers[p], label=labels[p], s=0,
                    color=cmap(norm(zs[papers == p][okinds])))
        ax2.scatter(M, r_es[papers == p][okinds],
                    marker=markers[p], label=labels[p], s=0,
                    color=cmap(norm(zs[papers == p][okinds])))

    else:
        continue

        # legend_elements.append(Line2D([0], [0], marker=markers[p], color='w',
        #                           label=labels[p], markerfacecolor='k',
        #                           markersize=6))
        #
        # M = photconv.lum_to_M(lumin)
        #
        # ax.scatter(M, hlr,
        #            marker=markers[p], label=labels[p], s=10,
        #            color="k", alpha=0.7)
        # ax1.scatter(M, hlr * z9_conv,
        #             marker=markers[p], label=labels[p], s=0,
        #             color="k")
        # ax2.scatter(M, hlr,
        #             marker=markers[p], label=labels[p], s=0,
        #             color="k")

ax.set_xlabel("$M_{\mathrm{UV}}$")
ax.set_ylabel("$R_e / \mathrm{pkpc}$")
ax1.set_ylabel("$R_e(z=9) / \mathrm{arcseconds}$")

new_tick_locations = np.arange(m_to_M(23, cosmo, 9), m_to_M(29.5, cosmo, 9), 1)
bins = np.arange(-24, -10, 1)

# # Plot median
# plot_meidan_stat(m_to_M(mags, cosmo, zs), r_es, ax,
#                  lab="Median", color="darkorange", bins=bins,
#                  ls='-')

def tick_function(M):
    m = M_to_m(M, cosmo, 9)
    return ["%.2f" % im for im in m]

# # Move twinned axis ticks and label from top to bottom
# ax2.xaxis.set_ticks_position("bottom")
# ax2.xaxis.set_label_position("bottom")

# # Offset the twin axis below the host
# ax2.spines["bottom"].set_position(("axes", -0.15))

# # Turn on the frame for the twin axis, but then hide all
# # but the bottom spine
# ax2.set_frame_on(True)
# ax2.patch.set_visible(False)

# # as @ali14 pointed out, for python3, use this
# for sp in ax2.spines.values():
#     sp.set_visible(False)
# ax2.spines["bottom"].set_visible(True)

ax2.set_xticks(new_tick_locations)
ax2.set_xticklabels(tick_function(new_tick_locations))
ax2.set_xlabel(r"$m(z=9)$")

handles2, labels2 = ax1.get_legend_handles_labels()

legend_elements.insert(0, Line2D([0], [0], linestyle="-", color=colors["K18"],
                                 label=labels["K18"], markersize=6))
# legend_elements.insert(0, Line2D([0], [0], linestyle="-", color='darkorange',
#                                  label="Median", markersize=8))
# legend_elements.extend(handles2[:5])

ax.legend(handles=legend_elements, ncol=2, loc="upper right", fontsize=8)
ax1.legend(handles=handles2[:5], ncol=2, loc="lower left", fontsize=8)

# sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
# sm._A = []  # # fake up the array of the scalar mappable
# cbaxes = ax.inset_axes([0.0, 1.0, 1.0, 0.04])
# cbar = plt.colorbar(sm, cax=cbaxes, orientation="horizontal")
# cbaxes.xaxis.set_ticks_position("top")
# cbar.ax.set_xlabel("$z$", labelpad=-30)

ax.set_xlim(-24, m_to_M(29.5, cosmo, 9))
ax.set_ylim(0.05, 5)
ax1.set_ylim(0.05 * z9_conv, 5 * z9_conv)
ax2.set_xlim(-24, m_to_M(29.5, cosmo, z))

# plt.show()
fig.savefig("plots/proposal_obs_size_abmag_z9.png", bbox_inches="tight")

plt.close(fig)

# # ============================================================================
#
# fig = plt.figure()
# ax = fig.add_subplot(111)
# ax1 = ax.twinx()
#
# # Plot pixels
# ax1.axhline(5 * nircam_short, label="5 NIRCam-Short Pixels",
#             linestyle="dashed", color="k", alpha=0.2)
# ax1.axhline(5 * nircam_long, label="5 NIRCam-Long Pixels",
#             linestyle="dashdot", color="k", alpha=0.2)
# ax1.axhline(3 * wfc3, label="3 WFC3-IR Pixels",
#             linestyle="dotted", color="k", alpha=0.2)
#
# # ax.plot([26, 26.1], [100, 1000], color="k", alpha=0.8, label=labels["K18"])
#
# # for z in [7, 8, 9]:
# #     ax.plot(np.log10(fit_lumins), #lum_to_M(fit_lumins) + cosmo.distmod(z).value,
# #             kawa_fit(fit_lumins,
# #                      kawa_params['r_0'][int(z)],
# #                      kawa_params['beta'][int(z)]),
# #             color=cmap(norm(z)), alpha=0.8)
#
# for p in ["G11", "G12", "C16", "K18"]:
#
#     ax.scatter(np.log10(photconv.M_to_lum(mags[papers == p]
#                         - cosmo.distmod(zs[papers == p]).value)),
#                r_es[papers == p],
#                marker=markers[p], label=labels[p], s=16,
#                color=cmap(norm(zs[papers == p])))
#     ax1.scatter(np.log10(photconv.M_to_lum(mags[papers == p]
#                          - cosmo.distmod(zs[papers == p]).value)),
#                 r_es[papers == p] * z9_conv,
#                marker=markers[p], label=labels[p], s=0,
#                 color=cmap(norm(zs[papers == p])))
#
# ax.set_xlabel("$\log_{10}(L / \mathrm{erg s}^{-1} \mathrm{ Hz}^{-1})$")
# ax.set_ylabel("$R_e / \mathrm{pkpc}$")
# ax1.set_ylabel("$R_e(z=9) / \mathrm{arcseconds}$")
#
# ax.set_ylim(0, 3.0)
# ax1.set_ylim(0, 3.0 * z9_conv)
# ax.set_xlim(28, 33)
# ax1.set_xlim(28, 33)
#
# handles2, _ = ax1.get_legend_handles_labels()
#
# legend_elements = [Line2D([0], [0], marker=markers[p], color='w',
#                           label=labels[p], markerfacecolor='k', markersize=8)
#                    for p in ["G11", "G12", "C16", "K18"]]
# legend_elements.extend(handles2[:3])
#
# ax.legend(handles=legend_elements)
#
# sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
# sm._A = []  # # fake up the array of the scalar mappable
# cbaxes = ax.inset_axes([0.0, 1.0, 1.0, 0.04])
# cbar = plt.colorbar(sm, cax=cbaxes, orientation="horizontal")
# cbaxes.xaxis.set_ticks_position("top")
# cbar.ax.set_xlabel("$z$", labelpad=-30)
#
# # plt.show()
# fig.savefig("plots/proposal_obs_size_lumin.png", bbox_inches="tight")
#
# plt.close(fig)
#
# # ============================================================================
#
# fig = plt.figure()
# ax = fig.add_subplot(111)
# ax1 = ax.twinx()
#
# ax.semilogy()
# ax1.semilogy()
#
# # Plot pixels
# ax1.axhline(5 * nircam_short, label="5 NIRCam-Short Pixels",
#             linestyle="dashed", color="k", alpha=0.2)
# ax1.axhline(5 * nircam_long, label="5 NIRCam-Long Pixels",
#             linestyle="dashdot", color="k", alpha=0.2)
# ax1.axhline(3 * wfc3, label="3 WFC3-IR Pixels",
#             linestyle="dotted", color="k", alpha=0.2)
#
# # ax.plot([26, 26.1], [100, 1000], color="k", alpha=0.8, label=labels["K18"])
# #
# # for z in [7, 8, 9]:
# #     ax.plot(lum_to_M(fit_lumins) + cosmo.distmod(z).value,
# #             kawa_fit(fit_lumins,
# #                      kawa_params['r_0'][int(z)],
# #                      kawa_params['beta'][int(z)]),
# #             color=cmap(norm(z)), alpha=0.8)
#
# for p in ["G11", "G12", "C16", "K18"]:
#
#     ax.scatter(mags[papers == p],
#                r_es[papers == p],
#                marker=markers[p], label=labels[p], s=16,
#                color=cmap(norm(zs[papers == p])))
#     ax1.scatter(mags[papers == p],
#                 r_es[papers == p] * z9_conv,
#                marker=markers[p], label=labels[p], s=0,
#                 color=cmap(norm(zs[papers == p])))
#
# ax.set_xlabel("$m_{i}$")
# ax.set_ylabel("$R_e / \mathrm{pkpc}$")
# ax1.set_ylabel("$R_e(z=9) / \mathrm{arcseconds}$")
#
# ax.set_ylim(0, 3.0)
# ax1.set_ylim(0, 3.0 * z9_conv)
# ax.set_xlim(22.1, 32.5)
# ax1.set_xlim(22.1, 32.5)
#
# handles2, _ = ax1.get_legend_handles_labels()
#
# legend_elements = [Line2D([0], [0], marker=markers[p], color='w',
#                           label=labels[p], markerfacecolor='k', markersize=8)
#                    for p in ["G11", "G12", "C16", "K18"]]
# legend_elements.extend(handles2[:3])
#
# ax.legend(handles=legend_elements)
#
# sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
# sm._A = []  # # fake up the array of the scalar mappable
# cbaxes = ax.inset_axes([0.0, 1.0, 1.0, 0.04])
# cbar = plt.colorbar(sm, cax=cbaxes, orientation="horizontal")
# cbaxes.xaxis.set_ticks_position("top")
# cbar.ax.set_xlabel("$z$", labelpad=-30)
#
# # plt.show()
# fig.savefig("plots/proposal_obs_size_mag_log.png", bbox_inches="tight")
#
# plt.close(fig)
#
# # ============================================================================
#
# fig = plt.figure()
# ax = fig.add_subplot(111)
# ax1 = ax.twinx()
#
# ax.semilogy()
# ax1.semilogy()
#
# # Plot pixels
# ax1.axhline(5 * nircam_short, label="5 NIRCam-Short Pixels",
#             linestyle="dashed", color="k", alpha=0.2)
# ax1.axhline(5 * nircam_long, label="5 NIRCam-Long Pixels",
#             linestyle="dashdot", color="k", alpha=0.2)
# ax1.axhline(3 * wfc3, label="3 WFC3-IR Pixels",
#             linestyle="dotted", color="k", alpha=0.2)
#
# # ax.plot([26, 26.1], [100, 1000], color="k", alpha=0.8, label=labels["K18"])
#
# # for z in [7, 8, 9]:
# #     ax.plot(np.log10(fit_lumins), #lum_to_M(fit_lumins) + cosmo.distmod(z).value,
# #             kawa_fit(fit_lumins,
# #                      kawa_params['r_0'][int(z)],
# #                      kawa_params['beta'][int(z)]),
# #             color=cmap(norm(z)), alpha=0.8)
#
# for p in ["G11", "G12", "C16", "K18"]:
#
#     ax.scatter(np.log10(photconv.M_to_lum(m_to_M(mags[papers == p], cosmo, zs[papers == p]))),
#                r_es[papers == p],
#                marker=markers[p], label=labels[p], s=16,
#                color=cmap(norm(zs[papers == p])))
#     ax1.scatter(np.log10(photconv.M_to_lum(m_to_M(mags[papers == p], cosmo, zs[papers == p]))),
#                 r_es[papers == p] * z9_conv,
#                marker=markers[p], label=labels[p], s=0,
#                 color=cmap(norm(zs[papers == p])))
#
# ax.set_xlabel("$\log_{10}(L / \mathrm{erg s}^{-1} \mathrm{ Hz}^{-1})$")
# ax.set_ylabel("$R_e / \mathrm{pkpc}$")
# ax1.set_ylabel("$R_e(z=9) / \mathrm{arcseconds}$")
#
# ax.set_ylim(0, 3.0)
# ax1.set_ylim(0, 3.0 * z9_conv)
# ax.set_xlim(28, 33)
# ax1.set_xlim(28, 33)
#
# handles2, _ = ax1.get_legend_handles_labels()
#
# legend_elements = [Line2D([0], [0], marker=markers[p], color='w',
#                           label=labels[p], markerfacecolor='k', markersize=8)
#                    for p in ["G11", "G12", "C16", "K18"]]
# legend_elements.extend(handles2[:3])
#
# ax.legend(handles=legend_elements)
#
# sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
# sm._A = []  # # fake up the array of the scalar mappable
# cbaxes = ax.inset_axes([0.0, 1.0, 1.0, 0.04])
# cbar = plt.colorbar(sm, cax=cbaxes, orientation="horizontal")
# cbaxes.xaxis.set_ticks_position("top")
# cbar.ax.set_xlabel("$z$", labelpad=-30)
#
# # plt.show()
# fig.savefig("plots/proposal_obs_size_lumin_log.png", bbox_inches="tight")
#
# plt.close(fig)

