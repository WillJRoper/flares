import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from astropy.cosmology import Planck13 as cosmo
import astropy.units as u
import FLARE.plt
from FLARE.photom import lum_to_M, M_to_lum
import seaborn as sns


# Define factor relating the L to M in cm^2
geo = 4. * np.pi * (100. * 10. * 3.0867 * 10 ** 16) ** 2


def M_to_lum(M):
    return 10 ** (-0.4 * (M + 48.6)) * geo

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
kawa_fit = lambda l, r0, b: r0 * (l / M_to_lum(-21)) ** b


df = pd.read_csv("/Users/willroper/Downloads/HighzSizes.csv")

papers = df["Paper"]
mags = df["Magnitude"]
r_es_arcs = df["R_e (arcsec)"]
zs = df["Redshift"]

# Define pixel resolutions
wfc3 = 0.13
nircam_short = 0.031
nircam_long = 0.063

# Convert to physical kiloparsecs
r_es = np.zeros(len(papers))
for (ind, r), z in zip(enumerate(r_es_arcs), zs):
    if papers[ind] == "K18":
        continue
    r_es[ind] = r / cosmo.arcsec_per_kpc_proper(z).value

labels = {"G11": "Grazian+2011: $i=$J",
          "G12": "Grazian+2012: $i=$J",
          "C16": "Calvi+2016: $i=$H",
          "K18": "Kawamata+2018: $i=\mathrm{UV}$"}
markers = {"G11": "s", "G12": "v", "C16": "D",
           "K18": "o"}
colors = {"G11": "lightskyblue", "G12": "khaki", "C16": "lightcoral",
           "K18": "yellowgreen"}

z9_conv = cosmo.arcsec_per_kpc_proper(9).value

cmap = mpl.cm.get_cmap("plasma")
norm = plt.Normalize(vmin=7, vmax=10.5)

fit_lumins = np.logspace(24.5, 31, 1000)

fig = plt.figure()
ax = fig.add_subplot(111)
ax1 = ax.twinx()

# Plot pixels
ax1.axhline(5 * nircam_short, label="5 NIRCam-Short Pixels",
            linestyle="dashed", color="k", alpha=0.2)
ax1.axhline(5 * nircam_long, label="5 NIRCam-Long Pixels",
            linestyle="dashdot", color="k", alpha=0.2)
ax1.axhline(3 * wfc3, label="3 WFC3-IR Pixels",
            linestyle="dotted", color="k", alpha=0.2)

ax.plot([26, 26.1], [100, 1000], color="k", alpha=0.8, label=labels["K18"])

for z in [7, 8, 9]:
    ax.plot(lum_to_M(fit_lumins) + cosmo.distmod(z).value,
            kawa_fit(fit_lumins,
                     kawa_params['r_0'][int(z)],
                     kawa_params['beta'][int(z)]),
            color=cmap(norm(z)), alpha=0.8)

for p in ["G11", "G12", "C16"]:

    ax.scatter(mags[papers == p],
               r_es[papers == p],
               marker=markers[p], label=labels[p], s=20,
               color=cmap(norm(zs[papers == p])))
    ax1.scatter(mags[papers == p],
                r_es[papers == p] * z9_conv,
               marker=markers[p], label=labels[p], s=0,
                color=cmap(norm(zs[papers == p])))

# ax.set_xlabel("$\log_{10}(L / \mathrm{erg s}^{-1} \mathrm{ Hz}^{-1})$")
ax.set_xlabel("$m_{i}$")
ax.set_ylabel("$R_e / \mathrm{pkpc}$")
ax1.set_ylabel("$R_e(z=9) / \mathrm{arcseconds}$")

ax.set_ylim(0, 3.0)
ax1.set_ylim(0, 3.0 * z9_conv)
ax.set_xlim(24.1, 30.5)
ax1.set_xlim(24.1, 30.5)

handles, labels = [], []
handles1, labels1 = ax.get_legend_handles_labels()
handles2, labels2 = ax1.get_legend_handles_labels()
handles.extend(handles1)
handles.extend(handles2[:3])
labels.extend(labels1)
labels.extend(labels2[:3])
ax.legend(handles, labels)

sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
sm._A = []  # # fake up the array of the scalar mappable
cbaxes = ax.inset_axes([0.0, 1.0, 1.0, 0.04])
cbar = plt.colorbar(sm, cax=cbaxes, orientation="horizontal")
cbaxes.xaxis.set_ticks_position("top")
cbar.ax.set_xlabel("$z$", labelpad=-30)

# plt.show()
fig.savefig("plots/proposal_obs_size_lumin.png", bbox_inches="tight")
