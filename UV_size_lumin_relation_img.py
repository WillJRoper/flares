#!/cosma/home/dp004/dc-rope1/.conda/envs/flares-env/bin/python
import numpy as np
import h5py
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LogNorm
import matplotlib.gridspec as gridspec
from scipy.stats import binned_statistic
from astropy.cosmology import Planck13 as cosmo
from astropy import units as u
from scipy.optimize import curve_fit


sns.set_context("paper")
sns.set_style('whitegrid')

geo = 4.*np.pi*(100.*10.*3.0867*10**16)**2 # factor relating the L to M in cm^2


def M_to_lum(M):

    return 10**(-0.4*(M+48.6)) * geo


def plot_meidan_stat(xs, ys, ax, lab, color, bins=None, ls='-'):

    if bins == None:
        bin = np.logspace(np.log10(xs.min()), np.log10(xs.max()), 15)
    else:
        bin = bins

    # Compute binned statistics
    y_stat, binedges, bin_ind = binned_statistic(xs, ys, statistic='median', bins=bin)

    # Compute bincentres
    bin_wid = binedges[1] - binedges[0]
    bin_cents = binedges[1:] - bin_wid / 2

    okinds = np.logical_and(~np.isnan(bin_cents), ~np.isnan(y_stat))

    ax.plot(bin_cents[okinds], y_stat[okinds], color=color, linestyle=ls, label=lab)


# Define Kawamata17 fit and parameters
kawa_params = {'beta': {6: 0.46, 7: 0.46, 8: 0.38, 9: 0.56}, 'r_0': {6: 0.94, 7: 0.94, 8: 0.81, 9: 1.2}}
kawa_up_params = {'beta': {6: 0.46 + 0.08, 7: 0.46 + 0.08, 8: 0.38 + 0.28, 9: 0.56 + 1.01}, 'r_0': {6: 0.94 + 0.2, 7: 0.94 + 0.2, 8: 0.81 + 5.28, 9: 1.2 + 367.64}}
kawa_low_params = {'beta': {6: 0.46 - 0.09, 7: 0.46 - 0.09, 8: 0.38 - 0.78, 9: 0.56 - 0.27}, 'r_0': {6: 0.94 - 0.15, 7: 0.94 - 0.15, 8: 0.81 - 0.26, 9: 1.2 - 0.74}}
kawa_fit = lambda l, r0, b: r0 * (l / M_to_lum(-21))**b

regions = []
for reg in range(0, 40):
    if reg < 10:
        regions.append('0' + str(reg))
    else:
        regions.append(str(reg))

snaps = ['003_z012p000', '004_z011p000', '005_z010p000',
         '006_z009p000', '007_z008p000', '008_z007p000',
         '009_z006p000', '010_z005p000', '011_z004p770']

# Define filter
f = 'FAKE.TH.FUV'

reg_snaps = []
for reg in regions:

    for snap in snaps:

        reg_snaps.append((reg, snap))

# Initialise dictionaries
hlr_dict = {}
lumin_dict = {}

for snap in snaps:

    hlr_dict[snap] = []
    lumin_dict[snap] = []

for ind in range(len(reg_snaps)):

    reg, snap = reg_snaps[ind]

    hdfpath = '/cosma/home/dp004/dc-rope1/FLARES/FLARES-1/WebbData/GEAGLE_' + reg + '/'

    print(reg_snaps[ind])

    try:
        hdf = h5py.File(hdfpath + "RestUV" + snap + '.hdf5', 'r')

        try:
            lumins = hdf[f]['Aperture_Luminosity_30kpc'][:, 0]
            ms = hdf[f]['Aperture_Mass_30kpc'][:, 0] * 10**10
            hlrs = hdf[f]['Image_half_light_rad'][:, 0]

            okinds = ms > 10**9
            lumins = lumins[okinds]
            hlrs = hlrs[okinds]
        except UnboundLocalError:
            continue
        except KeyError:
            continue

        print(lumins.shape)
        print(hlrs.shape)

        hlr_dict[snap].extend(hlrs)
        lumin_dict[snap].extend(lumins)
    except OSError:
        continue

# Define comoving softening length in kpc
csoft = 0.001802390 / 0.6777

fit_lumins = np.logspace(28, 31, 1000)

axlims_x = []
axlims_y = []

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

    hlrs = np.array(hlr_dict[snap])
    lumins = np.array(lumin_dict[snap])

    okinds = np.logical_and(hlrs / (csoft / (1 + z)) > 10**-1, np.logical_and(lumins > 10**28, lumins < 10**50))
    lumins = lumins[okinds]
    hlrs = hlrs[okinds]
    try:
        cbar = ax.hexbin(lumins, hlrs / (csoft / (1 + z)), gridsize=100, mincnt=1, xscale='log', yscale='log',
                         norm=LogNorm(), linewidths=0.2, cmap='viridis')
        # plot_meidan_stat(lumins, hlrs / (csoft / (1 + z)), ax, lab='REF', color='r')
        popt, pcov = curve_fit(kawa_fit, lumins, hlrs, p0=[1, 1])
        ax.plot(fit_lumins, kawa_fit(fit_lumins, popt[0], popt[1]) / (csoft / (1 + z)) , color='b')
    except ValueError:
        continue

    if int(z) in [6, 7, 8, 9]:
        ax.plot(fit_lumins,
                kawa_fit(fit_lumins, kawa_params['r_0'][int(z)], kawa_params['beta'][int(z)]) / (csoft / (1 + z)),
                linestyle='dashed', color='k', alpha=0.9)

    ax.text(0.8, 0.1, f'$z={z}$', bbox=dict(boxstyle="round,pad=0.3", fc='w', ec="k", lw=1, alpha=0.8),
            transform=ax.transAxes, horizontalalignment='right', fontsize=8)

    axlims_x.extend(ax.get_xlim())
    axlims_y.extend(ax.get_ylim())

    # Label axes
    if i == 2:
        ax.set_xlabel(r'$L_{FUV}/$ [erg $/$ s $/$ Hz]')
    if j == 0:
        ax.set_ylabel('$R_{1/2}/\epsilon$')

for ax in [ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8, ax9]:
    ax.set_xlim(np.min(axlims_x), np.max(axlims_x))
    ax.set_ylim(10**-1.1, 10**2.2)
    for spine in ax.spines.values():
        spine.set_edgecolor('k')

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

fig.savefig('plots/Img_HalfLightRadiusFUV_soft.png', bbox_inches='tight')

plt.close(fig)

axlims_x = []
axlims_y = []

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

    hlrs = np.array(hlr_dict[snap])
    lumins = np.array(lumin_dict[snap])

    okinds = np.logical_and(hlrs / (csoft / (1 + z)) > 10**-1, np.logical_and(lumins > 10**28, lumins < 10**50))
    lumins = lumins[okinds]
    hlrs = hlrs[okinds] * 1000
    try:
        cbar = ax.hexbin(lumins, hlrs, gridsize=50, mincnt=1, xscale='log', yscale='log',
                         norm=LogNorm(), linewidths=0.2, cmap='viridis')
        popt, pcov = curve_fit(kawa_fit, lumins, hlrs, p0=[1, 1])
        ax.plot(fit_lumins, kawa_fit(fit_lumins, popt[0], popt[1]), color='b')
        # plot_meidan_stat(lumins, hlrs, ax, lab='REF', color='r')
    except ValueError:
        continue

    if int(z) in [6, 7, 8, 9]:
        ax.plot(fit_lumins, kawa_fit(fit_lumins, kawa_params['r_0'][int(z)], kawa_params['beta'][int(z)]),
                linestyle='dashed', color='k', alpha=0.9)

    ax.text(0.8, 0.1, f'$z={z}$', bbox=dict(boxstyle="round,pad=0.3", fc='w', ec="k", lw=1, alpha=0.8),
            transform=ax.transAxes, horizontalalignment='right', fontsize=8)

    axlims_x.extend(ax.get_xlim())
    axlims_y.extend(ax.get_ylim())

    # Label axes
    if i == 2:
        ax.set_xlabel(r'$L_{FUV}/$ [erg $/$ s $/$ Hz]')
    if j == 0:
        ax.set_ylabel('$R_{1/2}/ [pkpc]$')

for ax in [ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8, ax9]:
    ax.set_xlim(np.min(axlims_x), np.max(axlims_x))
    ax.set_ylim(np.min(axlims_y), np.max(axlims_y))
    for spine in ax.spines.values():
        spine.set_edgecolor('k')

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

fig.savefig('plots/Img_HalfLightRadiusFUV.png', bbox_inches='tight')

plt.close(fig)

for snap in snaps:

    z_str = snap.split('z')[1].split('p')
    z = float(z_str[0] + '.' + z_str[1])

    hlrs = np.array(hlr_dict[snap])
    lumins = np.array(lumin_dict[snap])

    okinds = np.logical_and(hlrs / (csoft / (1 + z)) > 10**-1, np.logical_and(lumins > 10**28, lumins < 10**50))
    lumins = lumins[okinds]
    hlrs = hlrs[okinds] * 1000

    fig = plt.figure()
    ax = fig.add_subplot(111)
    
    try:
        cbar = ax.hexbin(lumins, hlrs, gridsize=50, mincnt=1, xscale='log', yscale='log',
                         norm=LogNorm(), linewidths=0.2, cmap='viridis', zorder=0)
        popt, pcov = curve_fit(kawa_fit, lumins, hlrs, p0=[1, 1])
        ax.plot(fit_lumins, kawa_fit(fit_lumins, popt[0], popt[1]), color='b')
    except ValueError:
        continue

    if int(z) in [6, 7, 8, 9]:
        ax.plot(fit_lumins, kawa_fit(fit_lumins, kawa_params['r_0'][int(z)], kawa_params['beta'][int(z)]),
                linestyle='dashed', color='k', alpha=0.9, zorder=2)
        ax.fill_between(fit_lumins,
                        kawa_fit(fit_lumins, kawa_low_params['r_0'][int(z)], kawa_low_params['beta'][int(z)]),
                        kawa_fit(fit_lumins, kawa_up_params['r_0'][int(z)], kawa_up_params['beta'][int(z)]),
                        color='k', alpha=0.4, zorder=1)

    ax.text(0.8, 0.1, f'$z={z}$', bbox=dict(boxstyle="round,pad=0.3", fc='w', ec="k", lw=1, alpha=0.8),
            transform=ax.transAxes, horizontalalignment='right', fontsize=8)

    # Label axes
    ax.set_xlabel(r'$L_{FUV}/$ [erg $/$ s $/$ Hz]')
    ax.set_ylabel('$R_{1/2}/ [pkpc]$')

    fig.savefig('plots/Img_HalfLightRadiusFUV_' + str(z) + '.png', bbox_inches='tight')

    plt.close(fig)

