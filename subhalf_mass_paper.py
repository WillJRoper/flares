#!/cosma/home/dp004/dc-rope1/.conda/envs/flares-env/bin/python
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import astropy.units as u
from matplotlib.colors import LogNorm
import matplotlib.gridspec as gridspec
from scipy.stats import binned_statistic
import eagle_IO.eagle_IO as E
import seaborn as sns
from astropy.cosmology import Planck13 as cosmo
matplotlib.use('Agg')

sns.set_style('whitegrid')


master_path = "/cosma7/data/dp004/dc-payy1/my_files/flares_pipeline/data/flares.hdf5"

regions = []
for reg in range(0, 40):

    if reg < 10:
        regions.append('000' + str(reg))
    else:
        regions.append('00' + str(reg))

snaps = ['003_z012p000', '004_z011p000', '005_z010p000',
         '006_z009p000', '007_z008p000', '008_z007p000',
         '009_z006p000', '010_z005p000', '011_z004p770']
all_snaps = ['000_z015p000', '001_z014p000', '002_z013p000', '003_z012p000', '004_z011p000', '005_z010p000',
         '006_z009p000', '007_z008p000', '008_z007p000', '009_z006p000', '010_z005p000', '011_z004p770']
axlims_x = []
axlims_y = []


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
    y_stat, binedges, bin_ind = binned_statistic(xs, ys, statistic='median', bins=bin)

    # Compute bincentres
    bin_wid = binedges[1] - binedges[0]
    bin_cents = binedges[1:] - bin_wid / 2

    okinds = np.logical_and(~np.isnan(bin_cents), ~np.isnan(y_stat))

    ax.plot(bin_cents[okinds], y_stat[okinds], color=color, linestyle=ls, label=lab)
    

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
    y_stat_16, binedges, bin_ind = binned_statistic(zs, ys, statistic=lambda y: np.percentile(y, 16), bins=bin)
    y_stat_84, binedges, bin_ind = binned_statistic(zs, ys, statistic=lambda y: np.percentile(y, 84), bins=bin)

    # Compute bincentres
    bin_cents = uniz

    okinds = np.logical_and(~np.isnan(bin_cents), np.logical_and(~np.isnan(y_stat_16), ~np.isnan(y_stat_84)))

    ax.fill_between(bin_cents[okinds], y_stat_16[okinds], y_stat_84[okinds], alpha=0.3, color=color)


half_mass_rads_dict = {}
xaxis_dict = {}
for snap in all_snaps:

    half_mass_rads_dict[snap] = []
    xaxis_dict[snap] = []

for reg in regions:

    for snap in all_snaps:

        print(reg, snap)

        path = '/cosma7/data/dp004/dc-love2/data/G-EAGLE/geagle_' + reg + '/data/'
        # path = '/cosma/home/dp004/dc-rope1/FLARES/FLARES-1/FLARES_00_medFBlim/data/'
        try:
            half_mass_rads_dict[snap].extend(E.read_array('SUBFIND', path, snap, 'Subhalo/HalfMassRad', noH=True,
                                                          numThreads=8)[:, 4] * 1e3)
            xaxis_dict[snap].extend(E.read_array('SUBFIND', path, snap, 'Subhalo/ApertureMeasurements/Mass/030kpc',
                                                 noH=True, numThreads=8)[:, 4] * 10**10)
        except OSError:
            continue

evo_zs_hm = []
evo_hmrs_hm = []
eagle_evo_zs_hm = []
eagle_evo_hmrs_hm = []
evo_zs_lm = []
evo_hmrs_lm = []
eagle_evo_zs_lm = []
eagle_evo_hmrs_lm = []

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

running_total = 0

for ax, snap, (i, j) in zip([ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8, ax9], snaps,
                            [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2), (2, 0), (2, 1), (2, 2)]):

    z_str = snap.split('z')[1].split('p')
    z = float(z_str[0] + '.' + z_str[1])

    if z <= 2.8:
        soft = 0.000474390 / 0.6777 * 1e3
    else:
        soft = 0.001802390 / (0.6777 * (1 + z)) * 1e3

    xs = np.array(xaxis_dict[snap])
    half_mass_rads_plt = np.array(half_mass_rads_dict[snap])
    
    xs_plt = xs[half_mass_rads_plt > 0]
    half_mass_rads_plt = half_mass_rads_plt[half_mass_rads_plt > 0]
    half_mass_rads_plt = half_mass_rads_plt[xs_plt > 1e8]
    xs_plt = xs_plt[xs_plt > 1e8]

    fig1 = plt.figure()
    ax10 = fig1.add_subplot(111)

    try:
        cbar = ax.hexbin(xs_plt, half_mass_rads_plt, gridsize=100, mincnt=1, xscale='log', yscale='log', norm=LogNorm(),
                         linewidths=0.2, cmap='viridis', alpha=0.7)
        plot_meidan_stat(xs_plt, half_mass_rads_plt, ax, lab='REF', color='r')

        cbar1 = ax10.hexbin(xs_plt, half_mass_rads_plt, gridsize=100, mincnt=1, xscale='log', yscale='log', norm=LogNorm(),
                         linewidths=0.2, cmap='viridis', alpha=0.7)
        plot_meidan_stat(xs_plt, half_mass_rads_plt, ax10, lab='REF', color='r')

        print(snap, xs_plt[xs_plt > 1e11].size)
        running_total += xs_plt[xs_plt > 1e11].size
    except ValueError:
        continue

    ax.text(0.8, 0.9, f'$z={z}$', bbox=dict(boxstyle="round,pad=0.3", fc='w', ec="k", lw=1, alpha=0.8),
            transform=ax.transAxes, horizontalalignment='right', fontsize=8)

    axlims_x.extend(ax.get_xlim())
    axlims_y.extend(ax.get_ylim())

    # Label axes
    if i == 2:
        ax.set_xlabel(r'$M_{\star}/M_\odot$')
    if j == 0:
        ax.set_ylabel('$R_{1/2,*}/ [pkpc]$')
    ax10.set_xlabel(r'$M_{\star}/M_\odot$')
    ax10.set_ylabel('$R_{1/2,*}/\epsilon$')

    fig1.savefig('plots/HalfMassRadius_' + snap + '.png',
                bbox_inches='tight')

for ax in [ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8, ax9]:

    ax.set_xlim(10**8, 10**11.25)
    ax.set_ylim(10**-1.2, 10**2.2)

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

# fig.savefig('plots/HalfMassRadius_all_snaps.png',
#             bbox_inches='tight')
fig.savefig('plots/HalfMassRadius_all_snaps.png',
            bbox_inches='tight')

plt.close(fig)

print("In total", running_total, "galaxies with mass greater than 10^11")

for snap in all_snaps:

    z_str = snap.split('z')[1].split('p')
    z = float(z_str[0] + '.' + z_str[1])

    xs = np.array(xaxis_dict[snap])
    half_mass_rads_plt = np.array(half_mass_rads_dict[snap])

    xs_plt = xs[half_mass_rads_plt > 0]
    half_mass_rads_plt = half_mass_rads_plt[half_mass_rads_plt > 0]
    half_mass_rads_plt = half_mass_rads_plt[xs_plt > 1e8]
    xs_plt = xs_plt[xs_plt > 1e8]

    evo_hmr = half_mass_rads_plt[xs_plt > 1e9]
    evo_zs_hm.extend(np.full_like(evo_hmr, z))
    evo_hmrs_hm.extend(evo_hmr)
    evo_hmr = half_mass_rads_plt[xs_plt < 1e9]
    evo_zs_lm.extend(np.full_like(evo_hmr, z))
    evo_hmrs_lm.extend(evo_hmr)

snaps = ['000_z020p000', '003_z008p988', '006_z005p971', '009_z004p485', '012_z003p017', '015_z002p012',
         '018_z001p259', '021_z000p736', '024_z000p366', '027_z000p101', '001_z015p132', '004_z008p075',
         '007_z005p487', '010_z003p984', '013_z002p478', '016_z001p737', '019_z001p004', '022_z000p615',
         '025_z000p271', '028_z000p000', '002_z009p993', '005_z007p050', '008_z005p037', '011_z003p528',
         '014_z002p237', '017_z001p487', '020_z000p865', '023_z000p503', '026_z000p183']
path = '/cosma7/data//Eagle/ScienceRuns/Planck1/L0050N0752/PE/AGNdT9/data/'
axlims_x = []
axlims_y = []

half_mass_rads_dict = {}
xaxis_dict = {}

for snap in snaps:

    print(snap)

    half_mass_rads_dict[snap] = E.read_array('SUBFIND', path, snap, 'Subhalo/HalfMassRad', noH=True,
                                                  numThreads=8)[:, 4] * 1e3
    xaxis_dict[snap] = E.read_array('SUBFIND', path, snap, 'Subhalo/ApertureMeasurements/Mass/030kpc',
                                         noH=True, numThreads=8)[:, 4] * 10**10

for snap in snaps:

    z_str = snap.split('z')[1].split('p')
    z = float(z_str[0] + '.' + z_str[1])

    xs = np.array(xaxis_dict[snap])
    half_mass_rads_plt = np.array(half_mass_rads_dict[snap])

    xs_plt = xs[half_mass_rads_plt > 0]
    half_mass_rads_plt = half_mass_rads_plt[half_mass_rads_plt > 0]
    half_mass_rads_plt = half_mass_rads_plt[xs_plt > 1e8]
    xs_plt = xs_plt[xs_plt > 1e8]

    evo_hmr = half_mass_rads_plt[xs_plt > 1e9]
    eagle_evo_zs_hm.extend(np.full_like(evo_hmr, z))
    eagle_evo_hmrs_hm.extend(evo_hmr)
    evo_hmr = half_mass_rads_plt[xs_plt < 1e9]
    eagle_evo_zs_lm.extend(np.full_like(evo_hmr, z))
    eagle_evo_hmrs_lm.extend(evo_hmr)

fig = plt.figure()
ax = fig.add_subplot(111)

plot_meidan_stat(eagle_evo_zs_lm, eagle_evo_hmrs_lm, ax, lab='EAGLE-LM', color='darkorange', bins=1, ls="--")
plot_spread_stat(eagle_evo_zs_lm, eagle_evo_hmrs_lm, ax, color='darkorange')
plot_meidan_stat(eagle_evo_zs_hm, eagle_evo_hmrs_hm, ax, lab='EAGLE-HM', color='blueviolet', bins=1, ls="--")
plot_spread_stat(eagle_evo_zs_hm, eagle_evo_hmrs_hm, ax, color='blueviolet')

plot_meidan_stat(evo_zs_lm, evo_hmrs_lm, ax, lab='FLARES-LM', color='orangered', bins=1)
plot_spread_stat(evo_zs_lm, evo_hmrs_lm, ax, color='orangered')
plot_meidan_stat(evo_zs_hm, evo_hmrs_hm, ax, lab='FLARES-HM', color='royalblue', bins=1)
plot_spread_stat(evo_zs_hm, evo_hmrs_hm, ax, color='royalblue')

ax.set_xlabel("$z$")
ax.set_ylabel('$R_{1/2,*}/ [\mathrm{pkpc}]$')

handles, labels = ax.get_legend_handles_labels()
ax.legend(handles, labels)

fig.savefig("plots/HalfMassRadius_z_evolution.png", bbox_inches="tight")

plt.close(fig)

fig = plt.figure()
ax = fig.add_subplot(111)

ax.loglog()

plot_meidan_stat(cosmo.age(eagle_evo_zs_lm).value, eagle_evo_hmrs_lm, ax, lab='EAGLE-LM', color='darkorange', bins=1, ls="--")
plot_spread_stat(cosmo.age(eagle_evo_zs_lm).value, eagle_evo_hmrs_lm, ax, color='darkorange')
plot_meidan_stat(cosmo.age(eagle_evo_zs_hm).value, eagle_evo_hmrs_hm, ax, lab='EAGLE-HM', color='blueviolet', bins=1, ls="--")
plot_spread_stat(cosmo.age(eagle_evo_zs_hm).value, eagle_evo_hmrs_hm, ax, color='blueviolet')

plot_meidan_stat(cosmo.age(evo_zs_lm).value, evo_hmrs_lm, ax, lab='FLARES-LM', color='orangered', bins=1)
plot_spread_stat(cosmo.age(evo_zs_lm).value, evo_hmrs_lm, ax, color='orangered')
plot_meidan_stat(cosmo.age(evo_zs_hm).value, evo_hmrs_hm, ax, lab='FLARES-HM', color='royalblue', bins=1)
plot_spread_stat(cosmo.age(evo_zs_hm).value, evo_hmrs_hm, ax, color='royalblue')

ax.set_xlabel("$t/[\mathrm{Gyr}]$")
ax.set_ylabel('$R_{1/2,*}/ [\mathrm{pkpc}]$')

handles, labels = ax.get_legend_handles_labels()
ax.legend(handles, labels)

fig.savefig("plots/HalfMassRadius_z_evolution_age.png", bbox_inches="tight")

