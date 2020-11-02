#!/cosma/home/dp004/dc-rope1/.conda/envs/flares-env/bin/python
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import astropy.units as u
from scipy.stats import binned_statistic
from matplotlib.colors import LogNorm
import matplotlib.gridspec as gridspec
import eagle_IO.eagle_IO as E
import seaborn as sns
import pickle
import itertools
matplotlib.use('Agg')

sns.set_style('whitegrid')


snaps = ['004_z008p075', '008_z005p037', '010_z003p984',
         '012_z003p017', '013_z002p478', '018_z001p259',
         '019_z001p004', '020_z000p865', '027_z000p101']
path = "/cosma7/data/Eagle/ScienceRuns/Planck1/L0025N0376/" \
                       "PE/EagleVariation_NoZDEPSFthresh/data"
path2 = '/cosma7/data//Eagle/ScienceRuns/Planck1/L0100N1504/PE/REFERENCE/data'
axlims_x = []
axlims_y = []

# # Define comoving softening length in kpc
# csoft = 0.001802390/0.677*1e3


def plot_meidan_stat(xs, ys, ax, lab, color, bins=None, ls='-'):

    if bins == None:
        bin = np.logspace(np.log10(xs.min()), np.log10(xs.max()), 20)
    else:
        bin = bins

    # Compute binned statistics
    y_stat, binedges, bin_ind = binned_statistic(xs, ys, statistic='median', bins=bin)

    # Compute bincentres
    bin_wid = binedges[1] - binedges[0]
    bin_cents = binedges[1:] - bin_wid / 2

    okinds = np.logical_and(~np.isnan(bin_cents), ~np.isnan(y_stat))

    ax.plot(bin_cents[okinds], y_stat[okinds], color=color, linestyle=ls, label=lab)


half_mass_rads_dict = {}
xaxis_dict = {}

for snap in snaps:

    print(snap)

    half_mass_rads_dict[snap] = E.read_array('SUBFIND', path, snap, 'Subhalo/HalfMassRad', noH=True,
                                                  numThreads=8)[:, 4]
    xaxis_dict[snap] = E.read_array('SUBFIND', path, snap, 'Subhalo/ApertureMeasurements/Mass/030kpc',
                                         noH=True, numThreads=8)[:, 4] * 10**10

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

    if z <= 2.8:
        soft = 0.000474390 / 0.6777
    else:
        soft = 0.001802390 / (0.6777 * (1 + z))

    xs = xaxis_dict[snap]
    half_mass_rads_plt = half_mass_rads_dict[snap]
    
    xs_plt = xs[half_mass_rads_plt > 0]
    half_mass_rads_plt = half_mass_rads_plt[half_mass_rads_plt > 0]
    half_mass_rads_plt = half_mass_rads_plt[xs_plt > 1e8]
    xs_plt = xs_plt[xs_plt > 1e8]
    
    # cbar = ax.hexbin(xs_plt, half_mass_rads_plt / (csoft / (1 + z)), gridsize=100, mincnt=1, xscale='log', yscale='log', norm=LogNorm(),
    #                  linewidths=0.2, cmap='viridis')  # uncomment to include softening
    cbar = ax.hexbin(xs_plt, half_mass_rads_plt / soft, gridsize=100, mincnt=1, xscale='log', yscale='log',
                     norm=LogNorm(), linewidths=0.2, cmap='viridis', alpha=0.7)
    plot_meidan_stat(xs_plt, half_mass_rads_plt / soft, ax, lab='noZSFthresh', color='r')

    ax.text(0.8, 0.9, f'$z={z}$', bbox=dict(boxstyle="round,pad=0.3", fc='w', ec="k", lw=1, alpha=0.8),
            transform=ax.transAxes, horizontalalignment='right', fontsize=8)

    axlims_x.extend(ax.get_xlim())
    axlims_y.extend(ax.get_ylim())

    # Label axes
    if i == 2:
        ax.set_xlabel(r'$M_{\star}/M_\odot$')
    if j == 0:
        # ax.set_ylabel('$R_{1/2}/\epsilon$')  # uncomment to include softening
        ax.set_ylabel('$R_{1/2}/\epsilon$')

half_mass_rads_dict = {}
xaxis_dict = {}

for snap in snaps:

    print(snap)

    half_mass_rads_dict[snap] = E.read_array('SUBFIND', path2, snap, 'Subhalo/HalfMassRad', noH=True,
                                                  numThreads=8)[:, 4]
    xaxis_dict[snap] = E.read_array('SUBFIND', path2, snap, 'Subhalo/ApertureMeasurements/Mass/030kpc',
                                         noH=True, numThreads=8)[:, 4] * 10**10

for ax, snap, (i, j) in zip([ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8, ax9], snaps,
                            [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2), (2, 0), (2, 1), (2, 2)]):

    z_str = snap.split('z')[1].split('p')
    z = float(z_str[0] + '.' + z_str[1])

    if z <= 2.8:
        soft = 0.000474390 / 0.6777
    else:
        soft = 0.001802390 / (0.6777 * (1 + z))

    xs = xaxis_dict[snap]
    half_mass_rads_plt = half_mass_rads_dict[snap]

    xs_plt = xs[half_mass_rads_plt > 0]
    half_mass_rads_plt = half_mass_rads_plt[half_mass_rads_plt > 0]
    half_mass_rads_plt = half_mass_rads_plt[xs_plt > 1e8]
    xs_plt = xs_plt[xs_plt > 1e8]

    # cbar = ax.hexbin(xs_plt, half_mass_rads_plt / (csoft / (1 + z)), gridsize=100, mincnt=1, xscale='log', yscale='log', norm=LogNorm(),
    #                  linewidths=0.2, cmap='viridis')  # uncomment to include softening
    # cbar = ax.hexbin(xs_plt, half_mass_rads_plt / soft, gridsize=100, mincnt=1, xscale='log', yscale='log',
    #                  norm=LogNorm(), linewidths=0.2, cmap='plasma', alpha=0.7)
    plot_meidan_stat(xs_plt, half_mass_rads_plt / soft, ax, lab='REF', color='y', ls='--')

    ax.text(0.8, 0.9, f'$z={z}$', bbox=dict(boxstyle="round,pad=0.3", fc='w', ec="k", lw=1, alpha=0.8),
            transform=ax.transAxes, horizontalalignment='right', fontsize=8)

    axlims_x.extend(ax.get_xlim())
    axlims_y.extend(ax.get_ylim())

    # Label axes
    if i == 2:
        ax.set_xlabel(r'$M_{\star}/M_\odot$')
    if j == 0:
        # ax.set_ylabel('$R_{1/2}/\epsilon$')  # uncomment to include softening
        ax.set_ylabel('$R_{1/2}/\epsilon$')

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

handles, labels = ax1.get_legend_handles_labels()
ax1.legend(handles, labels, loc='upper left')

fig.savefig('plots/HalfMassRadius_all_snaps_noZSFthresh.png',
            bbox_inches='tight')

plt.close(fig)
