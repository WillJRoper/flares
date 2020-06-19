#!/cosma/home/dp004/dc-rope1/.conda/envs/flares-env/bin/python
import numpy as np
import h5py
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LogNorm
import matplotlib.gridspec as gridspec


sns.set_context("paper")
sns.set_style('whitegrid')


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


regions = []
for reg in range(0, 40):
    if reg < 10:
        regions.append('0' + str(reg))
    else:
        regions.append(str(reg))

snaps = ['000_z015p000', '001_z014p000', '002_z013p000', '003_z012p000', '004_z011p000', '005_z010p000',
         '006_z009p000', '007_z008p000', '008_z007p000', '009_z006p000', '010_z005p000', '011_z004p770']

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
        hdf = h5py.File(hdfpath + 'ObsWebbLumins_' + snap + '.hdf5', 'r')

        lumins = hdf[f]['Aperture_Luminosity_30kpc'][:, 0]
        hlrs = hdf[f]['half_lift_rad'][:, 0]

        hlr_dict[snap].extend(hlrs)
        lumin_dict[snap].extend(lumins)
    except OSError:
        continue

# Define comoving softening length in kpc
csoft = 0.001802390 / 0.6777

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

    print(hlrs)
    print(lumins)

    okinds = np.logical_and(hlrs != 0, lumins != 0)
    lumins = lumins[okinds]
    hlrs = hlrs[okinds]

    cbar = ax.hexbin(lumins, hlrs / (csoft / (1 + z)), gridsize=100, mincnt=1, xscale='log', yscale='log',
                     norm=LogNorm(), linewidths=0.2, cmap='viridis')
    plot_meidan_stat(lumins, hlrs / (csoft / (1 + z)), ax, lab='REF', color='r')

    ax.text(0.8, 0.9, f'$z={z}$', bbox=dict(boxstyle="round,pad=0.3", fc='w', ec="k", lw=1, alpha=0.8),
            transform=ax.transAxes, horizontalalignment='right', fontsize=8)

    axlims_x.extend(ax.get_xlim())
    axlims_y.extend(ax.get_ylim())

    # Label axes
    if i == 2:
        ax.set_xlabel(r'$L_{FUV}/$ [nJy]')
    if j == 0:
        ax.set_ylabel('$R_{1/2}/\epsilon$')

for ax in [ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8, ax9]:
    ax.set_xlim(np.min(axlims_x), np.max(axlims_x))
    ax.set_ylim(np.min(axlims_y), np.max(axlims_y))

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

fig.savefig('plots/HalfLightRadiusFUV.png', bbox_inches='tight')

plt.close(fig)

