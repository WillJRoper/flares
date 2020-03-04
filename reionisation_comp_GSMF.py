#!/cosma/home/dp004/dc-rope1/.conda/envs/flares-env/bin/python
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib
import eagle_IO as E
import seaborn as sns
matplotlib.use('Agg')

sns.set_style('whitegrid')


def get_mass_data(path, snap, tag, group="SUBFIND", noH=True, cut_bounds=True):

    # Extract mass data
    M_dat = E.read_array(group, path, snap, tag, noH=noH)

    # If boundaries to be eliminated
    if cut_bounds:
        R_cop = E.read_array("SUBFIND", path, snap, "FOF/GroupCentreOfPotential", noH=noH)

        # Get the radius of each group
        R_cop -= np.mean(R_cop, axis=0)
        radii = np.linalg.norm(R_cop, axis=1)
        M_dat = M_dat[np.where(radii < 14 / 0.677700)]

    return M_dat

# Set up plot
fig = plt.figure(figsize=(18, 6))
gs = gridspec.GridSpec(2, 6)
gs.update(wspace=0.0, hspace=0.0)
ax1 = fig.add_subplot(gs[0, 0])
ax2 = fig.add_subplot(gs[0, 1])
ax3 = fig.add_subplot(gs[0, 2])
ax4 = fig.add_subplot(gs[1, 0])
ax5 = fig.add_subplot(gs[1, 1])
ax6 = fig.add_subplot(gs[1, 2])

snaps = ['005_z010p000', '006_z009p000', '007_z008p000', '008_z007p000', '009_z006p000', '010_z005p000']
axlims_x = []
axlims_y = []

# Extract galactic stellar masses
for snap, color, ax, (i, j) in zip(snaps, matplotlib.cm.get_cmap('plasma')(np.linspace(0, len(snaps))),
                                   [ax1, ax2, ax3, ax4, ax5, ax6], [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2)]):

    tag = "Subhalo/Stars/Mass"
    # snap = '009_z006p000'
    group = "SUBFIND"
    z = int(snap.split('_')[1][1:4])

    print(snap)

    M_11p5 = get_mass_data('/cosma7/data/dp004/dc-rope1/G-EAGLE_Reion/GEAGLE_24_AGNdT9_11p5/GEAGLE_24/data/',
                           snap, tag, group=group, noH=True, cut_bounds=False)
    print('Loaded 11p5')
    M_9p5 = get_mass_data('/cosma7/data/dp004/dc-rope1/G-EAGLE_Reion/GEAGLE_24_AGNdT9_9p5/data/',
                          snap, tag, group=group, noH=True, cut_bounds=False)
    print('Loaded 9p5')
    M_7p5 = get_mass_data('/cosma7/data/dp004/dc-rope1/G-EAGLE_Reion/GEAGLE_24_AGNdT9_7p5/data/',
                          snap, tag, group=group, noH=True, cut_bounds=False)
    print('Loaded 7p5')

    M_11p5 = M_11p5[np.where(M_11p5 != 0.0)] * 10**10
    M_9p5 = M_9p5[np.where(M_9p5 != 0.0)] * 10**10
    M_7p5 = M_7p5[np.where(M_7p5 != 0.0)] * 10**10

    bins = np.arange(4.0, 11.5, 0.4)

    interval = bins[1:] - bins[:-1]

    # Histogram the DMLJ halo masses
    H_11p5, bins = np.histogram(np.log10(M_11p5), bins=bins)
    H_9p5, _ = np.histogram(np.log10(M_9p5), bins=bins)
    H_7p5, _ = np.histogram(np.log10(M_7p5), bins=bins)

    # Compute bin centres
    bin_cents = bins[1:] - ((bins[1] - bins[0]) / 2)

    # Plot each histogram
    ax.plot(bin_cents, np.log10(H_11p5 / interval / (3.2*10**3)**3), color=color, label='$z_{\mathrm{reion}}=11.5$')
    ax.plot(bin_cents, np.log10(H_9p5 / interval / (3.2*10**3)**3), color=color, linestyle='--', label='$z_{\mathrm{reion}}=9.5$')
    ax.plot(bin_cents, np.log10(H_7p5 / interval / (3.2*10**3)**3), color=color, linestyle=':', label='$z_{\mathrm{reion}}=7.5$')

    ax.text(0.8, 0.9, f'$z={z}$', bbox=dict(boxstyle="round,pad=0.3", fc='w', ec="k", lw=1, alpha=0.8),
            transform=ax.transAxes, horizontalalignment='right', fontsize=8)

    axlims_x.extend(ax.get_xlim())
    axlims_y.extend(ax.get_ylim())

    # Label axes
    if i == 1:
        ax.set_xlabel(r'$\log_{10}(M_{\star}/M_\odot$)')
    if j == 0:
        ax.set_ylabel(r'$\log_{10}(\phi \mathrm{Mpc}^{-3} \mathrm{dex}^{-1})$')

for ax in [ax1, ax2, ax3, ax4, ax5, ax6]:

    ax.set_xlim(np.min(axlims_x), np.max(axlims_x))
    ax.set_ylim(np.min(axlims_y), np.max(axlims_y))

# Remove axis labels
ax1.tick_params(axis='x', top=False, bottom=False, labeltop=False, labelbottom=False)
ax2.tick_params(axis='both', left=False, top=False, right=False, bottom=False, labelleft=False, labeltop=False,
                labelright=False, labelbottom=False)
ax3.tick_params(axis='both', left=False, top=False, right=False, bottom=False, labelleft=False, labeltop=False,
                labelright=False, labelbottom=False)
ax5.tick_params(axis='y', left=False, right=False, labelleft=False, labelright=False)
ax6.tick_params(axis='y', left=False, right=False, labelleft=False, labelright=False)

# Get and draw legend
handles, labels = ax1.get_legend_handles_labels()
ax1.legend(handles, labels, fontsize=8, loc='lower right')

# Save figure
fig.savefig('GSMF_reion_comp.png', bbox_inches='tight', dpi=300)
