#!/cosma/home/dp004/dc-rope1/.conda/envs/flares-env/bin/python
import numpy as np
import eagle_IO.eagle_IO as E
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import matplotlib
matplotlib.use('Agg')
import seaborn as sns


sns.set_style("white")

# reg = "00"
# snaps = ['005_z010p000', '006_z009p000', '007_z008p000', '008_z007p000',
#          '009_z006p000', '010_z005p000']
# grps = [23, 30, 2, 5, 8, 3]
# subgrps = [0, 0, 1, 1, 1, 9]
# sec_grps = [None, 30, 2, 5, 8, 3]
# sec_subgrps = [None, 1, 0, 0, 0, 1]
# path = '/cosma/home/dp004/dc-rope1/FLARES/FLARES-1/G-EAGLE_' + reg + '/data/'

reg = "00"
snaps = ['005_z010p000', '006_z009p000', '007_z008p000', '008_z007p000',
         '009_z006p000', '010_z005p000']
snaps.reverse()
grps = [211, 2811, None, None, None, None]
subgrps = [0, 0, None, None, None, None]
sec_grps = [None, None, None, None, None, None]
sec_subgrps = [None, None, None, None, None, None]
path = '/cosma/home/dp004/dc-rope1/FLARES/FLARES-1/G-EAGLE_' + reg + '/data/'

# Set up images
width = 750
soft = 0.001802390 / 0.6777 * 1e3
scale = 50

for num, snap, grp, subgrp, secgrp, secsubgrp in zip(range(len(snaps)), snaps, grps, subgrps, sec_grps, sec_subgrps):

    z_str = snap.split('z')[1].split('p')
    z = float(z_str[0] + '.' + z_str[1])

    # Load all necessary arrays
    subfind_grp_ids = E.read_array('SUBFIND', path, snap, 'Subhalo/GroupNumber', numThreads=8)
    subfind_subgrp_ids = E.read_array('SUBFIND', path, snap, 'Subhalo/SubGroupNumber', numThreads=8)
    gal_cops = E.read_array('SUBFIND', path, snap, 'Subhalo/CentreOfPotential', noH=True, numThreads=8) * 1e3

    # Get cop
    if num == 0:
        cop = gal_cops[np.logical_and(subfind_grp_ids == grp, subfind_subgrp_ids == subgrp)]
    if subgrp != None:
        current_cop = gal_cops[np.logical_and(subfind_grp_ids == grp, subfind_subgrp_ids == subgrp)]

    print(current_cop)

    # Set up figure
    fig = plt.figure(figsize=(4, 3))
    gs = gridspec.GridSpec(3, 4)
    gs.update(wspace=0.0, hspace=0.0)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[0, 2])
    ax4 = fig.add_subplot(gs[0, 3])
    ax5 = fig.add_subplot(gs[1, 0])
    ax6 = fig.add_subplot(gs[1, 1])
    ax7 = fig.add_subplot(gs[1, 2])
    ax8 = fig.add_subplot(gs[1, 3])
    ax9 = fig.add_subplot(gs[2, 0])
    ax10 = fig.add_subplot(gs[2, 1])
    ax11 = fig.add_subplot(gs[2, 2])
    ax12 = fig.add_subplot(gs[2, 3])

    all_parts_poss_gal = []
    all_parts_poss_secgal = []
    all_parts_poss = []

    axes = [[ax1, ax2, ax3, ax4], [ax5, ax6, ax7, ax8], [ax9, ax10, ax11, ax12]]

    for ipart_type, part_type in enumerate(["0", "1", "4"]):

        all_poss = E.read_array('PARTDATA', path, snap, 'PartType' + str(part_type) + '/Coordinates', noH=True,
                                numThreads=8) * 1e3 - cop

        grp_ids = E.read_array('PARTDATA', path, snap, 'PartType' + str(part_type) + '/GroupNumber',
                               verbose=False, numThreads=8)
        subgrp_ids = E.read_array('PARTDATA', path, snap, 'PartType' + str(part_type) + '/SubGroupNumber',
                                  verbose=False, numThreads=8)

        okinds = np.logical_and(np.abs(all_poss[:, 0]) < width,
                                np.logical_and(np.abs(all_poss[:, 1]) < width,
                                               np.abs(all_poss[:, 2]) < width))
        img_poss = all_poss[okinds, :]

        if subgrp != None:
            okinds = np.logical_and(grp_ids == grp, subgrp_ids == subgrp)
            gal_poss = all_poss[okinds, :]
            all_parts_poss_gal.extend(gal_poss)

        if secgrp != None:
            okinds = np.logical_and(grp_ids == secgrp, subgrp_ids == secsubgrp)
            secgal_poss = all_poss[okinds, :]
            all_parts_poss_secgal.extend(secgal_poss)

        all_parts_poss.extend(img_poss)


        for row, (i, j) in enumerate(((0, 1), (1, 2), (0, 2))):

            H, _, _ = np.histogram2d(img_poss[:, i], img_poss[:, j], bins=int(width / soft),
                                                       range=((-width, width), (-width, width)))

            axes[row][ipart_type].imshow(np.zeros_like(H), extent=(-width, width, -width, width),
                                         cmap='Greys_r')
            axes[row][ipart_type].imshow(np.log10(H), extent=(-width, width, -width, width),
                                         cmap='Greys_r')

            if secgrp != None:
                H, _, _ = np.histogram2d(secgal_poss[:, i], secgal_poss[:, j], bins=int(width / soft),
                                                           range=((-width, width), (-width, width)))

                axes[row][ipart_type].imshow(np.log10(H), extent=(-width, width, -width, width),
                                             cmap='plasma')
            if subgrp != None:
                H, _, _ = np.histogram2d(gal_poss[:, i], gal_poss[:, j], bins=int(width / soft),
                                                           range=((-width, width), (-width, width)))

                axes[row][ipart_type].imshow(np.log10(H), extent=(-width, width, -width, width),
                                             cmap='viridis')

            # Draw scale line
            right_side = width - (width * 0.25)
            vert = width - (width * 0.25)
            lab_vert = vert + (width * 0.1) * 5 / 8
            lab_horz = right_side - scale / 2
            axes[row][ipart_type].plot([right_side - scale, right_side], [vert, vert], color='w', linewidth=0.75)
            axes[row][ipart_type].text(lab_horz, lab_vert, str(int(scale)) + ' ckpc', horizontalalignment='center',
                                       fontsize=4, color='w')

    all_parts_poss = np.array(all_parts_poss)
    all_parts_poss_gal = np.array(all_parts_poss_gal)
    all_parts_poss_secgal = np.array(all_parts_poss_secgal)

    for row, (i, j) in enumerate(((0, 1), (1, 2), (0, 2))):

        H, _, _ = np.histogram2d(all_parts_poss[:, i], all_parts_poss[:, j], bins=int(width / soft),
                                                   range=((-width, width), (-width, width)))

        axes[row][3].imshow(np.zeros_like(H), extent=(-width, width, -width, width),
                                     cmap='Greys_r')
        axes[row][3].imshow(np.log10(H), extent=(-width, width, -width, width),
                                     cmap='Greys_r')

        if secgrp != None:
            H, _, _ = np.histogram2d(all_parts_poss_secgal[:, i], all_parts_poss_secgal[:, j], bins=int(width / soft),
                                     range=((-width, width), (-width, width)))

            axes[row][3].imshow(np.log10(H), extent=(-width, width, -width, width),
                                         cmap='plasma')
        if subgrp != None:
            H, _, _ = np.histogram2d(all_parts_poss_gal[:, i], all_parts_poss_gal[:, j], bins=int(width / soft),
                                                       range=((-width, width), (-width, width)))

            axes[row][3].imshow(np.log10(H), extent=(-width, width, -width, width),
                                         cmap='viridis')

        # Draw scale line
        right_side = width - (width * 0.25)
        vert = width - (width * 0.25)
        lab_vert = vert + (width * 0.1) * 5 / 8
        lab_horz = right_side - scale / 2
        axes[row][3].plot([right_side - scale, right_side], [vert, vert], color='w', linewidth=0.75)
        axes[row][3].text(lab_horz, lab_vert, str(int(scale)) + ' ckpc', horizontalalignment='center',
                          fontsize=4, color='w')

    axes = np.array(axes).flatten()
    for ax in axes:
        # Remove ticks
        ax.tick_params(axis='both', left=False, top=False, right=False, bottom=False, labelleft=False,
                        labeltop=False, labelright=False, labelbottom=False)
        ax.text(0.1, 0.1, f'$z={z}$', bbox=dict(boxstyle="round,pad=0.3", fc='w', ec="k", lw=1, alpha=0.8),
                transform=ax.transAxes, horizontalalignment='left', fontsize=4)
    # # Add colorbars
    # cax1 = inset_axes(ax1, width="50%", height="3%", loc='lower left')
    # cbar1 = fig.colorbar(im1, cax=cax1, orientation="horizontal")
    #
    # # Label colorbars
    # cbar1.ax.set_xlabel(r'$\log_{10}(M_{\star}/M_{\odot})$', fontsize=2, color='w', labelpad=1.0)
    # cbar1.ax.xaxis.set_label_position('top')
    # cbar1.outline.set_edgecolor('w')
    # cbar1.outline.set_linewidth(0.05)
    # cbar1.ax.tick_params(axis='x', length=1, width=0.2, pad=0.01, labelsize=2, color='w', labelcolor='w')

    ax1.set_title("Gas", fontsize=6)
    ax2.set_title("Dark Matter", fontsize=6)
    ax3.set_title("Stellar", fontsize=6)
    ax4.set_title("All", fontsize=6)

    fig.savefig('plots/passive_animation/passive_ani_' + str(grps[0]) + '_' + str(subgrps[0]) + '_%03d.png' % num,
                bbox_inches='tight', dpi=600)

plt.close(fig)
