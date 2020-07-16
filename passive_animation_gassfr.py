#!/cosma/home/dp004/dc-rope1/.conda/envs/flares-env/bin/python
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import matplotlib
import astropy.units as u
import astropy.constants as cons
from astropy.cosmology import Planck13 as cosmo
from matplotlib.colors import LogNorm
import eagle_IO.eagle_IO as E
import seaborn as sns
import h5py
matplotlib.use('Agg')


def calc_srf(z, a_born, mass, t_bin=100):

    # Convert scale factor into redshift
    z_born = 1 / a_born - 1

    # Convert to time in Gyrs
    t = cosmo.age(z)
    t_born = cosmo.age(z_born)

    # Calculate the VR
    age = (t - t_born).to(u.Myr)

    ok = np.where(age.value <= t_bin)[0]
    if len(ok) > 0:

        # Calculate the SFR
        sfr = np.sum(mass[ok]) / (t_bin * 1e6)

    else:
        sfr = 0.0

    return sfr


def get_part_inds(halo_ids, part_ids, group_part_ids, sorted):
    """ A function to find the indexes and halo IDs associated to particles/a particle producing an array for each

    :param halo_ids:
    :param part_ids:
    :param group_part_ids:
    :return:
    """

    # Sort particle IDs if required and store an unsorted version in an array
    if sorted:
        part_ids = np.sort(part_ids)
    unsort_part_ids = np.copy(part_ids)

    # Get the indices that would sort the array (if the array is sorted this is just a range form 0-Npart)
    if sorted:
        sinds = np.arange(part_ids.size)
    else:
        sinds = np.argsort(part_ids)
        part_ids = part_ids[sinds]

    # Get the index of particles in the snapshot array from the in particles in a group array
    sorted_index = np.searchsorted(part_ids, group_part_ids)  # find the indices that would sort the array
    yindex = np.take(sinds, sorted_index, mode="raise")  # take the indices at the indices found above
    mask = unsort_part_ids[yindex] != group_part_ids  # define the mask based on these particles
    result = np.ma.array(yindex, mask=mask)  # create a mask array

    # Apply the mask to the id arrays to get halo ids and the particle indices
    part_groups = halo_ids[np.logical_not(result.mask)]  # halo ids
    parts_in_groups = result.data[np.logical_not(result.mask)]  # particle indices

    return parts_in_groups, part_groups


regions = []
for reg in range(30, 31):

    if reg < 10:
        regions.append('0' + str(reg))
    else:
        regions.append(str(reg))

snaps = ['000_z015p000', '001_z014p000', '002_z013p000', '003_z012p000', '004_z011p000', '005_z010p000',
         '006_z009p000', '007_z008p000', '008_z007p000', '009_z006p000', '010_z005p000', '011_z004p770']
snaps.reverse()

snips = ['000_z014p500', '001_z013p500', '002_z012p500', '003_z011p500', '004_z010p500', '005_z009p500',
         '006_z008p500', '007_z007p500', '008_z006p500', '009_z005p500']

# Define galaxy thresholds
ssfr_thresh = 0.5
m_thresh

snap = '011_z004p770'

cops_dict = {}

for reg in regions:

    cops_dict[reg] = []

    print(reg, snap)

    z_str = snap.split('z')[1].split('p')
    z = float(z_str[0] + '.' + z_str[1])

    path = '/cosma/home/dp004/dc-rope1/FLARES/FLARES-1/G-EAGLE_' + reg + '/data'

    try:
        app_mass = E.read_array('SUBFIND', path, snap, 'Subhalo/ApertureMeasurements/Mass/030kpc', numThreads=8)[:, 4] * 10**10
        sfrs = E.read_array('SUBFIND', path, snap, 'Subhalo/ApertureMeasurements/SFR/030kpc', numThreads=8) * 10**10
        subfind_grp_ids = E.read_array('SUBFIND', path, snap, 'Subhalo/GroupNumber', numThreads=8)
        subfind_subgrp_ids = E.read_array('SUBFIND', path, snap, 'Subhalo/SubGroupNumber', numThreads=8)
        gal_cops = E.read_array('SUBFIND', path, snap, 'Subhalo/CentreOfPotential',
                              numThreads=8)
    except ValueError:
        continue

    okinds = app_mass > 1e9
    app_mass = app_mass[okinds]
    sfrs = sfrs[okinds]
    subfind_grp_ids = subfind_grp_ids[okinds]
    subfind_subgrp_ids = subfind_subgrp_ids[okinds]
    gal_cops = gal_cops[okinds]

    # Convert IDs to float(groupNumber.SubGroupNumber) format, i.e. group 1 subgroup 11 = 1.00011
    halo_ids = np.zeros(subfind_grp_ids.size, dtype=float)
    for (ind, g), sg in zip(enumerate(subfind_grp_ids), subfind_subgrp_ids):
        halo_ids[ind] = float(str(int(g)) + '.%05d' % int(sg))

    for sfr, m, cop in zip(sfrs, app_mass, gal_cops):
        if m == 0:
            continue
        grp_ssfr = sfr / m
        if grp_ssfr < ssfr_thresh and grp_ssfr != 0:
            # print(grp_ssfr)
            cops_dict[reg].append(cop)

    print(len(cops_dict[reg]))

lim = 100 / 1000
soft = 0.001802390 / 0.6777
scale = 10 / 1000

snaps.extend(snips)

star_poss_dict = {}
gas_poss_dict = {}
sfr_dict = {}

for reg in regions:

    path = '/cosma/home/dp004/dc-rope1/FLARES/FLARES-1/G-EAGLE_' + reg + '/data'

    for snap in snaps:

        try:
            star_poss_dict[(reg, snap)] = E.read_array('SNAP', path, snap, 'PartType4/Coordinates', numThreads=8)
            gas_poss_dict[(reg, snap)] = E.read_array('SNAP', path, snap, 'PartType0/Coordinates', numThreads=8)
            # sfr_dict[(reg, snap)] = E.read_array('SNAP', path, snap, 'PartType0/StarFormationRate', numThreads=8)
        except ValueError:
            print("Error")
            continue

for reg in regions:

    for ind, cop in enumerate(cops_dict[reg]):

        for snap in snaps:

            z_str = snap.split('z')[1].split('p')
            z = float(z_str[0] + '.' + z_str[1])

            print(reg, snap, ind)

            path = '/cosma/home/dp004/dc-rope1/FLARES/FLARES-1/G-EAGLE_' + reg + '/data'

            star_poss = star_poss_dict[(reg, snap)] - cop
            gas_poss = gas_poss_dict[(reg, snap)] - cop
            # gas_sfr = sfr_dict[(reg, snap)]

            # Get only stars within the aperture
            star_okinds = np.logical_and(np.abs(star_poss[:, 0]) < lim,
                                         np.logical_and(np.abs(star_poss[:, 1]) < lim, np.abs(star_poss[:, 2]) < lim))
            gas_okinds = np.logical_and(np.abs(gas_poss[:, 0]) < lim,
                                        np.logical_and(np.abs(gas_poss[:, 1]) < lim, np.abs(gas_poss[:, 2]) < lim))
            this_star_poss = star_poss[star_okinds, :]
            this_gas_poss = gas_poss[gas_okinds, :]
            # this_gas_sfr = gas_sfr[gas_okinds]

            # Define resolution
            res = 2 * lim / soft

            # Histogram positions into images 
            Hstar, _, _ = np.histogram2d(this_star_poss[:, 0], this_star_poss[:, 1], bins=res, range=((-lim, lim), (-lim, lim)))
            Hgas, _, _ = np.histogram2d(this_gas_poss[:, 0], this_gas_poss[:, 1], bins=res, range=((-lim, lim), (-lim, lim)))
            # Hsfr, _, _ = np.histogram2d(this_gas_poss[:, 0], this_gas_poss[:, 1], bins=res, range=((-lim, lim), (-lim, lim)),
            #                             weights=this_gas_sfr)

            fig = plt.figure()
            ax1 = fig.add_subplot(121)
            ax2 = fig.add_subplot(122)
            # ax3 = fig.add_subplot(133)

            ax1.imshow(np.zeros_like(Hstar), extent=(-lim, lim, -lim, lim), cmap='Greys_r')
            ax2.imshow(np.zeros_like(Hgas), extent=(-lim, lim, -lim, lim), cmap='Greys_r')
            # ax3.imshow(np.zeros_like(Hsfr), extent=(-lim, lim, -lim, lim), cmap='Greys_r')
            
            im1 = ax1.imshow(np.log10(Hstar), cmap='Greys_r', extent=(-lim, lim, -lim, lim))
            im2 = ax2.imshow(np.log10(Hgas), cmap='plasma', extent=(-lim, lim, -lim, lim))
            # im3 = ax3.imshow(np.log10(Hsfr), cmap='magma', extent=(-lim, lim, -lim, lim))

            # Remove ticks
            ax1.tick_params(axis='both', left=False, top=False, right=False, bottom=False, labelleft=False,
                            labeltop=False, labelright=False, labelbottom=False)
            ax2.tick_params(axis='both', left=False, top=False, right=False, bottom=False, labelleft=False,
                            labeltop=False, labelright=False, labelbottom=False)
            # ax3.tick_params(axis='both', left=False, top=False, right=False, bottom=False, labelleft=False,
            #                 labeltop=False, labelright=False, labelbottom=False)

            ax1.text(0.1, 0.9, f'$z={z}$', bbox=dict(boxstyle="round,pad=0.3", fc='w', ec="k", lw=1, alpha=0.8),
                     transform=ax1.transAxes, horizontalalignment='left', fontsize=8)
            
            # Draw scale line
            right_side = lim - (lim * 0.1)
            vert = lim - (lim * 0.175)
            lab_vert = vert + (lim * 0.1) * 5 / 8
            lab_horz = right_side - scale / 2
            ax1.plot([right_side - scale, right_side], [vert, vert], color='w', linewidth=0.5)
            ax2.plot([right_side - scale, right_side], [vert, vert], color='w', linewidth=0.5)
            # ax3.plot([right_side - scale, right_side], [vert, vert], color='w', linewidth=0.5)

            # Label scale
            ax1.text(lab_horz, lab_vert, str(int(scale*1e3)) + ' ckpc', horizontalalignment='center',
                     fontsize=5, color='w')
            ax2.text(lab_horz, lab_vert, str(int(scale*1e3)) + ' ckpc', horizontalalignment='center',
                     fontsize=5, color='w')
            # ax3.text(lab_horz, lab_vert, str(int(scale*1e3)) + ' ckpc', horizontalalignment='center',
            #          fontsize=5, color='w')
            
            # Add colorbars
            cax1 = inset_axes(ax1, width="50%", height="3%", loc='lower left')
            cax2 = inset_axes(ax2, width="50%", height="3%", loc='lower left')
            # cax3 = inset_axes(ax3, width="50%", height="3%", loc='lower left')
            cbar1 = fig.colorbar(im1, cax=cax1, orientation="horizontal")
            cbar2 = fig.colorbar(im2, cax=cax2, orientation="horizontal")
            # cbar3 = fig.colorbar(im3, cax=cax3, orientation="horizontal")
            
            # Label colorbars
            cbar1.ax.set_xlabel(r'$\log_{10}(N_(\star})$', fontsize=3, color='w', labelpad=1.0)
            cbar1.ax.xaxis.set_label_position('top')
            cbar1.outline.set_edgecolor('w')
            cbar1.outline.set_linewidth(0.05)
            cbar1.ax.tick_params(axis='x', length=1, width=0.2, pad=0.01, labelsize=2, color='w', labelcolor='w')
            cbar2.ax.set_xlabel(r'$\log_{10}(N_(\mathrm{gas}})$', fontsize=3, color='w',
                                labelpad=1.0)
            cbar2.ax.xaxis.set_label_position('top')
            cbar2.outline.set_edgecolor('w')
            cbar2.outline.set_linewidth(0.05)
            cbar2.ax.tick_params(axis='x', length=1, width=0.2, pad=0.01, labelsize=2, color='w', labelcolor='w')
            # cbar3.ax.set_xlabel(r'$\log_{10}(SFR/[M_{\odot}/\mathrm{Myr}^{-1}])$', fontsize=3,
            #                     color='w', labelpad=1.0)
            # cbar3.ax.xaxis.set_label_position('top')
            # cbar3.outline.set_edgecolor('w')
            # cbar3.outline.set_linewidth(0.05)
            # cbar3.ax.tick_params(axis='x', length=1, width=0.2, pad=0.01, labelsize=2, color='w', labelcolor='w')

            fig.savefig("plots/passive_animation/passive_ani" + reg + "_" + str(ind) + "_" + snap.split("_")[0] + ".png",
                        bbox_inches='tight', dpi=300)