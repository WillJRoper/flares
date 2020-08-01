#!/cosma/home/dp004/dc-rope1/.conda/envs/flares-env/bin/python
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import eagle_IO.eagle_IO as E
from matplotlib.colors import LogNorm
import matplotlib.gridspec as gridspec
from scipy.stats import binned_statistic
from unyt import mh, cm, Gyr, g, Msun, Mpc
import warnings
import seaborn as sns

sns.set_style("whitegrid")
matplotlib.use('Agg')
warnings.filterwarnings('ignore')


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


def main():
    
    regions = []
    for reg in range(0, 1):
        if reg < 10:
            regions.append('0' + str(reg))
        else:
            regions.append(str(reg))
    regions.reverse()
    snaps = ['003_z012p000', '004_z011p000', '005_z010p000',
             '006_z009p000', '007_z008p000', '008_z007p000',
             '009_z006p000', '010_z005p000', '011_z004p770']
    
    gal_mean_bd = {}
    gal_mean_met = {}
    gas_mean_den = {}
    gas_mean_met = {}
    gal_ms = {}

    reg_snaps = []
    for reg in regions:

        for snap in snaps:
            
            gal_mean_bd[snap] = []
            gal_mean_met[snap] = []
            gas_mean_den[snap] = []
            gas_mean_met[snap] = []
            gal_ms[snap] = []
            
            reg_snaps.append((reg, snap))

    for reg, snap in reg_snaps:

        print(reg, snap)

        path = '/cosma/home/dp004/dc-rope1/FLARES/FLARES-1/G-EAGLE_' + reg + '/data'
    
        # Get the redshift
        z_str = snap.split('z')[1].split('p')
        z = float(z_str[0] + '.' + z_str[1])
    
        # Load all necessary arrays
        try:
            subfind_grp_ids = E.read_array('SUBFIND', path, snap, 'Subhalo/GroupNumber', numThreads=8)
            subfind_subgrp_ids = E.read_array('SUBFIND', path, snap, 'Subhalo/SubGroupNumber', numThreads=8)
            all_gal_ms = E.read_array('SUBFIND', path, snap, 'Subhalo/ApertureMeasurements/Mass/030kpc',
                                      numThreads=8) * 10**10
            all_gal_ns = E.read_array('SUBFIND', path, snap, 'Subhalo/SubLengthType',
                                      numThreads=8)
        except OSError:
            continue
        except KeyError:
            continue
    
        # Remove particles not in a subgroup
        okinds = np.logical_and(subfind_subgrp_ids != 1073741824, 
                                np.logical_and(all_gal_ms[:, 4] >= 1e8,
                                               np.logical_and(all_gal_ms[:, 1] > 0, all_gal_ms[:, 0] > 0)))
        subfind_grp_ids = subfind_grp_ids[okinds]
        subfind_subgrp_ids = subfind_subgrp_ids[okinds]
        all_gal_ms = all_gal_ms[okinds, :]

        gal_star_m = all_gal_ns[:, 0]
    
        # Convert IDs to float(groupNumber.SubGroupNumber) format, i.e. group 1 subgroup 11 = 1.00011
        halo_ids = np.zeros(subfind_grp_ids.size, dtype=float)
        for (ind, g), sg in zip(enumerate(subfind_grp_ids), subfind_subgrp_ids):
            halo_ids[ind] = float(str(int(g)) + '.%05d' % int(sg))
    
        star_halo_ids = np.copy(halo_ids)
    
        try:
            # Load data for luminosities
            gal_bd = E.read_array('PARTDATA', path, snap, 'PartType4/BirthDensity', noH=True,
                                   physicalUnits=True, numThreads=8)
            metallicities = E.read_array('PARTDATA', path, snap, 'PartType4/SmoothedMetallicity', noH=True,
                                         physicalUnits=True, numThreads=8)
            grp_ids = E.read_array('PARTDATA', path, snap, 'PartType4/GroupNumber', noH=True,
                                   physicalUnits=True, verbose=False, numThreads=8)
    
            subgrp_ids = E.read_array('PARTDATA', path, snap, 'PartType4/SubGroupNumber', noH=True,
                                      physicalUnits=True, verbose=False, numThreads=8)
    
            part_ids = E.read_array('PARTDATA', path, snap, 'PartType4/ParticleIDs', noH=True,
                                    physicalUnits=True, verbose=False, numThreads=8)
        except OSError:
            continue
        except KeyError:
            continue
    
        # A copy of this array is needed for the extraction method
        group_part_ids = np.copy(part_ids)
    
        print("There are", len(subgrp_ids), "particles")
    
        # Convert IDs to float(groupNumber.SubGroupNumber) format, i.e. group 1 subgroup 11 = 1.00011
        part_halo_ids = np.zeros(grp_ids.size, dtype=float)
        for (ind, g), sg in zip(enumerate(grp_ids), subgrp_ids):
            part_halo_ids[ind] = float(str(int(g)) + '.%05d' % int(sg))
    
        okinds = np.isin(part_halo_ids, star_halo_ids)
        part_halo_ids = part_halo_ids[okinds]
        part_ids = part_ids[okinds]
        group_part_ids = group_part_ids[okinds]
        bds = gal_bd[okinds]
        metallicities = metallicities[okinds]
    
        print("There are", len(part_halo_ids), "particles")
    
        print("Got halo IDs")
    
        parts_in_groups, part_groups = get_part_inds(part_halo_ids, part_ids, group_part_ids, False)
    
        # Produce a dictionary containing the index of particles in each halo
        halo_part_inds = {}
        for ind, grp in zip(parts_in_groups, part_groups):
            halo_part_inds.setdefault(grp, set()).update({ind})
    
        # Now the dictionary is fully populated convert values from sets to arrays for indexing
        for key, val in halo_part_inds.items():
            halo_part_inds[key] = np.array(list(val))
    
        # Get the position of each of these galaxies
        gal_bds = np.zeros_like(star_halo_ids)
        gal_mets = np.zeros_like(star_halo_ids)
        for ind, id in enumerate(star_halo_ids):
            mask = halo_part_inds[id]
            gal_bds[ind] = np.mean(bds[mask])
            gal_mets[ind] = np.mean(metallicities[mask])
    
        print('There are', len(gal_bds), 'galaxies')
        
        print('Got galaxy properties')
        
        # Get gas particle information
        gas_density = E.read_array('PARTDATA', path, snap, 'PartType0/Density', noH=True, physicalUnits=True,
                                    numThreads=8)
        gas_metallicities = E.read_array('PARTDATA', path, snap, 'PartType0/SmoothedMetallicity', noH=True,
                                         physicalUnits=True, numThreads=8)
        grp_ids = E.read_array('PARTDATA', path, snap, 'PartType0/GroupNumber', noH=True,
                               physicalUnits=True, verbose=False, numThreads=8)

        subgrp_ids = E.read_array('PARTDATA', path, snap, 'PartType0/SubGroupNumber', noH=True,
                                  physicalUnits=True, verbose=False, numThreads=8)

        part_ids = E.read_array('PARTDATA', path, snap, 'PartType0/ParticleIDs', noH=True,
                                physicalUnits=True, verbose=False, numThreads=8)
    
        # A copy of this array is needed for the extraction method
        group_part_ids = np.copy(part_ids)
    
        print("There are", len(subgrp_ids), "particles")
    
        # Convert IDs to float(groupNumber.SubGroupNumber) format, i.e. group 1 subgroup 11 = 1.00011
        part_halo_ids = np.zeros(grp_ids.size, dtype=float)
        for (ind, g), sg in zip(enumerate(grp_ids), subgrp_ids):
            part_halo_ids[ind] = float(str(int(g)) + '.%05d' % int(sg))
    
        okinds = np.isin(part_halo_ids, star_halo_ids)
        part_halo_ids = part_halo_ids[okinds]
        part_ids = part_ids[okinds]
        group_part_ids = group_part_ids[okinds]
        gas_density = gas_density[okinds]
        gas_metallicities = gas_metallicities[okinds]
        
        print("There are", len(part_halo_ids), "particles")
    
        print("Got halo IDs")
    
        parts_in_groups, part_groups = get_part_inds(part_halo_ids, part_ids, group_part_ids, False)
    
        # Produce a dictionary containing the index of particles in each halo
        halo_part_inds = {}
        for ind, grp in zip(parts_in_groups, part_groups):
            halo_part_inds.setdefault(grp, set()).update({ind})
    
        # Now the dictionary is fully populated convert values from sets to arrays for indexing
        for key, val in halo_part_inds.items():
            halo_part_inds[key] = np.array(list(val))
    
        # Get the position of each of these galaxies
        gas_dens = np.zeros_like(star_halo_ids)
        gas_mets = np.zeros_like(star_halo_ids)
        for ind, id in enumerate(star_halo_ids):
            mask = halo_part_inds[id]
            gas_dens[ind] = np.mean(gas_density[mask])
            gas_mets[ind] = np.mean(gas_metallicities[mask])
    
        print('Got gas properties')

        gal_mean_bd[snap].extend(gal_bds)
        gal_mean_met[snap].extend(gal_mets)
        gas_mean_den[snap].extend(gas_dens)
        gas_mean_met[snap].extend(gas_mets)
        gal_ms[snap].extend(gal_star_m)

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

        star_bd = np.array(gal_mean_bd[snap]) * (10**10 * Msun / Mpc ** 3 / mh).to(1 / cm ** 3).value
        gas_den = np.array(gas_mean_den[snap]) * (10**10 * Msun / Mpc ** 3 / mh).to(1 / cm ** 3).value
        ms = np.array(gal_ms[snap])
        print(ms)

        # try:
        #     cbar = ax.hexbin(ms, gas_den, gridsize=50, mincnt=1, xscale='log', yscale='log',
        #                      norm=LogNorm(), linewidths=0.2, cmap='plasma', alpha=0.4)
        #     plot_meidan_stat(ms, gas_den, ax, lab='Gas Median', color='g')
        #     cbar = ax.hexbin(ms, star_bd, gridsize=50, mincnt=1, xscale='log', yscale='log',
        #                      norm=LogNorm(), linewidths=0.2, cmap='viridis', alpha=0.4)
        #     plot_meidan_stat(ms, star_bd, ax, lab='Stellar Median', color='r', ls='dashed')
        # except ValueError:
        #     continue
        # except TypeError:
        #     continue

        cbar = ax.hexbin(ms, gas_den, gridsize=50, mincnt=1, xscale='log', yscale='log',
                         norm=LogNorm(), linewidths=0.2, cmap='plasma', alpha=0.4)
        plot_meidan_stat(ms, gas_den, ax, lab='Gas Median', color='g')
        cbar = ax.hexbin(ms, star_bd, gridsize=50, mincnt=1, xscale='log', yscale='log',
                         norm=LogNorm(), linewidths=0.2, cmap='viridis', alpha=0.4)
        plot_meidan_stat(ms, star_bd, ax, lab='Stellar Median', color='r', ls='dashed')

        ax.text(0.9, 0.1, f'$z={z}$', bbox=dict(boxstyle="round,pad=0.3", fc='w', ec="k", lw=1, alpha=0.8),
                transform=ax.transAxes, horizontalalignment='right', fontsize=8)

        axlims_x.extend(ax.get_xlim())
        axlims_y.extend(ax.get_ylim())

        # Label axes
        if i == 2:
            ax.set_xlabel(r'$M_{\star}/M_\odot$')
        if j == 0:
            ax.set_ylabel(r'$<n_H>$ [cm$^{-3}$]')

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

    handles, labels = ax9.get_legend_handles_labels()
    ax1.legend(handles, labels)

    fig.savefig('plots/Stellar_gas_density_comp_gasnpart.png', bbox_inches='tight')

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

        star_met = np.array(gal_mean_met[snap])
        gas_met = np.array(gas_mean_met[snap])
        ms = np.array(gal_ms[snap])

        try:
            cbar = ax.hexbin(ms, gas_met, gridsize=50, mincnt=1, xscale='log', yscale='log',
                             norm=LogNorm(), linewidths=0.2, cmap='plasma', alpha=0.4)
            plot_meidan_stat(ms, gas_met, ax, lab='Gas Median', color='g')
            cbar = ax.hexbin(ms, star_met, gridsize=50, mincnt=1, xscale='log', yscale='log',
                             norm=LogNorm(), linewidths=0.2, cmap='viridis', alpha=0.4)
            plot_meidan_stat(ms, star_met, ax, lab='Stellar Median', color='r', ls='dashed')
        except ValueError:
            continue
        except TypeError:
            continue

        ax.text(0.9, 0.1, f'$z={z}$', bbox=dict(boxstyle="round,pad=0.3", fc='w', ec="k", lw=1, alpha=0.8),
                transform=ax.transAxes, horizontalalignment='right', fontsize=8)

        axlims_x.extend(ax.get_xlim())
        axlims_y.extend(ax.get_ylim())

        # Label axes
        if i == 2:
            ax.set_xlabel(r'$M_{\star}/M_\odot$')
        if j == 0:
            ax.set_ylabel(r'$Z$')

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

    handles, labels = ax9.get_legend_handles_labels()
    ax1.legend(handles, labels)

    fig.savefig('plots/Stellar_gas_metallicity_comp_gasnpart.png', bbox_inches='tight')

    plt.close(fig)


main()
