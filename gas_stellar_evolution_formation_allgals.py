#!/cosma/home/dp004/dc-rope1/.conda/envs/flares-env/bin/python
import matplotlib as ml
ml.use('Agg')
import numpy as np
import matplotlib.gridspec as gridspec
from scipy.stats import binned_statistic
from astropy.cosmology import Planck13 as cosmo
import matplotlib.pyplot as plt
import eagle_IO.eagle_IO as E
from matplotlib.colors import LogNorm
import seaborn as sns
import numba as nb


sns.set_style('whitegrid')


def plot_median_stat(xs, ys, ax, lab, color, bins=None, ls='-', func="median"):

    if bins == None:
        bin = np.logspace(np.log10(xs.min()), np.log10(xs.max()), 20)
    else:
        bin = bins

    # Compute binned statistics
    y_stat, binedges, bin_ind = binned_statistic(xs, ys, statistic=func, bins=bin)

    # Compute bincentres
    bin_wid = binedges[1] - binedges[0]
    bin_cents = binedges[1:] - bin_wid / 2

    okinds = np.logical_and(~np.isnan(bin_cents), ~np.isnan(y_stat))

    ax.plot(bin_cents[okinds], y_stat[okinds], color=color, linestyle=ls, label=lab)


def plot_spread_stat(xs, ys, ax, color, bins=None):

    if bins == None:
        bin = np.logspace(np.log10(xs.min()), np.log10(xs.max()), 20)
    else:
        bin = bins

    # Compute binned statistics
    y_stat_16, binedges, bin_ind = binned_statistic(xs, ys, statistic=lambda y: np.percentile(y, 16), bins=bin)
    y_stat_84, binedges, bin_ind = binned_statistic(xs, ys, statistic=lambda y: np.percentile(y, 84), bins=bin)

    # Compute bincentres
    bin_wid = binedges[1] - binedges[0]
    bin_cents = binedges[1:] - bin_wid / 2

    okinds = np.logical_and(~np.isnan(bin_cents), np.logical_and(~np.isnan(y_stat_16), ~np.isnan(y_stat_84)))

    ax.fill_between(bin_cents[okinds], y_stat_16[okinds], y_stat_84[okinds], color=color, alpha=0.4)


@nb.njit(nogil=True, parallel=True)
def calc_3drad(poss):

    # Get galaxy particle indices
    rs = np.sqrt(poss[:, 0]**2 + poss[:, 1]**2 + poss[:, 2]**2)

    return rs


def get_linked_halo_data(all_linked_halos, start_ind, nlinked_halos):
    """ A helper function for extracting a halo's linked halos
        (i.e. progenitors and descendants)

    :param all_linked_halos: Array containing all progenitors and descendants.
    :type all_linked_halos: float[N_linked halos]
    :param start_ind: The start index for this halos progenitors or descendents elements in all_linked_halos
    :type start_ind: int
    :param nlinked_halos: The number of progenitors or descendents (linked halos) the halo in question has
    :type nlinked_halos: int
    :return:
    """

    return all_linked_halos[start_ind: start_ind + nlinked_halos]


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


def get_main_branch(z0halo, data_dict):
    """ A funciton which traverses a graph including all linked halos.

    :param tree_data: The tree data dictionary produced by the Merger Graph.
    :param z0halo: The halo ID of a z=0 halo for which the graph is desired.

    :return: graph_dict: The dictionary containing the graph. Each key is the snapshot ID and
             the value is a list of halos in this snapshot of the graph.
             massgrowth: The mass history of the graph.
             tree: The dictionary containing the tree. Each key is the snapshot ID and
             the value is a list of halos in this snapshot of the tree.
             main_growth: The mass history of the main branch.
    """

    # Create snapshot list in reverse order (present day to past) for the progenitor searching loop
    snaplist = ['000_z015p000', '001_z014p000', '002_z013p000', '003_z012p000',
                '004_z011p000', '005_z010p000', '006_z009p000', '007_z008p000',
                '008_z007p000', '009_z006p000', '010_z005p000', '011_z004p770']
    rev_snaplist = list(reversed(snaplist))

    # Initialise dictionary instances
    graph_dict = {}

    # Initialise the halo's set for tree walking
    halos = {(z0halo, '011_z004p770')}

    # Initialise entries in the graph dictionary
    for snap in snaplist:
        graph_dict[snap] = set()

    # Initialise the graph dictionary with the present day halo as the first entry
    graph_dict['011_z004p770'] = halos

    # Loop over snapshots and progenitor snapshots
    for i in range(len(snaplist) - 1):

        snap = rev_snaplist[i]
        prog_snap = rev_snaplist[i + 1]

        # Assign the halos variable for the next stage of the tree
        halos = graph_dict[snap]

        # Loop over halos in this snapshot
        for halo in halos:

            # Get the progenitors
            try:
                start_ind = \
                data_dict['prog_start_index'][snap][data_dict['mega'][snap][data_dict['sim'][snap] == halo[0]]][0]
                nprog = data_dict['nprogs'][snap][data_dict['mega'][snap][data_dict['sim'][snap] == halo[0]]][0]
            except IndexError:
                print(halo, "does not appear in the graph arrays")
                continue
            if nprog == 0:
                continue
            these_progs = get_linked_halo_data(data_dict['progs'][snap], start_ind, nprog)

            # Assign progenitors using a tuple to keep track of the snapshot ID
            # in addition to the halo ID
            graph_dict[prog_snap].update({(these_progs[0], prog_snap)})

    # Get the number of particle in each halo and sort based on mass
    for snap in graph_dict:

        if len(graph_dict[snap]) == 0:
            continue

        # Convert entry to an array for sorting
        graph_dict[snap] = np.array([halo[0] for halo in graph_dict[snap]], dtype=float)

    return graph_dict


def get_graph(z0halo, data_dict):
    """ A funciton which traverses a graph including all linked halos.

    :param tree_data: The tree data dictionary produced by the Merger Graph.
    :param z0halo: The halo ID of a z=0 halo for which the graph is desired.

    :return: graph_dict: The dictionary containing the graph. Each key is the snapshot ID and
             the value is a list of halos in this snapshot of the graph.
             massgrowth: The mass history of the graph.
             tree: The dictionary containing the tree. Each key is the snapshot ID and
             the value is a list of halos in this snapshot of the tree.
             main_growth: The mass history of the main branch.
    """

    # Create snapshot list in reverse order (present day to past) for the progenitor searching loop
    snaplist = ['000_z015p000', '001_z014p000', '002_z013p000', '003_z012p000',
                '004_z011p000', '005_z010p000', '006_z009p000', '007_z008p000',
                '008_z007p000', '009_z006p000', '010_z005p000', '011_z004p770']
    rev_snaplist = list(reversed(snaplist))

    # Initialise dictionary instances
    graph_dict = {}
    mass_dict = {}

    # Initialise the halo's set for tree walking
    halos = {(z0halo, '011_z004p770')}

    # Initialise entries in the graph dictionary
    for snap in snaplist:
        graph_dict[snap] = set()

    # Initialise the graph dictionary with the present day halo as the first entry
    graph_dict['011_z004p770'] = halos

    # Initialise the set of new found halos used to loop until no new halos are found
    new_halos = halos

    # Initialise the set of found halos used to check if the halo has been found or not
    found_halos = set()

    # =============== Progenitors ===============

    count = 0
    # Loop until no new halos are found
    while len(new_halos) != 0:

        print(count)
        count += 1

        # Overwrite the last set of new_halos
        new_halos = set()

        # =============== Progenitors ===============

        # Loop over snapshots and progenitor snapshots
        for i in range(len(snaplist) - 1):

            snap = rev_snaplist[i]
            prog_snap = rev_snaplist[i + 1]

            # Assign the halos variable for the next stage of the tree
            halos = graph_dict[snap]

            # Loop over halos in this snapshot
            for halo in halos:

                # Get the progenitors
                try:
                    start_ind = data_dict['prog_start_index'][snap][data_dict['mega'][snap][data_dict['sim'][snap] == halo[0]]][0]
                    nprog = data_dict['nprogs'][snap][data_dict['mega'][snap][data_dict['sim'][snap] == halo[0]]][0]
                except IndexError:
                    print(halo, "does not appear in the graph arrays")
                    continue
                if nprog == 0:
                    continue
                these_progs = get_linked_halo_data(data_dict['progs'][snap], start_ind, nprog)

                # Assign progenitors using a tuple to keep track of the snapshot ID
                # in addition to the halo ID
                graph_dict[prog_snap].update({(p, prog_snap) for p in these_progs})

            # Add any new halos not found in found halos to the new halos set
            new_halos.update(graph_dict[prog_snap] - found_halos)

        # =============== Descendants ===============

        # Loop over halos found during the progenitor step
        for i in range(len(snaplist) - 1):

            snap = snaplist[i]
            desc_snap = snaplist[i + 1]

            # Assign the halos variable for the next stage of the tree
            halos = graph_dict[snap]

            # Loop over the progenitor halos
            for halo in halos:

                # Get the descendants
                try:
                    start_ind = data_dict['desc_start_index'][snap][data_dict['mega'][snap][data_dict['sim'][snap] == halo[0]]][0]
                    ndesc = data_dict['ndescs'][snap][data_dict['mega'][snap][data_dict['sim'][snap] == halo[0]]][0]
                except IndexError:
                    print(halo, "does not appear in the graph arrays")
                    continue
                if ndesc == 0:
                    continue
                these_descs = get_linked_halo_data(data_dict['descs'][snap], start_ind, ndesc)

                # Load descendants adding the snapshot * 100000 to keep track of the snapshot ID
                # in addition to the halo ID
                graph_dict[desc_snap].update({(d, desc_snap) for d in these_descs})

            # Redefine the new halos set to have any new halos not found in found halos
            new_halos.update(graph_dict[desc_snap] - found_halos)

        # Add the new_halos to the found halos set
        found_halos.update(new_halos)

    # Get the number of particle in each halo and sort based on mass
    for snap in graph_dict:

        if len(graph_dict[snap]) == 0:
            continue

        # Convert entry to an array for sorting
        graph_dict[snap] = np.array([halo[0] for halo in graph_dict[snap]], dtype=float)

    return graph_dict


def forest_worker(z0halo, data_dict):

    # Get the forest with this halo at it's root
    forest_dict = get_main_branch(z0halo, data_dict)

    print('Halo ' + str(z0halo) + '\'s Forest extracted...')

    return forest_dict


def get_evolution(path, snaps):

    # Set up dictionaries to store data
    gas_hmrs = {}
    star_hmrs = {}
    form = {}

    for snap in snaps:

        z_str = snap.split('z')[1].split('p')
        z = float(z_str[0] + '.' + z_str[1])

        # Define comoving softening length in kpc
        soft = 0.001802390 / 0.6777 * 1 / (1 + z)

        # Get halo IDs and halo data
        try:
            subgrp_ids = E.read_array('SUBFIND', path, snap, 'Subhalo/SubGroupNumber', numThreads=8)
            grp_ids = E.read_array('SUBFIND', path, snap, 'Subhalo/GroupNumber', numThreads=8)
            gal_hmrs = E.read_array('SUBFIND', path, snap, 'Subhalo/HalfMassRad', noH=True,
                                    physicalUnits=True, numThreads=8)
            gal_ms = E.read_array('SUBFIND', path, snap, 'Subhalo/ApertureMeasurements/Mass/030kpc',
                                  noH=True, physicalUnits=True, numThreads=8) * 10 ** 10
        except OSError:
            continue
        except ValueError:
            continue

        # Remove particles not associated to a subgroup
        okinds = np.logical_and(subgrp_ids != 1073741824,
                                np.logical_and(gal_ms[:, 1] != 0, gal_ms[:, 4] > 10**8))
        gal_hmrs = gal_hmrs[okinds]
        grp_ids = grp_ids[okinds]
        subgrp_ids = subgrp_ids[okinds]
        halo_ids = np.zeros(grp_ids.size, dtype=float)
        for (ind, g), sg in zip(enumerate(grp_ids), subgrp_ids):
            halo_ids[ind] = float(str(int(g)) + '.%05d' % int(sg))

        star_halo_ids = np.copy(halo_ids)

        # Load data for luminosities
        try:
            form_a = E.read_array('PARTDATA', path, snap, 'PartType4/StellarFormationTime', noH=True,
                                       physicalUnits=True, numThreads=8)
            grp_ids = E.read_array('PARTDATA', path, snap, 'PartType4/GroupNumber', noH=True,
                                   physicalUnits=True, verbose=False, numThreads=8)

            subgrp_ids = E.read_array('PARTDATA', path, snap, 'PartType4/SubGroupNumber', noH=True,
                                      physicalUnits=True, verbose=False, numThreads=8)

            part_ids = E.read_array('PARTDATA', path, snap, 'PartType4/ParticleIDs', noH=True,
                                    physicalUnits=True, verbose=False, numThreads=8)
        except ValueError:
            continue
        except KeyError:
            continue
        except OSError:
            continue

        # A copy of this array is needed for the extraction method
        group_part_ids = np.copy(part_ids)

        print("There are", len(subgrp_ids), "particles")

        # Convert IDs to float(groupNumber.SubGroupNumber) format, i.e. group 1 subgroup 11 = 1.00011
        part_halo_ids = np.zeros(grp_ids.size, dtype=float)
        for (ind, g), sg in zip(enumerate(grp_ids), subgrp_ids):
            part_halo_ids[ind] = float(str(int(g)) + '.%05d' % int(sg))

        print("There are", len(part_halo_ids), "particles")

        print("Got halo IDs")

        okinds = np.isin(part_halo_ids, star_halo_ids)
        part_halo_ids = part_halo_ids[okinds]
        part_ids = part_ids[okinds]
        group_part_ids = group_part_ids[okinds]
        form_a = form_a[okinds]

        parts_in_groups, part_groups = get_part_inds(part_halo_ids, part_ids, group_part_ids, False)

        # Produce a dictionary containing the index of particles in each halo
        halo_part_inds = {}
        for ind, grp in zip(parts_in_groups, part_groups):
            halo_part_inds.setdefault(grp, set()).update({ind})

        # Now the dictionary is fully populated convert values from sets to arrays for indexing
        for key, val in halo_part_inds.items():
            halo_part_inds[key] = np.array(list(val))

        # Get the position of each of these galaxies
        for id, hmr in zip(star_halo_ids, gal_hmrs):
            mask = halo_part_inds[id]
            form.setdefault(snap, []).append(np.median(1 / form_a[mask] - 1))
            star_hmrs.setdefault(snap, []).append(hmr[4] / soft)
            gas_hmrs.setdefault(snap, []).append(hmr[0] / soft)

    return gas_hmrs, star_hmrs, form


def main_evolve_graph(snap):

    regions = []
    for reg in range(0, 1):
        if reg < 10:
            regions.append('0' + str(reg))
        else:
            regions.append(str(reg))

    snaplist = ['003_z012p000', '004_z011p000', '005_z010p000',
                '006_z009p000', '007_z008p000', '008_z007p000',
                '009_z006p000', '010_z005p000', '011_z004p770']

    stellar_hmr_dict = {}
    gas_hmr_dict = {}
    form_zs_dict = {}

    for reg in regions:

        path = '/cosma/home/dp004/dc-rope1/FLARES/FLARES-1/G-EAGLE_' + reg + '/data'

        try:
            gas_hmrs, star_hmrs, form = get_evolution(path, snaplist)
        except OSError:
            continue
        except ValueError:
            continue

        for snap in snaplist:

            print(reg, snap)

            try:
                stellar_hmr_dict.setdefault(snap, []).extend(star_hmrs[snap])
                gas_hmr_dict.setdefault(snap, []).extend(gas_hmrs[snap])
                form_zs_dict.setdefault(snap, []).extend(form[snap])
            except KeyError:
                continue

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

    for ax, snap, (i, j) in zip([ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8, ax9], snaplist,
                                [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2), (2, 0), (2, 1), (2, 2)]):

        z_str = snap.split('z')[1].split('p')
        z = float(z_str[0] + '.' + z_str[1])

        try:
            stellar_hmr = np.array(stellar_hmr_dict[snap])
            gas_hmr = np.array(gas_hmr_dict[snap])
            form_zs = np.array(form_zs_dict[snap])
        except KeyError:
            continue

        okinds = np.logical_and(stellar_hmr > 0, np.logical_and(gas_hmr > 0, form_zs > 0))
        stellar_hmr = stellar_hmr[okinds]
        gas_hmr = gas_hmr[okinds]
        form_zs = form_zs[okinds]

        try:
            cbar = ax.hexbin(cosmo.age(z).value - cosmo.age(np.array(form_zs)).value, gas_hmr, gridsize=100, mincnt=1, yscale='log',
                             norm=LogNorm(), linewidths=0.2, cmap='viridis', alpha=0.7)

            # plot_spread_stat(np.array(form_zs), np.array(gas_hmr), ax1, color='orangered')
            plot_median_stat(cosmo.age(z).value - cosmo.age(np.array(form_zs)).value, np.array(gas_hmr), ax, lab='Gas',
                             color='orangered')
        except ValueError:
            continue
        except OverflowError:
            continue

        ax.text(0.8, 0.9, f'$z={z}$', bbox=dict(boxstyle="round,pad=0.3", fc='w', ec="k", lw=1, alpha=0.8),
                transform=ax.transAxes, horizontalalignment='right', fontsize=8)

        axlims_x.extend(ax.get_xlim())
        axlims_y.extend(ax.get_ylim())

        # Label axes
        if i == 2:
            ax.set_xlabel(r"$t_{\mathrm{current}} - t_{50^{\mathrm{th}}, \mathrm{form}}$")
        if j == 0:
            ax.set_ylabel("$R_{1/2} / \epsilon$")

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

    # handles, labels = ax9.get_legend_handles_labels()
    # ax1.legend(handles, labels, loc='upper left', ncol=2, fontsize=6)

    fig.savefig("plots/gas_stellar_formation_hmr_allgals.png", bbox_inches="tight")

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

    for ax, snap, (i, j) in zip([ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8, ax9], snaplist,
                                [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2), (2, 0), (2, 1), (2, 2)]):

        z_str = snap.split('z')[1].split('p')
        z = float(z_str[0] + '.' + z_str[1])

        try:
            stellar_hmr = np.array(stellar_hmr_dict[snap])
            gas_hmr = np.array(gas_hmr_dict[snap])
            form_zs = np.array(form_zs_dict[snap])
        except KeyError:
            continue

        okinds = np.logical_and(stellar_hmr > 0, np.logical_and(gas_hmr > 0, form_zs > 0))
        stellar_hmr = stellar_hmr[okinds]
        gas_hmr = gas_hmr[okinds]
        form_zs = form_zs[okinds]

        try:
            cbar = ax.hexbin(cosmo.age(z).value - cosmo.age(np.array(form_zs)).value, stellar_hmr, gridsize=100, mincnt=1,
                             yscale='log', norm=LogNorm(), linewidths=0.2, cmap='viridis', alpha=0.7)

            # plot_spread_stat(np.array(form_zs), np.array(stellar_hmr), ax2, color='limegreen')
            plot_median_stat(cosmo.age(z).value - cosmo.age(np.array(form_zs)).value, np.array(stellar_hmr), ax, lab='Stellar',
                             color='limegreen')

        except ValueError:
            continue
        except OverflowError:
            continue

        ax.text(0.8, 0.9, f'$z={z}$', bbox=dict(boxstyle="round,pad=0.3", fc='w', ec="k", lw=1, alpha=0.8),
                transform=ax.transAxes, horizontalalignment='right', fontsize=8)

        axlims_x.extend(ax.get_xlim())
        axlims_y.extend(ax.get_ylim())

        # Label axes
        if i == 2:
            ax.set_xlabel(r"$t_{\mathrm{current}} - t_{50^{\mathrm{th}}, \mathrm{form}}$")
        if j == 0:
            ax.set_ylabel("$R_{1/2} / \epsilon$")

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

    # handles, labels = ax9.get_legend_handles_labels()
    # ax1.legend(handles, labels, loc='upper left', ncol=2, fontsize=6)

    fig.savefig("plots/gas_stellar_formation_stellarhmr_allgals.png", bbox_inches="tight")

    plt.close(fig)


main_evolve_graph(snap='011_z004p770')
