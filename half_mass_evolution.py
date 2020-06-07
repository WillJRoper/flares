#!/cosma/home/dp004/dc-rope1/.conda/envs/flares-env/bin/python
import matplotlib as ml
ml.use('Agg')
import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.gridspec as gridspec
from matplotlib.colors import LogNorm
import eagle_IO.eagle_IO as E
import h5py
import sys
import multiprocessing
import pickle
import seaborn as sns


sns.set_style('whitegrid')


def get_forest(z0halo, treepath):
    """ A funciton which traverses a tree including all halos which have interacted with the tree
    in a 'forest/mangrove'.

    :param tree_data: The tree data dictionary produced by the Merger Graph.
    :param z0halo: The halo ID of a z=0 halo for which the forest/mangrove plot is desired.

    :return: forest_dict: The dictionary containing the forest. Each key is the snapshot ID and
             the value is a list of halos in this snapshot of the forest.
             massgrowth: The mass history of the forest.
             tree: The dictionary containing the tree. Each key is the snapshot ID and
             the value is a list of halos in this snapshot of the tree.
             main_growth: The mass history of the main branch.
    """

    # Initialise dictionary instances
    forest_dict = defaultdict(set)
    main_branch = {'011_z004p770': z0halo}

    # Create snapshot list in reverse order (present day to past) for the progenitor searching loop
    snaplist = ['000_z015p000', '001_z014p000', '002_z013p000', '003_z012p000',
                '004_z011p000', '005_z010p000', '006_z009p000', '007_z008p000',
                '008_z007p000', '009_z006p000', '010_z005p000', '011_z004p770']
    snaplist.reverse()

    # Initialise the halo's set for tree walking
    halos = {(z0halo, '011_z004p770')}

    # Initialise the forest dictionary with the present day halo as the first entry
    forest_dict[snaplist[0]] = halos

    # Initialise the set of new found halos used to loop until no new halos are found
    new_halos = halos

    # Initialise the set of found halos used to check if the halo has been found or not
    found_halos = set()

    # Get the main branch
    main = z0halo
    # Loop over snapshots
    for snap, prog_snap in zip(snaplist[:-1], snaplist[1:]):

        # Open this snapshots root group
        snap_tree_data = h5py.File(treepath + 'SubMgraph_' + snap + '.hdf5', 'r')
        progs = snap_tree_data[str(main)]['Prog_haloIDs'][...]
        pconts = snap_tree_data[str(main)]['prog_npart_contribution'][...]
        sinds = np.argsort(pconts)
        try:
            main_branch[prog_snap] = progs[sinds][-1]
            main = main_branch[prog_snap]
        except IndexError:
            snap_tree_data.close()
            break

        snap_tree_data.close()

    main_branch[snaplist[-1]] = []

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
        for prog_snap, snap in zip(snaplist[1:], snaplist[:-1]):

            # Assign the halos variable for the next stage of the tree
            halos = forest_dict[snap]

            # Open this snapshots root group
            snap_tree_data = h5py.File(treepath + 'SubMgraph_' + snap + '.hdf5', 'r')

            # Loop over halos in this snapshot
            for halo in halos:

                # if halo in found_halos:
                #     continue

                # Assign progenitors adding the snapshot * 100000 to the ID to keep track of the snapshot ID
                # in addition to the halo ID
                forest_dict.setdefault(prog_snap, set()).update({(p, prog_snap) for p in
                                                                 snap_tree_data[str(halo[0])]['Prog_haloIDs'][...]})
            snap_tree_data.close()

            # Add any new halos not found in found halos to the new halos set
            new_halos.update(forest_dict[prog_snap] - found_halos)

        # =============== Descendants ===============

        # Loop over halos found during the progenitor step
        snapshots = list(reversed(list(forest_dict.keys())))
        for desc_snap, snap in zip(snapshots[1:], snapshots[:-1]):

            # Assign the halos variable for the next stage of the tree
            halos = forest_dict[snap]

            # Open this snapshots root group
            snap_tree_data = h5py.File(treepath + 'SubMgraph_' + snap + '.hdf5', 'r')

            # Loop over the progenitor halos
            for halo in halos:

                # if halo zinue

                # Load descendants adding the snapshot * 100000 to keep track of the snapshot ID
                # in addition to the halo ID
                forest_dict.setdefault(desc_snap, set()).update({(d, desc_snap) for d in
                                                                 snap_tree_data[str(halo[0])]['Desc_haloIDs'][...]})

            snap_tree_data.close()

            # Redefine the new halos set to have any new halos not found in found halos
            new_halos.update(forest_dict[desc_snap] - found_halos)

        # Add the new_halos to the found halos set
        found_halos.update(new_halos)

    forest_snaps = list(forest_dict.keys())

    for snap in forest_snaps:

        if len(forest_dict[snap]) == 0:
            del forest_dict[snap]
            continue

        forest_dict[snap] = np.array([float(halo[0]) for halo in forest_dict[snap]])

    return forest_dict, main_branch


def forest_worker(z0halo, treepath):

    # Get the forest with this halo at it's root
    forest_dict = get_forest(z0halo, treepath)

    print('Halo ' + str(z0halo) + '\'s Forest extracted...')

    return forest_dict


def get_evolution(forest, main_branch, path, graphpath, snaps):

    # Set up dictionaries to store data
    hmrs = {}
    masses = {}
    progs = {}
    main_snap = []
    main_hmr = []

    for snap in snaps:

        # Intialise dictionaries for each snap
        hmrs[snap] = {}
        masses[snap] = {}
        progs[snap] = {}

    for snap in forest.keys():

        z_str = snap.split('z')[1].split('p')
        z = float(z_str[0] + '.' + z_str[1])

        # Define comoving softening length in kpc
        soft = 0.001802390 / 0.677 * 1 / (1 + z)

        # Get halo IDs and halo data
        subgrp_ids = E.read_array('SUBFIND', path, snap, 'Subhalo/SubGroupNumber', numThreads=8)
        grp_ids = E.read_array('SUBFIND', path, snap, 'Subhalo/GroupNumber', numThreads=8)
        gal_hmrs = E.read_array('SUBFIND', path, snap, 'Subhalo/HalfMassRad', noH=True,
                                physicalUnits=True, numThreads=8)[:, 4]
        gal_ms = E.read_array('SUBFIND', path, snap, 'Subhalo/ApertureMeasurements/Mass/030kpc',
                              noH=False, physicalUnits=False, numThreads=8)[:, 4] * 10 ** 10

        # Remove particles not associated to a subgroup
        okinds = subgrp_ids != 1073741824
        gal_hmrs = gal_hmrs[okinds]
        gal_ms = gal_ms[okinds]
        grp_ids = grp_ids[okinds]
        subgrp_ids = subgrp_ids[okinds]
        halo_ids = np.zeros(grp_ids.size, dtype=float)
        for (ind, g), sg in zip(enumerate(grp_ids), subgrp_ids):
            halo_ids[ind] = float(str(int(g)) + '.%05d' % int(sg))

        # Open graph file
        hdf = h5py.File(graphpath + 'SubMgraph_' + snap + '.hdf5', 'r')

        try:
            main_hmr.append(gal_hmrs[halo_ids == main_branch[snap]][0] / soft)
            main_snap.append(int(snap.split('_')[0]))
        except IndexError:
            continue

        # Get halo properties
        for halo in forest[snap]:
            hmrs[snap][float(halo)] = gal_hmrs[halo_ids == halo]
            masses[snap][float(halo)] = gal_ms[halo_ids == halo]
            progs[snap][float(halo)] = hdf[str(halo)]['Prog_haloIDs'][...]

        hdf.close()

    return hmrs, masses, progs, main_snap, main_hmr


def main_evolve(reg, root_snap='011_z004p770', lim=1):

    snaplist = ['000_z015p000', '001_z014p000', '002_z013p000', '003_z012p000',
                '004_z011p000', '005_z010p000', '006_z009p000', '007_z008p000',
                '008_z007p000', '009_z006p000', '010_z005p000', '011_z004p770']

    graphpath = '/cosma/home/dp004/dc-rope1/FLARES/FLARES-1/MergerGraphs/GEAGLE_' + reg + '/'

    path = '/cosma/home/dp004/dc-rope1/FLARES/FLARES-1/G-EAGLE_' + reg + '/data'

    # Get halo IDs and halo data
    subgrp_ids = E.read_array('SUBFIND', path, root_snap, 'Subhalo/SubGroupNumber', numThreads=8)
    grp_ids = E.read_array('SUBFIND', path, root_snap, 'Subhalo/GroupNumber', numThreads=8)
    gal_hmrs = E.read_array('SUBFIND', path, root_snap, 'Subhalo/HalfMassRad', noH=True,
                            physicalUnits=True, numThreads=8)[:, 4]
    gal_ms = E.read_array('SUBFIND', path, root_snap, 'Subhalo/ApertureMeasurements/Mass/030kpc',
                          noH=True, physicalUnits=True, numThreads=8)[:, 4] * 10 ** 10

    z_str = root_snap.split('z')[1].split('p')
    z = float(z_str[0] + '.' + z_str[1])

    # Convert inputs to physical kpc
    convert_pMpc = 1 / (1 + z)

    # Define comoving softening length in kpc
    csoft = 0.001802390 / 0.677 * convert_pMpc

    # Remove particles not associated to a subgroup
    okinds = np.logical_and(subgrp_ids != 1073741824, np.logical_and(gal_ms > 10**9.5, gal_hmrs / csoft < 1.2))
    gal_hmrs = gal_hmrs[okinds]
    gal_ms = gal_ms[okinds]
    grp_ids = grp_ids[okinds]
    subgrp_ids = subgrp_ids[okinds]
    halo_ids = np.zeros(grp_ids.size, dtype=float)
    for (ind, g), sg in zip(enumerate(grp_ids), subgrp_ids):
        halo_ids[ind] = float(str(int(g)) + '.%05d' % int(sg))

    # Initialise counter
    count = 0

    for root, hr, m in zip(halo_ids, gal_hmrs, gal_ms):

        count += 1

        # Get the graph
        forest, main_branch = forest_worker(root, graphpath)

        hmrs, masses, progs, main_snap, main_hmr = get_evolution(forest, main_branch, path, graphpath, snaplist)

        forest_snaps = list(forest.keys())
        forest_progsnaps = list(forest.keys())

        fig = plt.figure()
        ax = fig.add_subplot(111)

        masses_plt = []
        hmrs_plt = []
        colors = []

        for snap, prog_snap in zip(forest_snaps, forest_progsnaps):

            z_str = snap.split('z')[1].split('p')
            z = float(z_str[0] + '.' + z_str[1])

            z_str_prog = prog_snap.split('z')[1].split('p')
            zp = float(z_str_prog[0] + '.' + z_str_prog[1])

            # Define comoving softening length in kpc
            soft = 0.001802390 / 0.677 * 1 / (1 + z)
            # prog_soft = 0.001802390 / 0.677 * 1 / (1 + zp)

            for halo in forest[snap]:

                mass = masses[snap][halo]
                hmr = hmrs[snap][halo]

                if mass == 0 or hmr == 0:
                    continue

                print(z)

                masses_plt.extend(mass)
                hmrs_plt.extend(hmr / soft)
                colors.append(z)

                if z < 5:
                    ax.scatter(mass, hmr, marker='*', color='k')

                # # Get prog data
                # prog_hmrs = []
                # prog_mass = []
                # for prog in progs[snap][halo]:
                #     try:
                #         prog_hmrs.append(hmrs[prog_snap][prog])
                #         prog_mass.append(masses[prog_snap][prog])
                #     except KeyError:
                #         continue

                # for phmr, pm in zip(prog_hmrs, prog_mass):
                #     if pm == 0 or phmr == 0:
                #         continue
                #     print(pm, phmr, mass - pm, hmr - phmr)
                    # ax.arrow(pm[0], phmr[0], mass[0] - pm[0], hmr[0] - phmr[0])
                    # ax.scatter(pm, phmr / prog_soft, marker='.', color='r')

        masses_plt = np.array(masses_plt)
        hmrs_plt = np.array(hmrs_plt)
        colors = np.array(colors)

        im = ax.scatter(masses_plt[masses_plt > 1e8], hmrs_plt[masses_plt > 1e8], marker='.',
                        c=colors[masses_plt > 1e8], cmap='plasma')

        ax.set_xlabel(r'$M_{\mathrm{\star}}/M_\odot$')
        ax.set_ylabel('$R_{1/2,\mathrm{\star}}/\epsilon$')

        ax.set_yscale('log')
        ax.set_xscale('log')

        cbar = fig.colorbar(im)
        cbar.ax.set_ylabel(r'$z$')

        fig.savefig('plots/Evolution_HalfMassRadius_Mass' + str(root) + '.png',
                    bbox_inches='tight')

        # if count >= lim:
        #     break


def main_evolve_graph(reg, root_snap='011_z004p770', lim=1):

    snaplist = ['000_z015p000', '001_z014p000', '002_z013p000', '003_z012p000',
                '004_z011p000', '005_z010p000', '006_z009p000', '007_z008p000',
                '008_z007p000', '009_z006p000', '010_z005p000', '011_z004p770']

    graphpath = '/cosma/home/dp004/dc-rope1/FLARES/FLARES-1/MergerGraphs/GEAGLE_' + reg + '/'

    path = '/cosma/home/dp004/dc-rope1/FLARES/FLARES-1/G-EAGLE_' + reg + '/data'

    # Get halo IDs and halo data
    subgrp_ids = E.read_array('SUBFIND', path, root_snap, 'Subhalo/SubGroupNumber', numThreads=8)
    grp_ids = E.read_array('SUBFIND', path, root_snap, 'Subhalo/GroupNumber', numThreads=8)
    gal_hmrs = E.read_array('SUBFIND', path, root_snap, 'Subhalo/HalfMassRad', noH=True,
                            physicalUnits=True, numThreads=8)[:, 4]
    gal_ms = E.read_array('SUBFIND', path, root_snap, 'Subhalo/ApertureMeasurements/Mass/030kpc',
                          noH=True, physicalUnits=True, numThreads=8)[:, 4] * 10 ** 10

    z_str = root_snap.split('z')[1].split('p')
    z = float(z_str[0] + '.' + z_str[1])

    # Convert inputs to physical kpc
    convert_pMpc = 1 / (1 + z)

    # Define comoving softening length in kpc
    csoft = 0.001802390 / 0.677 * convert_pMpc

    # Remove particles not associated to a subgroup
    okinds = np.logical_and(subgrp_ids != 1073741824, np.logical_and(gal_ms > 10**9.5, gal_hmrs / csoft < 1.2))
    gal_hmrs = gal_hmrs[okinds]
    gal_ms = gal_ms[okinds]
    grp_ids = grp_ids[okinds]
    subgrp_ids = subgrp_ids[okinds]
    halo_ids = np.zeros(grp_ids.size, dtype=float)
    for (ind, g), sg in zip(enumerate(grp_ids), subgrp_ids):
        halo_ids[ind] = float(str(int(g)) + '.%05d' % int(sg))

    # Initialise counter
    count = 0

    for root, hr, m in zip(halo_ids, gal_hmrs, gal_ms):

        count += 1

        # Get the graph
        forest, main_branch = forest_worker(root, graphpath)

        hmrs, masses, progs, main_snap, main_hmr = get_evolution(forest, main_branch, path, graphpath, snaplist)


        forest_snaps = list(forest.keys())
        forest_progsnaps = list(forest.keys())[1:]
        forest_progsnaps.append(None)

        fig = plt.figure()
        ax = fig.add_subplot(111)

        masses_plt = []
        hmrs_plt = []
        snaps = []
        hmr_plot_pairs = set()
        snap_plot_pairs = set()

        for snap, prog_snap in zip(forest_snaps, forest_progsnaps):

            z_str = snap.split('z')[1].split('p')
            z = float(z_str[0] + '.' + z_str[1])

            # Define comoving softening length in kpc
            soft = 0.001802390 / 0.677 * 1 / (1 + z)
            # prog_soft = 0.001802390 / 0.677 * 1 / (1 + zp)

            print(prog_snap, snap)

            for halo in forest[snap]:

                try:
                    mass = masses[snap][float(halo)]
                    hmr = hmrs[snap][float(halo)]
                except KeyError:
                    continue

                masses_plt.extend(mass)
                hmrs_plt.extend(hmr / soft)
                snaps.append(int(snap.split('_')[0]))

                if prog_snap != None:

                    for prog in progs[snap][halo]:
                        try:
                            if masses[prog_snap][prog][0] > 1e8 and mass > 1e8:
                                hmr_plot_pairs.update({(hmrs[prog_snap][prog][0] / soft, hmr[0] / soft)})
                                snap_plot_pairs.update({(int(prog_snap.split('_')[0]), int(snap.split('_')[0]))})
                        except KeyError:
                            continue

        masses_plt = np.array(masses_plt)
        hmrs_plt = np.array(hmrs_plt)
        snaps = np.array(snaps)
        snap_plot_pairs = list(snap_plot_pairs)
        hmr_plot_pairs = list(hmr_plot_pairs)
        main_snap, main_hmr = np.array(main_snap), np.array(main_hmr)

        # for hpair, spair in zip(hmr_plot_pairs, snap_plot_pairs):
        #     print(hpair, spair)
        #     ax.plot(spair, hpair, linestyle='--', color='k')

        print(main_snap, main_hmr)

        ax.plot(main_snap, main_hmr, linestyle='-', color='r')

        im = ax.scatter(snaps[masses_plt > 1e8], hmrs_plt[masses_plt > 1e8],
                        s=masses_plt[masses_plt > 1e8] / max(masses_plt) * 30, cmap='plasma')

        ax.set_xlabel(r'$S_{\mathrm{num}}$')
        ax.set_ylabel('$R_{1/2,\mathrm{\star}}/\epsilon$')

        ax.set_yscale('log')

        # cbar = fig.colorbar(im)
        # cbar.ax.set_ylabel(r'$M_{\mathrm{\star}}/M_\odot$')

        fig.savefig('plots/Evolution/Graph_HalfMassRadius_Mass' + str(root).split('.')[0] +
                    'p' + str(root).split('.')[1] + '.png',
                    bbox_inches='tight')

        if count >= lim:
            break


main_evolve_graph(reg='00', root_snap='011_z004p770', lim=20)
