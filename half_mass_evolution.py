#!/cosma/home/dp004/dc-rope1/.conda/envs/flares-env/bin/python
import matplotlib as ml
ml.use('Agg')
import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.gridspec as gridspec
from matplotlib.colors import LogNorm
import eagle_IO as E
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

    # Create snapshot list in reverse order (present day to past) for the progenitor searching loop
    snaplist = ['000_z015p000', '001_z014p000', '002_z013p000', '003_z012p000',
                '004_z011p000', '005_z010p000', '006_z009p000', '007_z008p000',
                '008_z007p000', '009_z006p000', '010_z005p000', '011_z004p770']
    snaplist.reverse()
    print(snaplist)

    # Initialise the halo's set for tree walking
    halos = {z0halo}

    # Initialise the forest dictionary with the present day halo as the first entry
    forest_dict[snaplist[0]] = halos

    # =============== Progenitors ===============

    # Loop over snapshots and progenitor snapshots
    for prog_snap, snap in zip(snaplist[1:], snaplist[:-1]):

        print(snap, prog_snap)

        # Assign the halos variable for the next stage of the tree
        halos = forest_dict[snap]

        # Loop over halos in this snapshot
        for halo in halos:

            # Open this snapshots root group
            snap_tree_data = h5py.File(treepath + 'SubMgraph_' + snap + '.hdf5', 'r')

            # Assign progenitors adding the snapshot * 100000 to the ID to keep track of the snapshot ID
            # in addition to the halo ID
            forest_dict.setdefault(prog_snap, set()).update(set((snap_tree_data[str(halo)]['Prog_haloIDs'][...])))
            snap_tree_data.close()

    forest_snaps = list(forest_dict.keys())

    return forest_dict


def forest_worker(z0halo, treepath):

    # Get the forest with this halo at it's root
    forest_dict = get_forest(z0halo, treepath)

    print('Halo ' + str(z0halo) + '\'s Forest extracted...')

    return forest_dict


def get_evolution(forest, path, graphpath, snaps):

    # Set up dictionaries to store data
    hmrs = {}
    masses = {}
    progs = {}

    for snap in snaps:

        # Intialise dictionaries for each snap
        hmrs[snap] = {}
        masses[snap] = {}
        progs[snap] = {}

    for snap in forest.keys():

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

        # Get halo properties
        for halo in forest[snap]:
            hmrs[snap][halo] = gal_hmrs[halo_ids == halo]
            masses[snap][halo] = gal_ms[halo_ids == halo]
            progs[snap][halo] = hdf[str(halo)]['Prog_haloIDs'][...]

        hdf.close()

    return hmrs, masses, progs


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
        forest = forest_worker(root, graphpath)

        hmrs, masses, progs = get_evolution(forest, path, graphpath, snaplist)

        forest_snaps = list(forest.keys())[1:]
        forest_progsnaps = list(forest.keys())[:-1]

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

        if count >= lim:
            break


main_evolve(reg='00', root_snap='011_z004p770', lim=1)
