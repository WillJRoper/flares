#!/cosma/home/dp004/dc-rope1/.conda/envs/flares-env/bin/python
import matplotlib as ml
ml.use('Agg')
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import eagle_IO as E
import h5py
import sys
import seaborn as sns


sns.set_style('whitegrid')


def get_change_in_radius(snap, prog_snap, savepath, gal_data, gals):

    # Open graph file
    hdf = h5py.File(savepath + 'SubMgraph_' + snap + '.hdf5', 'r')

    # Initialise arrays for results
    delta_hmrs = np.zeros(len(gals))
    delta_ms = np.zeros(len(gals))

    print('There are', len(gals), 'Galaxies to test in snapshot', snap)

    # Loop over galaxies
    for ind, i in enumerate(gals):

        # Get this halo's stellar mass and half mass radius
        mass, hmr = gal_data[snap][i]['m'], gal_data[snap][i]['hmr']

        # Get progenitors
        try:
            progs = hdf[str(i)]['Prog_haloIDs'][...]
        except KeyError:
            # Define change in properties
            delta_hmrs[ind] = 2**30
            delta_ms[ind] = 2**30
            print(i, ind, "Galaxy has no dark matter")
            continue

        if len(progs) == 0:

            # Define change in properties
            delta_hmrs[ind] = 2**30
            delta_ms[ind] = 2**30

        else:

            # Get progenitor properties
            prog_cont = hdf[str(i)]['prog_npart_contribution'][...]
            prog_masses = hdf[str(i)]['prog_stellar_mass_contribution'][...] * 10**10
            prog_hmrs = np.array([gal_data[prog_snap][p]['hmr'] for p in progs])

            # Get main progenitor information
            main = np.argmax(prog_cont)
            main_mass = prog_masses[main]
            main_hmr = prog_hmrs[main]

            print(mass, main_mass, np.sum(prog_masses), prog_masses)

            # Define change in properties
            delta_hmrs[ind] = (hmr - main_hmr) / main_hmr
            delta_ms[ind] = (mass - np.sum(prog_masses)) / np.sum(prog_masses)

    hdf.close()

    return delta_hmrs[delta_ms < 2**30], delta_ms[delta_ms < 2**30]


def main_change(snap, prog_snap, masslim=1e8):

    regions = []
    for reg in range(0, 40):
        if reg < 10:
            regions.append('0' + str(reg))
        else:
            regions.append(str(reg))

    delta_hmr_dict = {}
    delta_ms_dict = {}

    # Define snapshots
    snaps = ['001_z014p000', '002_z013p000', '003_z012p000', '004_z011p000', '005_z010p000',
             '006_z009p000', '007_z008p000', '008_z007p000', '009_z006p000', '010_z005p000', '011_z004p770']
    prog_snaps = ['000_z015p000', '001_z014p000', '002_z013p000', '003_z012p000', '004_z011p000', '005_z010p000',
                  '006_z009p000', '007_z008p000', '008_z007p000', '009_z006p000', '010_z005p000']

    for snap, prog_snap in zip(snaps, prog_snaps):

        delta_hmr_dict[snap] = {}
        delta_ms_dict[snap] = {}

        for reg in regions:

            savepath = '/cosma/home/dp004/dc-rope1/FLARES/FLARES-1/MergerGraphs/GEAGLE_' + reg + '/'

            path = '/cosma/home/dp004/dc-rope1/FLARES/FLARES-1/G-EAGLE_' + reg + '/data'

            # Get halo IDs and halo data
            try:
                subgrp_ids = E.read_array('SUBFIND', path, snap, 'Subhalo/SubGroupNumber', numThreads=8)
                grp_ids = E.read_array('SUBFIND', path, snap, 'Subhalo/GroupNumber', numThreads=8)
                gal_hmrs = E.read_array('SUBFIND', path, snap, 'Subhalo/HalfMassRad', noH=True,
                                        physicalUnits=True, numThreads=8)[:, 4]
                gal_ms = E.read_array('SUBFIND', path, snap, 'Subhalo/ApertureMeasurements/Mass/030kpc',
                                      noH=False, physicalUnits=False, numThreads=8)[:, 4] * 10**10

                # Get halo IDs and halo data
                prog_subgrp_ids = E.read_array('SUBFIND', path, prog_snap, 'Subhalo/SubGroupNumber', numThreads=8)
                prog_grp_ids = E.read_array('SUBFIND', path, prog_snap, 'Subhalo/GroupNumber', numThreads=8)
                prog_gal_hmrs = E.read_array('SUBFIND', path, prog_snap, 'Subhalo/HalfMassRad', noH=True,
                                        physicalUnits=True, numThreads=8)[:, 4]
                prog_gal_ms = E.read_array('SUBFIND', path, prog_snap, 'Subhalo/ApertureMeasurements/Mass/030kpc',
                                           noH=False, physicalUnits=False, numThreads=8)[:, 4] * 10**10
            except ValueError:
                continue
            except OSError:
                continue

            # Remove particles not associated to a subgroup
            okinds = np.logical_and(subgrp_ids != 1073741824, gal_ms > 0)
            gal_hmrs = gal_hmrs[okinds]
            gal_ms = gal_ms[okinds]
            grp_ids = grp_ids[okinds]
            subgrp_ids = subgrp_ids[okinds]
            halo_ids = np.zeros(grp_ids.size, dtype=float)
            for (ind, g), sg in zip(enumerate(grp_ids), subgrp_ids):
                halo_ids[ind] = float(str(int(g)) + '.' + str(int(sg)))

            # Remove particles not associated to a subgroup
            okinds = prog_subgrp_ids != 1073741824
            prog_gal_hmrs = prog_gal_hmrs[okinds]
            prog_gal_ms = prog_gal_ms[okinds]
            prog_grp_ids = prog_grp_ids[okinds]
            prog_subgrp_ids = prog_subgrp_ids[okinds]
            prog_ids = np.zeros(prog_grp_ids.size, dtype=float)
            for (ind, g), sg in zip(enumerate(prog_grp_ids), prog_subgrp_ids):
                prog_ids[ind] = float(str(int(g)) + '.' + str(int(sg)))

            # Initialise galaxy data
            gal_data = {snap: {}, prog_snap: {}}
            for m, hmr, i in zip(gal_ms, gal_hmrs, halo_ids):
                gal_data[snap][i] = {'m': m, 'hmr': hmr}
            for m, hmr, i in zip(prog_gal_ms, prog_gal_hmrs, prog_ids):
                gal_data[prog_snap][i] = {'m': m, 'hmr': hmr}

            # Get change in stellar mass and half mass radius
            try:
                delta_hmr_dict[snap][reg], delta_ms_dict[snap][reg] = get_change_in_radius(snap, prog_snap, savepath,
                                                                                           gal_data,
                                                                                           halo_ids[gal_ms > masslim])
            except OSError:
                continue

    delta_hmr = []
    delta_mass = []
    for snap in snaps:
        delta_hmr.append(np.concatenate(list(delta_hmr_dict[snap].values())))
        delta_mass.append(np.concatenate(list(delta_ms_dict[snap].values())))

    delta_hmr = np.concatenate(delta_hmr)
    delta_mass = np.concatenate(delta_mass)

    print(delta_hmr)
    print(delta_mass)

    # Set up plot
    fig = plt.figure()
    ax = fig.add_subplot(111)

    # Plot results
    cbar = ax.hexbin(delta_mass, delta_hmr, gridsize=100, mincnt=1,
                     norm=LogNorm(), linewidths=0.2, cmap='viridis')

    # Label axes
    ax.set_xlabel(r'$M_{\star}/M_{\star, \mathrm{from progs}} - 1$')
    ax.set_ylabel('$\Delta R_{1/2,\mathrm{\star}}/R_{1/2,\mathrm{\star},\mathrm{main prog}}$')

    fig.colorbar(cbar, ax=ax)

    fig.savefig('plots/change_in_halfmassradius.png', bbox_inches='tight')


def main_change_single_snap(snap, prog_snap, masslim=1e8):

    regions = []
    for reg in range(0, 40):
        if reg < 10:
            regions.append('0' + str(reg))
        else:
            regions.append(str(reg))

    delta_hmr_dict = {}
    delta_ms_dict = {}

    for reg in regions:

        savepath = '/cosma/home/dp004/dc-rope1/FLARES/FLARES-1/MergerGraphs/GEAGLE_' + reg + '/'

        path = '/cosma/home/dp004/dc-rope1/FLARES/FLARES-1/G-EAGLE_' + reg + '/data'

        # Get halo IDs and halo data
        try:
            subgrp_ids = E.read_array('SUBFIND', path, snap, 'Subhalo/SubGroupNumber', numThreads=8)
            grp_ids = E.read_array('SUBFIND', path, snap, 'Subhalo/GroupNumber', numThreads=8)
            gal_hmrs = E.read_array('SUBFIND', path, snap, 'Subhalo/HalfMassRad', noH=True,
                                    physicalUnits=True, numThreads=8)[:, 4]
            gal_ms = E.read_array('SUBFIND', path, snap, 'Subhalo/ApertureMeasurements/Mass/030kpc',
                                  noH=False, physicalUnits=False, numThreads=8)[:, 4] * 10**10

            # Get halo IDs and halo data
            prog_subgrp_ids = E.read_array('SUBFIND', path, prog_snap, 'Subhalo/SubGroupNumber', numThreads=8)
            prog_grp_ids = E.read_array('SUBFIND', path, prog_snap, 'Subhalo/GroupNumber', numThreads=8)
            prog_gal_hmrs = E.read_array('SUBFIND', path, prog_snap, 'Subhalo/HalfMassRad', noH=True,
                                    physicalUnits=True, numThreads=8)[:, 4]
            prog_gal_ms = E.read_array('SUBFIND', path, prog_snap, 'Subhalo/ApertureMeasurements/Mass/030kpc',
                                       noH=False, physicalUnits=False, numThreads=8)[:, 4] * 10**10
        except ValueError:
            continue
        except OSError:
            continue

        # Remove particles not associated to a subgroup
        okinds = np.logical_and(subgrp_ids != 1073741824, gal_ms > 0)
        gal_hmrs = gal_hmrs[okinds]
        gal_ms = gal_ms[okinds]
        grp_ids = grp_ids[okinds]
        subgrp_ids = subgrp_ids[okinds]
        halo_ids = np.zeros(grp_ids.size, dtype=float)
        for (ind, g), sg in zip(enumerate(grp_ids), subgrp_ids):
            halo_ids[ind] = float(str(int(g)) + '.%05d'%int(sg))

        # Remove particles not associated to a subgroup
        okinds = prog_subgrp_ids != 1073741824
        prog_gal_hmrs = prog_gal_hmrs[okinds]
        prog_gal_ms = prog_gal_ms[okinds]
        prog_grp_ids = prog_grp_ids[okinds]
        prog_subgrp_ids = prog_subgrp_ids[okinds]
        prog_ids = np.zeros(prog_grp_ids.size, dtype=float)
        for (ind, g), sg in zip(enumerate(prog_grp_ids), prog_subgrp_ids):
            prog_ids[ind] = float(str(int(g)) + '.%05d'%int(sg))

        # Initialise galaxy data
        gal_data = {snap: {}, prog_snap: {}}
        for m, hmr, i in zip(gal_ms, gal_hmrs, halo_ids):
            gal_data[snap][i] = {'m': m, 'hmr': hmr}
        for m, hmr, i in zip(prog_gal_ms, prog_gal_hmrs, prog_ids):
            gal_data[prog_snap][i] = {'m': m, 'hmr': hmr}

        # Get change in stellar mass and half mass radius
        try:
            delta_hmr_dict[reg], delta_ms_dict[reg] = get_change_in_radius(snap, prog_snap, savepath,
                                                                                       gal_data,
                                                                                       halo_ids[gal_ms > masslim])
        except OSError:
            continue

    delta_hmr = np.concatenate(list(delta_hmr_dict.values()))
    delta_mass = np.concatenate(list(delta_ms_dict.values()))

    print(delta_hmr)
    print(delta_mass)

    # Set up plot
    fig = plt.figure()
    ax = fig.add_subplot(111)

    # Plot results
    cbar = ax.hexbin(delta_mass, delta_hmr, gridsize=100, mincnt=1,
                     norm=LogNorm(), linewidths=0.2, cmap='viridis')

    # Label axes
    ax.set_xlabel(r'$M_{\star}/M_{\star, \mathrm{from progs}} - 1$')
    ax.set_ylabel('$\Delta R_{1/2,\mathrm{\star}}/R_{1/2,\mathrm{\star},\mathrm{main prog}}$')

    fig.colorbar(cbar, ax=ax)

    fig.savefig('plots/change_in_halfmassradius.png', bbox_inches='tight')


regions = []
for reg in range(0, 40):
    if reg < 10:
        regions.append('0' + str(reg))
    else:
        regions.append(str(reg))
if sys.argv[1] == 0:
    main_change(snap='010_z005p000', prog_snap='009_z006p000', masslim=10**8)
else:
    main_change_single_snap(snap='010_z005p000', prog_snap='009_z006p000', masslim=10 ** 8)
