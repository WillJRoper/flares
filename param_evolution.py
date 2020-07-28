#!/cosma/home/dp004/dc-rope1/.conda/envs/flares-env/bin/python
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.gridspec as gridspec
import astropy.units as u
import astropy.constants as cons
from astropy.cosmology import Planck13 as cosmo
from matplotlib.colors import LogNorm
import eagle_IO.eagle_IO as E
import seaborn as sns
import h5py
matplotlib.use('Agg')

sns.set_style('whitegrid')


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
    nprogs = {}
    ndescs = {}

    # Initialise the halo's set for tree walking
    halos = {(z0halo, '011_z004p770')}

    # Initialise entries in the graph dictionary
    for snap in snaplist:
        graph_dict[snap] = set()
        nprogs[snap] = -1
        ndescs[snap] = -1

    # Initialise the graph dictionary with the present day halo as the first entry
    graph_dict['011_z004p770'] = halos

    # Loop over snapshots and progenitor snapshots
    for i in range(len(snaplist) - 1):

        snap = rev_snaplist[i]
        prog_snap = rev_snaplist[i + 1]
        snap_halo_ids = data_dict['mega'][snap]
        sim_halo_ids = data_dict['sim'][snap]

        # Assign the halos variable for the next stage of the tree
        halos = graph_dict[snap]

        # Loop over halos in this snapshot
        for halo in halos:

            # Get the progenitors
            try:
                start_ind = data_dict['prog_start_index'][snap][snap_halo_ids[sim_halo_ids == halo[0]]][0]
                nprog = data_dict['nprogs'][snap][snap_halo_ids[sim_halo_ids == halo[0]]][0]
                ndesc = data_dict['ndescs'][snap][snap_halo_ids[sim_halo_ids == halo[0]]][0]
                nprogs[snap] = nprog
                ndescs[snap] = ndesc
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

    return graph_dict, nprogs, ndescs


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


regions = []
for reg in range(0, 2):

    if reg < 10:
        regions.append('0' + str(reg))
    else:
        regions.append(str(reg))

gregions = []
for reg in range(0, 2):

    if reg < 10:
        gregions.append('0' + str(reg))
    else:
        gregions.append(str(reg))

snaps = ['000_z015p000', '001_z014p000', '002_z013p000', '003_z012p000', '004_z011p000', '005_z010p000',
         '006_z009p000', '007_z008p000', '008_z007p000', '009_z006p000', '010_z005p000', '011_z004p770']
gsnaps = reversed(snaps)

# Define thresholds for roots
mthresh = 10**9.5
rthresh = 0.7

halo_ids_dict = {}
# halo_ms_dict = {}
# stellar_a_dict = {}
# starmass_dict = {}
# halo_id_part_inds = {}
gal_gas_hmrs = {}
gal_star_hmrs = {}
gal_dm_hmrs = {}
gal_star_ms = {}
gal_gas_ms = {}
gal_dm_ms = {}
gal_sfr = {}
gal_energy = {}
gal_bhmar = {}
gal_veldisp = {}
gal_birthden = {}
for snap in snaps:

    # stellar_a_dict[snap] = {}
    # starmass_dict[snap] = {}
    # halo_id_part_inds[snap] = {}
    halo_ids_dict[snap] = {}
    # halo_ms_dict[snap] = {}
    gal_gas_hmrs[snap] = {}
    gal_star_hmrs[snap] = {}
    gal_dm_hmrs[snap] = {}
    gal_dm_ms[snap] = {}
    gal_gas_ms[snap] = {}
    gal_star_ms[snap] = {}
    gal_sfr[snap] = {}
    gal_energy[snap] = {}
    gal_bhmar[snap] = {}
    gal_veldisp[snap] = {}
    gal_birthden[snap] = {}

for reg in regions:

    for snap in snaps:

        print(reg, snap)

        path = '/cosma/home/dp004/dc-rope1/FLARES/FLARES-1/G-EAGLE_' + reg + '/data'

        try:
            # starmass_dict[snap][reg] = E.read_array('PARTDATA', path, snap, 'PartType4/Mass',
            #                                         noH=True, numThreads=8) * 10**10
            gal_bd = E.read_array('PARTDATA', path, snap, 'PartType4/BirthDensity', noH=True,
                                  physicalUnits=True, numThreads=8)
            part_ids = E.read_array('PARTDATA', path, snap, 'PartType4/ParticleIDs', numThreads=8)
            grp_ids = E.read_array('PARTDATA', path, snap, 'PartType4/GroupNumber', numThreads=8)
            subgrp_ids = E.read_array('PARTDATA', path, snap, 'PartType4/SubGroupNumber', numThreads=8)
            subfind_grp_ids = E.read_array('SUBFIND', path, snap, 'Subhalo/GroupNumber', numThreads=8)
            subfind_subgrp_ids = E.read_array('SUBFIND', path, snap, 'Subhalo/SubGroupNumber', numThreads=8)
            gal_ms = E.read_array('SUBFIND', path, snap, 'Subhalo/ApertureMeasurements/Mass/030kpc', noH=True,
                                  numThreads=8) * 10 ** 10
            gal_hmr = E.read_array('SUBFIND', path, snap, 'Subhalo/HalfMassRad', noH=True,
                                   numThreads=8) * 1e3
            gal_sfr[snap][reg] = E.read_array('SUBFIND', path, snap, 'Subhalo/StarFormationRate', noH=True,
                                   numThreads=8) * 10 ** 10
            gal_energy[snap][reg] = E.read_array('SUBFIND', path, snap, 'Subhalo/TotalEnergy', noH=True,
                                              numThreads=8)
            gal_bhmar[snap][reg] = E.read_array('SUBFIND', path, snap, 'Subhalo/BlackHoleMassAccretionRate', noH=True,
                                                 numThreads=8) * 10**10
            gal_veldisp[snap][reg] = E.read_array('SUBFIND', path, snap, 'Subhalo/StellarVelDisp', noH=True,
                                                  numThreads=8)
            gal_gas_hmrs[snap][reg] = gal_hmr[:, 0]
            gal_star_hmrs[snap][reg] = gal_hmr[:, 4]
            gal_dm_hmrs[snap][reg] = gal_hmr[:, 1]
            gal_dm_ms[snap][reg] = gal_ms[:, 1]
            gal_gas_ms[snap][reg] = gal_ms[:, 0]
            gal_star_ms[snap][reg] = gal_ms[:, 4]
            gal_ms = gal_ms[:, 4]

        except:
            continue

        # A copy of this array is needed for the extraction method
        group_part_ids = np.copy(part_ids)

        # Convert IDs to float(groupNumber.SubGroupNumber) format, i.e. group 1 subgroup 11 = 1.00011
        halo_ids = np.zeros(subfind_grp_ids.size, dtype=float)
        for (ind, g), sg in zip(enumerate(subfind_grp_ids), subfind_subgrp_ids):
            halo_ids[ind] = float(str(int(g)) + '.%05d' % int(sg))

        halo_ids_dict[snap][reg] = halo_ids
        # halo_ms_dict[snap][reg] = gal_ms

        # Convert IDs to float(groupNumber.SubGroupNumber) format, i.e. group 1 subgroup 11 = 1.00011
        part_halo_ids = np.zeros(grp_ids.size, dtype=float)
        for (ind, g), sg in zip(enumerate(grp_ids), subgrp_ids):
            part_halo_ids[ind] = float(str(int(g)) + '.%05d' % int(sg))

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

        print("There are", len(halo_part_inds), "halos")

        bd = []
        for halo in halo_ids:
            try:
                bd.append(np.mean(gal_bd[halo_part_inds[halo]]))
            except KeyError:
                bd.append(np.nan)

        gal_birthden[snap][reg] = np.array(bd)


# Get halos which are in the distribution at the z=4.77
halos_in_pop = {}
count = 0
for reg in regions:

    try:
        halos_in_pop[reg] = halo_ids_dict['011_z004p770'][reg][np.logical_and(gal_star_ms['011_z004p770'][reg] >= mthresh, gal_gas_hmrs[snap][reg] < rthresh)]
        count += len(halos_in_pop[reg])
    except KeyError:
        continue

print("There are", count, "halos fullfilling condition")

# Define comoving softening length in kpc
csoft = 0.001802390 / 0.6777 * 1e3

# Build graphs
graphs = {}
for reg in halos_in_pop:

    treepath = '/cosma/home/dp004/dc-rope1/FLARES/FLARES-1/MergerGraphs/GEAGLE_' + reg + '/'

    # Get the start indices, progs, and descs and store them in dictionaries
    progs = {}
    descs = {}
    nprogs = {}
    ndescs = {}
    prog_start_index = {}
    desc_start_index = {}
    nparts = {}
    mega = {}
    sim = {}
    for snap in snaps:

        hdf = h5py.File(treepath + 'SubMgraph_' + snap + '.hdf5', 'r')

        # Assign
        progs[snap] = hdf['prog_halo_ids'][...]
        descs[snap] = hdf['desc_halo_ids'][...]
        nprogs[snap] = hdf['nProgs'][...]
        ndescs[snap] = hdf['nDescs'][...]
        prog_start_index[snap] = hdf['Prog_Start_Index'][...]
        desc_start_index[snap] = hdf['Desc_Start_Index'][...]
        mega[snap] = hdf['MEGA_halo_IDs'][...]
        sim[snap] = hdf['SUBFIND_halo_IDs'][...]
        nparts[snap] = hdf['nParts'][...]

        hdf.close()

    data_dict = {'progs': progs, 'descs': descs, 'nprogs': nprogs, 'ndescs': ndescs,
                 'prog_start_index': prog_start_index, 'desc_start_index': desc_start_index,
                 'nparts': nparts, 'mega': mega, 'sim': sim}

    for root in halos_in_pop[reg]:
        print("Building graph For", reg, root)
        graph, nprogs_dict, ndescs_dict = forest_worker(root, data_dict)
        print("Plotting graph")
        sfrs = []
        gas_ms = []
        star_ms = []
        dm_ms = []
        dm_hmr = []
        gas_hmr = []
        star_hmr = []
        energy = []
        zs = []
        bhmar = []
        nprogs = []
        ndescs = []
        vel_disp = []
        soft = []
        bd = []
        for snap in graph:
            z_str = snap.split('z')[1].split('p')
            z = float(z_str[0] + '.' + z_str[1])
            for grp in graph[snap]:
                zs.append(z)
                sfrs.append(gal_sfr[snap][reg][halo_ids_dict[snap][reg] == grp])
                gas_ms.append(gal_gas_ms[snap][reg][halo_ids_dict[snap][reg] == grp])
                star_ms.append(gal_star_ms[snap][reg][halo_ids_dict[snap][reg] == grp])
                gas_hmr.append(gal_gas_hmrs[snap][reg][halo_ids_dict[snap][reg] == grp])
                star_hmr .append(gal_star_hmrs[snap][reg][halo_ids_dict[snap][reg] == grp])
                dm_ms.append(gal_dm_ms[snap][reg][halo_ids_dict[snap][reg] == grp])
                dm_hmr.append(gal_dm_hmrs[snap][reg][halo_ids_dict[snap][reg] == grp])
                energy.append(gal_energy[snap][reg][halo_ids_dict[snap][reg] == grp])
                nprogs.append(nprogs_dict[snap])
                ndescs.append(ndescs_dict[snap])
                bhmar.append(gal_bhmar[snap][reg][halo_ids_dict[snap][reg] == grp])
                vel_disp.append(gal_veldisp[snap][reg][halo_ids_dict[snap][reg] == grp])
                soft.append(csoft / (1 + z))
                bd.append(gal_birthden[snap][reg][halo_ids_dict[snap][reg] == grp])

        fig = plt.figure(figsize=(12, 12))
        gs = gridspec.GridSpec(ncols=3, nrows=4)
        gs.update(wspace=0.3, hspace=0.0)
        ax1 = fig.add_subplot(gs[0, 0])
        ax2 = fig.add_subplot(gs[1, 0])
        ax3 = fig.add_subplot(gs[2, 0])
        ax4 = fig.add_subplot(gs[3, 0])
        ax5 = fig.add_subplot(gs[0, 1])
        ax6 = fig.add_subplot(gs[1, 1])
        ax7 = fig.add_subplot(gs[2, 1])
        ax8 = fig.add_subplot(gs[3, 1])
        ax9 = fig.add_subplot(gs[0, 2])
        ax10 = fig.add_subplot(gs[1, 2])
        ax11 = fig.add_subplot(gs[2, 2])
        ax12 = fig.add_subplot(gs[3, 2])

        ax1.plot(zs, dm_ms)
        ax2.plot(zs, gas_ms)
        ax3.plot(zs, star_ms)
        ax4.plot(zs, sfrs)
        ax5.plot(zs, soft, linestyle='--', color='k', label='soft')
        ax5.plot(zs, dm_hmr, label='Galaxy')
        ax6.plot(zs, soft, linestyle='--', color='k')
        ax6.plot(zs, gas_hmr)
        ax7.plot(zs, soft, linestyle='--', color='k')
        ax7.plot(zs, star_hmr)
        ax8.plot(zs, np.abs(energy))
        ax9.plot(zs, nprogs)
        ax10.plot(zs, ndescs)
        ax11.plot(zs, bhmar)
        ax12.plot(zs, bd)

        ax4.set_xlabel('$z$')
        ax8.set_xlabel('$z$')
        ax12.set_xlabel('$z$')

        ax1.set_ylabel('$M_{\mathrm{dm}} / M_\odot$')
        ax2.set_ylabel('$M_{\mathrm{gas}} / M_\odot$')
        ax3.set_ylabel('$M_{\star} / M_\odot$')
        ax4.set_ylabel('SFR / $[M_\odot/\mathrm{Gyr}]$')
        ax5.set_ylabel('$R_{1/2, \mathrm{dm}} / \mathrm{pkpc}$')
        ax6.set_ylabel('$R_{1/2, \mathrm{gas}} / \mathrm{pkpc}$')
        ax7.set_ylabel('$R_{1/2, \star} / \mathrm{pkpc}$')
        ax8.set_ylabel('|Total Energy| / $[???]$')
        ax9.set_ylabel('$N_{\mathrm{dprog}}$')
        ax10.set_ylabel('$N_{\mathrm{ddesc}}$')
        ax11.set_ylabel('$\dot{M_{BH}} / [M_\odot / \mathrm{Gyr}]$')
        ax12.set_ylabel(r'$<\rho_{birth}> /$ [$n_H$ cm$^{-3}$]')

        ax1.set_ylim(10**6.5, 10**12.5)
        ax2.set_ylim(10 ** 6.5, 10 ** 12.5)
        ax3.set_ylim(10 ** 6.5, 10 ** 12.5)
        ax5.set_ylim(10 ** -1, 10 ** 1.9)
        ax6.set_ylim(10 ** -1, 10 ** 1.9)
        ax7.set_ylim(10 ** -1, 10 ** 1.9)

        ax1.tick_params(axis='x', top=False, bottom=False, labeltop=False, labelbottom=False)
        ax2.tick_params(axis='x', top=False, bottom=False, labeltop=False, labelbottom=False)
        ax3.tick_params(axis='x', top=False, bottom=False, labeltop=False, labelbottom=False)
        ax5.tick_params(axis='x', top=False, bottom=False, labeltop=False, labelbottom=False)
        ax6.tick_params(axis='x', top=False, bottom=False, labeltop=False, labelbottom=False)
        ax7.tick_params(axis='x', top=False, bottom=False, labeltop=False, labelbottom=False)
        ax9.tick_params(axis='x', top=False, bottom=False, labeltop=False, labelbottom=False)
        ax10.tick_params(axis='x', top=False, bottom=False, labeltop=False, labelbottom=False)
        ax11.tick_params(axis='x', top=False, bottom=False, labeltop=False, labelbottom=False)

        ax1.set_yscale('log')
        ax2.set_yscale('log')
        ax3.set_yscale('log')
        ax4.set_yscale('log')
        ax5.set_yscale('log')
        ax6.set_yscale('log')
        ax7.set_yscale('log')
        ax8.set_yscale('log')
        ax11.set_yscale('log')
        ax12.set_yscale('log')

        for ax in [ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8, ax9, ax10, ax11, ax12]:
            ax.set_xlim(4.6, 15.1)
            for spine in ax.spines.values():
                spine.set_edgecolor('k')

        # cax = fig.colorbar(cbar, ax=ax)
        # cax.ax.set_ylabel(r'$N$')

        handles, labels = ax5.get_legend_handles_labels()
        ax5.legend(handles, labels)

        fig.savefig(f'plots/Evolution/Param_evolution_{reg}_{str(root).split(".")[0]}p{str(root).split(".")[1]}_compactgas.png')

        plt.close(fig)
