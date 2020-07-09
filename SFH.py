#!/cosma/home/dp004/dc-rope1/.conda/envs/flares-env/bin/python
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
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
    snaplist.reverse()

    # Initialise dictionary instances
    graph_dict = {}
    mass_dict = {}

    # Initialise the halo's set for tree walking
    halos = {(z0halo, snaplist[0])}

    # Initialise the graph dictionary with the present day halo as the first entry
    graph_dict[snaplist[0]] = halos

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
        for i in range(len(snaplist[:-1])):

            snap = snaplist[i]
            prog_snap = snaplist[i + 1]

            # Assign the halos variable for the next stage of the tree
            halos = graph_dict[snap]

            if len(halos) == 0:
                continue

            # Loop over halos in this snapshot
            for halo in halos:

                # Get the progenitors
                start_ind = data_dict['prog_start_index'][snap][data_dict['mega'][snap][data_dict['sim'][snap] == halo[0]]][0]
                nprog = data_dict['nprogs'][snap][data_dict['mega'][snap][data_dict['sim'][snap] == halo[0]]][0]
                print(start_ind, nprog, data_dict['nparts'][snap][data_dict['mega'][snap][data_dict['sim'][snap] == halo[0]]][0])
                if nprog == 0:
                    continue
                these_progs = get_linked_halo_data(data_dict['progs'][snap], start_ind, nprog)
                print(these_progs)

                # Assign progenitors using a tuple to keep track of the snapshot ID
                # in addition to the halo ID
                graph_dict.setdefault(prog_snap, set()).update({(p, prog_snap) for p in these_progs})
                print(graph_dict[prog_snap])
            # Add any new halos not found in found halos to the new halos set
            new_halos.update(graph_dict[prog_snap] - found_halos)

        # =============== Descendants ===============

        # Loop over halos found during the progenitor step
        snaplist.reverse()
        for i in range(len(snaplist[:-1])):

            snap = snaplist[i]
            desc_snap = snaplist[i + 1]

            # Assign the halos variable for the next stage of the tree
            halos = graph_dict[snap]

            if len(halos) == 0:
                continue

            # Loop over the progenitor halos
            for halo in halos:

                # Get the descendants
                start_ind = data_dict['desc_start_index'][snap][data_dict['mega'][snap][data_dict['sim'][snap] == halo[0]]][0]
                ndesc = data_dict['ndescs'][snap][data_dict['mega'][snap][data_dict['sim'][snap] == halo[0]]][0]
                if ndesc == 0:
                    continue
                these_descs = get_linked_halo_data(data_dict['descs'][snap], start_ind, ndesc)

                # Load descendants adding the snapshot * 100000 to keep track of the snapshot ID
                # in addition to the halo ID
                graph_dict.setdefault(desc_snap, set()).update({(d, desc_snap) for d in these_descs})

            # Redefine the new halos set to have any new halos not found in found halos
            new_halos.update(graph_dict[desc_snap] - found_halos)

        # Add the new_halos to the found halos set
        found_halos.update(new_halos)

    # Get the number of particle in each halo and sort based on mass
    for snap in graph_dict:

        if len(graph_dict[snap]) == 0:
            continue

        # Convert entry to an array for sorting
        graph_dict[snap] = np.array([int(halo[0]) for halo in graph_dict[snap]])

        # Get the halo masses
        mass_dict[snap] = data_dict['nparts'][graph_dict[snap]]

        # Sort by mass
        sinds = np.argsort(mass_dict[snap])[::-1]
        mass_dict[snap] = mass_dict[snap][sinds]
        graph_dict[snap] = graph_dict[snap][sinds]

    return graph_dict


def forest_worker(z0halo, data_dict):

    # Get the forest with this halo at it's root
    forest_dict = get_graph(z0halo, data_dict)

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
for reg in range(0, 1):

    if reg < 10:
        regions.append('0' + str(reg))
    else:
        regions.append(str(reg))

gregions = []
for reg in range(0, 1):

    if reg < 10:
        gregions.append('0' + str(reg))
    else:
        gregions.append(str(reg))

snaps = ['000_z015p000', '001_z014p000', '002_z013p000', '003_z012p000', '004_z011p000', '005_z010p000',
         '006_z009p000', '007_z008p000', '008_z007p000', '009_z006p000', '010_z005p000', '011_z004p770']
gsnaps = reversed(snaps)

# Define thresholds for roots
mthresh = 10**10
rthresh = 1.1

halo_ids_dict = {}
halo_ms_dict = {}
stellar_a_dict = {}
starmass_dict = {}
halo_id_part_inds = {}
for snap in snaps:

    stellar_a_dict[snap] = {}
    starmass_dict[snap] = {}
    halo_id_part_inds[snap] = {}
    halo_ids_dict[snap] = {}
    halo_ms_dict[snap] = {}

for reg in regions:

    for snap in snaps:

        print(reg, snap)

        path = '/cosma/home/dp004/dc-rope1/FLARES/FLARES-1/G-EAGLE_' + reg + '/data'

        try:
            starmass_dict[snap][reg] = E.read_array('PARTDATA', path, snap, 'PartType4/Mass',
                                                    noH=True, numThreads=8) * 10**10
            stellar_a_dict[snap][reg] = E.read_array('PARTDATA', path, snap, 'PartType4/StellarFormationTime',
                                                     noH=True, numThreads=8)
            part_ids = E.read_array('PARTDATA', path, snap, 'PartType4/ParticleIDs', numThreads=8)
            grp_ids = E.read_array('PARTDATA', path, snap, 'PartType4/GroupNumber', numThreads=8)
            subgrp_ids = E.read_array('PARTDATA', path, snap, 'PartType4/SubGroupNumber', numThreads=8)
            subfind_grp_ids = E.read_array('SUBFIND', path, snap, 'Subhalo/GroupNumber', numThreads=8)
            subfind_subgrp_ids = E.read_array('SUBFIND', path, snap, 'Subhalo/SubGroupNumber', numThreads=8)
            gal_ms = E.read_array('SUBFIND', path, snap, 'Subhalo/ApertureMeasurements/Mass/030kpc',
                                  numThreads=8)[:, 4] * 10 ** 10
        except:
            continue

        # A copy of this array is needed for the extraction method
        group_part_ids = np.copy(part_ids)

        # Convert IDs to float(groupNumber.SubGroupNumber) format, i.e. group 1 subgroup 11 = 1.00011
        halo_ids = np.zeros(subfind_grp_ids.size, dtype=float)
        for (ind, g), sg in zip(enumerate(subfind_grp_ids), subfind_subgrp_ids):
            halo_ids[ind] = float(str(int(g)) + '.%05d' % int(sg))

        halo_ids_dict[snap][reg] = halo_ids
        halo_ms_dict[snap][reg] = gal_ms

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
        halo_id_part_inds[snap][reg] = halo_part_inds

        print("There are", len(halo_part_inds), "halos")


# Get halos which are in the distribution at the z=4.77
halos_in_pop = {}
count = 0
for reg in regions:

    halos_in_pop[reg] = halo_ids_dict['011_z004p770'][reg][halo_ms_dict['011_z004p770'][reg] >= mthresh]
    count += len(halos_in_pop[reg])

print("There are", count, "halos fullfilling condition")

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
        sim[snap] = hdf['MEGA_halo_IDs'][...]
        nparts[snap] = hdf['nParts'][...]

        hdf.close()

    data_dict = {'progs': progs, 'descs': descs, 'nprogs': nprogs, 'ndescs': ndescs,
                 'prog_start_index': prog_start_index, 'desc_start_index': desc_start_index,
                 'nparts': nparts, 'mega': mega, 'sim': sim}

    for root in halos_in_pop[reg]:
        print("Building Tree For", reg, root)
        graphs.setdefault(reg, []).append(forest_worker(root, data_dict))

all_sfrs = []
all_zs = []
for reg in graphs:
    for graph in graphs[reg]:
        sfrs = []
        zs = []
        for snap in graph:
            z_str = snap.split('z')[1].split('p')
            z = float(z_str[0] + '.' + z_str[1])
            zs.append(z)
            this_sfr = 0
            for grp in graph[snap]:
                parts = list(halo_id_part_inds[snap][reg][grp])
                this_sfr += calc_srf(z, stellar_a_dict[snap][reg][parts], starmass_dict[snap][reg][parts])
            sfrs.append(this_sfr / (3200 / (1 + z)))
        all_zs.append(zs)
        all_sfrs.append(sfrs)

fig = plt.figure()
ax = fig.add_subplot(111)

# cbar = ax.hexbin(hex_zs, hex_sfrs, gridsize=100, mincnt=1, norm=LogNorm(), yscale='log',
#                  linewidths=0.2, cmap='Greys', zorder=0)

for z, sfr in zip(all_zs, all_sfrs):
    ax.plot(all_zs, all_sfrs, linestyle='-', color='k', alpha=0.3)


ax.set_xlabel('$z$')
ax.set_ylabel('SFRD / $[M_\odot/\mathrm{yr} / pMpc]$')

ax.set_yscale('log')

# cax = fig.colorbar(cbar, ax=ax)
# cax.ax.set_ylabel(r'$N$')

fig.savefig('plots/SFH.png')
