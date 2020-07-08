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
    forest_dict = {}
    mass_dict = {}
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

                if halo in found_halos:
                    continue

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

                if halo in found_halos:
                    continue

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

        # Open this snapshots root group
        snap_tree_data = h5py.File(treepath + 'SubMgraph_' + snap + '.hdf5', 'r')

        mass_dict[snap] = np.array([snap_tree_data[str(halo)].attrs['current_halo_nPart']
                                    for halo in forest_dict[snap]])

        snap_tree_data.close()

    gen0 = forest_dict['011_z004p770']
    root = gen0[np.argmax(mass_dict['011_z004p770'])]

    # Define the main branch start
    main = root

    # Loop over snapshots to get main branch
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
            main_branch[snap] = []
            break

        snap_tree_data.close()

    return forest_dict, main_branch, gen0, root


def forest_worker(z0halo, treepath):

    # Get the forest with this halo at it's root
    forest_dict = get_forest(z0halo, treepath)

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
    for root in halos_in_pop[reg]:
        print("Building Tree For", reg, root)
        graphs.setdefault(reg, []).append(forest_worker(root, '/cosma/home/dp004/dc-rope1/FLARES/FLARES-1/MergerGraphs/'
                                                              'GEAGLE_' + reg + '/SubMgraph_011_z004p770_'
                                                                                'PartType1.hdf5'))

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
ax.set_ylabel('SFR / $[M_\odot/\mathrm{yr} / pMpc]$')

ax.set_yscale('log')

# cax = fig.colorbar(cbar, ax=ax)
# cax.ax.set_ylabel(r'$N$')

fig.savefig('plots/SFH.png')
