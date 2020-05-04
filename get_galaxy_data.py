#!/cosma/home/dp004/dc-rope1/.conda/envs/flares-env/bin/python
import numpy as np
import matplotlib
import eagle_IO as E
import pickle
import gc
import os
import sys
from utilities import calc_ages, get_Z_LOS
matplotlib.use('Agg')


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


def get_subgroup_part_inds(sim, snapshot, part_type, all_parts=False, sorted=False):
    ''' A function to efficiently produce a dictionary of particle indexes from EAGLE particle data arrays
        for SUBFIND subgroups.

    :param sim:        Path to the snapshot file [str]
    :param snapshot:   Snapshot identifier [str]
    :param part_type:  The integer representing the particle type
                       (0, 1, 4, 5: gas, dark matter, stars, black hole) [int]
    :param all_parts:  Flag for whether to use all particles (SNAP group)
                       or only particles in halos (PARTDATA group)  [bool]
    :param sorted:     Flag for whether to produce indices in a sorted particle ID array
                       or unsorted (order they are stored in) [bool]
    :return:
    '''

    # Get the particle IDs for this particle type using eagle_IO
    if all_parts:

        # Get all particles in the simulation
        part_ids = E.read_array('SNAP', sim, snapshot, 'PartType' + str(part_type) + '/ParticleIDs',
                                numThreads=8)

        # Get only those particles in a halo
        group_part_ids = E.read_array('PARTDATA', sim, snapshot, 'PartType' + str(part_type) + '/ParticleIDs',
                                      numThreads=8)

    else:

        # Get only those particles in a halo
        part_ids = E.read_array('PARTDATA', sim, snapshot, 'PartType' + str(part_type) + '/ParticleIDs',
                                numThreads=8)

        # A copy of this array is needed for the extraction method
        group_part_ids = np.copy(part_ids)

    # Extract the group ID and subgroup ID each particle is contained within
    grp_ids = E.read_array('PARTDATA', sim, snapshot, 'PartType' + str(part_type) + '/GroupNumber',
                           numThreads=8)
    subgrp_ids = E.read_array('PARTDATA', sim, snapshot, 'PartType' + str(part_type) + '/SubGroupNumber',
                              numThreads=8)

    # Ensure no subgroup ID exceeds 99999
    assert subgrp_ids.max() < 99999, "Found too many subgroups, need to increase subgroup format string above %05d"

    # Remove particles not associated to a subgroup (subgroupnumber == 2**30 == 1073741824)
    okinds = subgrp_ids != 1073741824
    group_part_ids = group_part_ids[okinds]
    grp_ids = grp_ids[okinds]
    subgrp_ids = subgrp_ids[okinds]

    # Convert IDs to float(groupNumber.SubGroupNumber) format, i.e. group 1 subgroup 11 = 1.00011
    halo_ids = np.zeros(grp_ids.size, dtype=float)
    for (ind, g), sg in zip(enumerate(grp_ids), subgrp_ids):
        halo_ids[ind] = float(str(int(g)) + '.%05d' % int(sg))

    parts_in_groups, part_groups = get_part_inds(halo_ids, part_ids, group_part_ids, sorted)

    # Produce a dictionary containing the index of particles in each halo
    halo_part_inds = {}
    for ind, grp in zip(parts_in_groups, part_groups):
        halo_part_inds.setdefault(grp, set()).update({ind})

    # Now the dictionary is fully populated convert values from sets to arrays for indexing
    for key, val in halo_part_inds.items():
        halo_part_inds[key] = np.array(list(val))

    return halo_part_inds


def get_main(path, snap, reg):

    # Get the redshift
    z_str = snap.split('z')[1].split('p')
    z = float(z_str[0] + '.' + z_str[1])

    # Load all necessary arrays
    all_poss = E.read_array('PARTDATA', path, snap, 'PartType4/Coordinates', noH=True,
                            physicalUnits=True, numThreads=8)
    gal_sml = E.read_array('PARTDATA', path, snap, 'PartType4/SmoothingLength', noH=True,
                           physicalUnits=True, numThreads=8)
    grp_ids = E.read_array('PARTDATA', path, snap, 'PartType4/GroupNumber', numThreads=8)
    subgrp_ids = E.read_array('PARTDATA', path, snap, 'PartType4/SubGroupNumber', numThreads=8)
    gal_ids = E.read_array('SUBFIND', path, snap, 'Subhalo/SubGroupNumber', numThreads=8)
    gal_gids = E.read_array('SUBFIND', path, snap, 'Subhalo/GroupNumber', numThreads=8)
    gal_cops = E.read_array('SUBFIND', path, snap, 'Subhalo/CentreOfPotential', noH=True,
                            physicalUnits=True, numThreads=8)

    # Load data for luminosities
    a_born = E.read_array('PARTDATA', path, snap, 'PartType4/StellarFormationTime', noH=True,
                          physicalUnits=True, numThreads=8)
    metallicities = E.read_array('PARTDATA', path, snap, 'PartType4/SmoothedMetallicity', noH=True,
                                 physicalUnits=True, numThreads=8)
    masses = E.read_array('PARTDATA', path, snap, 'PartType4/Mass', noH=True, physicalUnits=True, numThreads=8) * 10**10

    print(gal_ids)
    print(gal_gids)
    print(gal_cops)

    # Remove particles not in a subgroup
    nosub_mask = subgrp_ids != 1073741824
    all_poss = all_poss[nosub_mask, :]
    gal_sml = gal_sml[nosub_mask]
    grp_ids = grp_ids[nosub_mask]
    subgrp_ids = subgrp_ids[nosub_mask]
    a_born = a_born[nosub_mask]
    metallicities = metallicities[nosub_mask]
    masses = masses[nosub_mask]

    # Convert IDs to float(groupNumber.SubGroupNumber) format, i.e. group 1 subgroup 11 = 1.00011
    halo_ids = np.zeros(grp_ids.size, dtype=float)
    for (ind, g), sg in zip(enumerate(grp_ids), subgrp_ids):
        halo_ids[ind] = float(str(int(g)) + '.%05d' % int(sg))

    # Calculate ages
    ages = calc_ages(z, a_born)

    # Get particle indices
    halo_part_inds = get_subgroup_part_inds(path, snap, part_type=4, all_parts=False, sorted=False)

    # Get the position of each of these galaxies
    gal_ages = {}
    gal_mets = {}
    gal_ms = {}
    gal_smls = {}
    all_gal_poss = {}
    means = {}
    for id, cop in zip(halo_ids, gal_cops):
        mask = halo_part_inds[id]
        id = float(str(int(g)) + '.' + str(int(sg)))
        all_gal_poss[id] = all_poss[mask, :]
        gal_ages[id] = ages[mask]
        gal_mets[id] = metallicities[mask]
        gal_ms[id] = masses[mask]
        gal_smls[id] = gal_sml[mask]
        means[id] = cop

    print('There are', len(gal_ages.keys()), 'galaxies')

    del subgrp_ids, ages, all_poss, metallicities, masses, gal_sml, a_born, grp_ids

    gc.collect()

    # If there are no galaxies exit
    if len(gal_ages.keys()) == 0:
        return
    
    print('Got galaxy properties')
    
    # Get gas particle information
    gas_all_poss = E.read_array('PARTDATA', path, snap, 'PartType0/Coordinates', noH=True, physicalUnits=True,
                                numThreads=8)
    ggrp_ids = E.read_array('PARTDATA', path, snap, 'PartType0/GroupNumber', numThreads=8)
    gsubgrp_ids = E.read_array('PARTDATA', path, snap, 'PartType0/SubGroupNumber', numThreads=8)
    gas_metallicities = E.read_array('PARTDATA', path, snap, 'PartType0/SmoothedMetallicity', noH=True,
                                     physicalUnits=True, numThreads=8)
    gas_smooth_ls = E.read_array('PARTDATA', path, snap, 'PartType0/SmoothingLength', noH=True, physicalUnits=True,
                                 numThreads=8)
    gas_masses = E.read_array('PARTDATA', path, snap, 'PartType0/Mass', noH=True, physicalUnits=True,
                              numThreads=8) * 10**10

    # Remove particles not in a subgroup
    nosub_mask = gsubgrp_ids != 1073741824
    gas_all_poss = gas_all_poss[nosub_mask, :]
    ggrp_ids = ggrp_ids[nosub_mask]
    gsubgrp_ids = gsubgrp_ids[nosub_mask]
    gas_metallicities = gas_metallicities[nosub_mask]
    gas_smooth_ls = gas_smooth_ls[nosub_mask]
    gas_masses = gas_masses[nosub_mask]

    # Get the position of each of these galaxies
    gas_mets = {}
    gas_ms = {}
    gas_smls = {}
    all_gas_poss = {}
    for g, sg in zip(gal_gids, gal_ids):
        mask = (ggrp_ids == g) & (gsubgrp_ids == sg)
        id = float(str(int(g)) + '.' + str(int(sg)))
        all_gas_poss[id] = gas_all_poss[mask, :]
        gas_mets[id] = gas_metallicities[mask]
        gas_ms[id] = gas_masses[mask]
        gas_smls[id] = gas_smooth_ls[mask]

    print('Got particle IDs')

    del ggrp_ids, gsubgrp_ids, gas_all_poss, gas_metallicities, gas_smooth_ls, gas_masses

    gc.collect()

    print('Got gas properties')

    save_dict = {'gal_ages': gal_ages, 'gal_mets': gal_mets, 'gal_ms': gal_ms, 'gas_mets': gas_mets,
                 'gas_ms': gas_ms, 'gal_smls': gal_smls, 'gas_smls': gas_smls, 'all_gas_poss': all_gas_poss,
                 'all_gal_poss': all_gal_poss, 'means': means}

    with open('UVimg_data/stellardata_reg' + reg + '_snap' + snap + '.pck', 'wb') as pfile1:
        pickle.dump(save_dict, pfile1)


regions = []
for reg in range(0, 40):
    if reg < 10:
        regions.append('0' + str(reg))
    else:
        regions.append(str(reg))

snaps = ['000_z015p000', '001_z014p000', '002_z013p000', '003_z012p000', '004_z011p000', '005_z010p000',
         '006_z009p000', '007_z008p000', '008_z007p000', '009_z006p000', '010_z005p000', '011_z004p770']

npart_lim=10**2

reg_snaps = []
for reg in regions:

    for snap in snaps:

        reg_snaps.append((reg, snap))

ind = int(sys.argv[1])
print(reg_snaps[ind])
reg, snap = reg_snaps[ind]

path = '/cosma/home/dp004/dc-rope1/FLARES/FLARES-1/G-EAGLE_' + reg + '/data'

files = os.listdir('UVimg_data/')

if 'stellardata_reg' + reg + '_snap' + snap + '.pck' in files:
    print('File Exists', reg, snap)
else:
    get_main(path, snap, reg)
