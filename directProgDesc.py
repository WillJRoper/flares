#!/cosma/home/dp004/dc-rope1/.conda/envs/flares-env/bin/python
import numpy as np
import matplotlib
import h5py
import eagle_IO as E
import os
import sys
matplotlib.use('Agg')


def getLinks(current_halo_pids, prog_snap_haloIDs, desc_snap_haloIDs,
             prog_counts, desc_counts):
    """

    :param current_halo_pids:
    :param prog_snap_haloIDs:
    :param desc_snap_haloIDs:
    :param prog_counts:
    :param desc_counts:
    :param part_threshold:
    :return:
    """

    # =============== Find Progenitor IDs ===============

    # If any progenitor halos exist (i.e. The current snapshot ID is not 000, enforced in the main function)
    if prog_snap_haloIDs.size != 0:

        # Find the halo IDs of the current halo's particles in the progenitor snapshot by indexing the
        # progenitor snapshot's particle halo IDs array with the halo's particle IDs, this can be done
        # since the particle halo IDs array is sorted by particle ID.
        prog_haloids = prog_snap_haloIDs[current_halo_pids]

        # Find the unique halo IDs and the number of times each appears
        uniprog_haloids, uniprog_counts = np.unique(prog_haloids, return_counts=True)

        # Remove single particle halos (ID=-2), since np.unique returns a sorted array this can be
        # done by removing the first value.
        if uniprog_haloids[0] == -2:
            uniprog_haloids = uniprog_haloids[1:]
            uniprog_counts = uniprog_counts[1:]

        uniprog_haloids = uniprog_haloids[np.where(uniprog_counts >= 10)]
        uniprog_counts = uniprog_counts[np.where(uniprog_counts >= 10)]

        # Find the number of progenitor halos from the size of the unique array
        nprog = uniprog_haloids.size

        # Assign the corresponding number of particles in each progenitor for sorting and storing
        # This can be done simply by using the ID of the progenitor since again np.unique returns
        # sorted results.
        prog_npart = prog_counts[uniprog_haloids]

        # Sort the halo IDs and number of particles in each progenitor halo by their contribution to the
        # current halo (number of particles from the current halo in the progenitor or descendant)
        sorting_inds = uniprog_counts.argsort()[::-1]
        prog_npart = prog_npart[sorting_inds]
        prog_haloids = uniprog_haloids[sorting_inds]
        prog_mass_contribution = uniprog_counts[sorting_inds]

    # If there is no progenitor store Null values
    else:
        nprog = -1
        prog_npart = np.array([], copy=False, dtype=int)
        prog_haloids = np.array([], copy=False, dtype=int)
        prog_mass_contribution = np.array([], copy=False, dtype=int)

    # =============== Find Descendant IDs ===============

    # If descendant halos exist (i.e. The current snapshot ID is not 061, enforced in the main function)
    if desc_snap_haloIDs.size != 0:

        # Find the halo IDs of the current halo's particles in the descendant snapshot by indexing the
        # descendant snapshot's particle halo IDs array with the halo's particle IDs, this can be done
        # since the particle halo IDs array is sorted by particle ID.
        desc_haloids = desc_snap_haloIDs[current_halo_pids]

        # Find the unique halo IDs and the number of times each appears
        unidesc_haloids, unidesc_counts = np.unique(desc_haloids, return_counts=True)

        # Remove single particle halos (ID=-2) for the counts, since np.unique returns a sorted array this can be
        # done by removing the first value.
        if unidesc_haloids[0] == -2:
            unidesc_haloids = unidesc_haloids[1:]
            unidesc_counts = unidesc_counts[1:]

        unidesc_haloids = unidesc_haloids[np.where(unidesc_counts >= 10)]
        unidesc_counts = unidesc_counts[np.where(unidesc_counts >= 10)]

        # Find the number of descendant halos from the size of the unique array
        ndesc = unidesc_haloids.size

        # Assign the corresponding number of particles in each descendant for storing.
        # Could be extracted later from halo data but make analysis faster to save it here.
        # This can be done simply by using the ID of the descendant since again np.unique returns
        # sorted results.
        desc_npart = desc_counts[unidesc_haloids]

        # Sort the halo IDs and number of particles in each progenitor halo by their contribution to the
        # current halo (number of particles from the current halo in the progenitor or descendant)
        sorting_inds = unidesc_counts.argsort()[::-1]
        desc_npart = desc_npart[sorting_inds]
        desc_haloids = unidesc_haloids[sorting_inds]
        desc_mass_contribution = unidesc_counts[sorting_inds]

    # If there is no descendant snapshot store Null values
    else:
        ndesc = -1
        desc_npart = np.array([], copy=False, dtype=int)
        desc_haloids = np.array([], copy=False, dtype=int)
        desc_mass_contribution = np.array([], copy=False, dtype=int)

    return (nprog, prog_haloids, prog_npart, prog_mass_contribution,
            ndesc, desc_haloids, desc_npart, desc_mass_contribution,
            current_halo_pids)


def mainDirectProgDesc(snap, prog_snap, desc_snap, path, part_type, savepath='MergerGraphs/'):
    """ A function which cycles through all halos in a snapshot finding and writing out the
    direct progenitor and descendant data.

    :param snapshot: The snapshot ID.
    :param halopath: The filepath to the halo finder HDF5 file.
    :param savepath: The filepath to the directory where the Merger Graph should be written out to.
    :param part_threshold: The mass (number of particles) threshold defining a halo.

    :return: None
    """

    # =============== Current Snapshot ===============

    # Extract the halo IDs (group names/keys) contained within this snapshot
    part_ids = E.read_array('SNAP', path, snap, 'PartType' + str(part_type) + '/ParticleIDs', numThreads=8)
    group_part_ids = E.read_array('PARTDATA', path, snap, 'PartType' + str(part_type) + '/ParticleIDs',
                                  numThreads=8)
    grp_ids = E.read_array('PARTDATA', path, snap, 'PartType' + str(part_type) + '/GroupNumber', numThreads=8)
    subgrp_ids = E.read_array('PARTDATA', path, snap, 'PartType' + str(part_type) + '/SubGroupNumber', numThreads=8)

    # Remove particles not associated to a subgroup
    group_part_ids = group_part_ids[subgrp_ids != 1073741824]
    grp_ids = grp_ids[subgrp_ids != 1073741824]
    subgrp_ids = subgrp_ids[subgrp_ids != 1073741824]
    halo_ids = np.zeros(grp_ids.size, dtype=float)
    for (ind, g), sg in zip(enumerate(grp_ids), subgrp_ids):
        halo_ids[ind] = float(str(int(g)) + '.' + str(int(sg) + 1))

    # Sort particle IDS
    part_ids = np.sort(part_ids)
    unsort_part_ids = part_ids[:]
    sinds = np.argsort(part_ids)
    part_ids = part_ids[sinds]

    sorted_index = np.searchsorted(part_ids, group_part_ids)

    yindex = np.take(sinds, sorted_index, mode="clip")
    mask = unsort_part_ids[yindex] != group_part_ids

    result = np.ma.array(yindex, mask=mask)

    part_groups = halo_ids[np.logical_not(result.mask)]
    parts_in_groups = result.data[np.logical_not(result.mask)]

    halo_id_part_inds = {}
    for ind, grp in zip(parts_in_groups, part_groups):
        halo_id_part_inds.setdefault(grp, set()).update({ind})

    del group_part_ids, halo_ids, subgrp_ids, part_ids
    gc.collect()

    # =============== Progenitor Snapshot ===============

    # Only look for descendant data if there is a descendant snapshot
    if prog_snap != None:

        prog_allpart_ids = E.read_array('SNAP', path, prog_snap, 'PartType' + str(part_type) + '/ParticleIDs',
                                        numThreads=8)
        grp_ids = E.read_array('PARTDATA', path, prog_snap, 'PartType' + str(part_type) + '/GroupNumber',
                               numThreads=8)
        subgrp_ids = E.read_array('PARTDATA', path, prog_snap, 'PartType' + str(part_type) + '/SubGroupNumber',
                                  numThreads=8)
        prog_part_ids = E.read_array('PARTDATA', path, prog_snap, 'PartType' + str(part_type) + '/ParticleIDs',
                                     numThreads=8)

        # Remove particles not associated to a subgroup
        prog_part_ids = prog_part_ids[subgrp_ids != 1073741824]
        grp_ids = grp_ids[subgrp_ids != 1073741824]
        subgrp_ids = subgrp_ids[subgrp_ids != 1073741824]
        prog_halo_ids = np.zeros(grp_ids.size, dtype=float)
        for (ind, g), sg in zip(enumerate(grp_ids), subgrp_ids):
            prog_halo_ids[ind] = float(str(int(g)) + '.' + str(int(sg) + 1))

        # Sort particle IDS
        prog_allpart_ids = np.sort(prog_allpart_ids)
        unsort_part_ids = prog_allpart_ids[:]
        sinds = np.argsort(prog_allpart_ids)
        prog_allpart_ids = prog_allpart_ids[sinds]

        sorted_index = np.searchsorted(prog_allpart_ids, prog_part_ids)

        yindex = np.take(sinds, sorted_index, mode="clip")
        mask = unsort_part_ids[yindex] != prog_part_ids

        result = np.ma.array(yindex, mask=mask)

        part_groups = prog_halo_ids[np.logical_not(result.mask)]
        parts_in_groups = result.data[np.logical_not(result.mask)]
        
        prog_snap_haloIDs = np.full(len(prog_allpart_ids), -2, dtype=int)
        internal_to_sim_haloID_prog = {}
        sim_to_internal_haloID_prog = {}
        internalID = -1
        for ind, prog in zip(parts_in_groups, part_groups):
            if prog in sim_to_internal_haloID_prog.keys():
                prog_snap_haloIDs[ind] = sim_to_internal_haloID_prog[prog]
            else:
                internalID += 1
                sim_to_internal_haloID_prog[prog] = internalID
                internal_to_sim_haloID_prog[internalID] = prog
                prog_snap_haloIDs[ind] = internalID
            
        # Get all the unique halo IDs in this snapshot and the number of times they appear
        prog_unique, prog_counts = np.unique(prog_snap_haloIDs, return_counts=True)

        # Remove single particle halos (ID=-2), since np.unique returns a sorted array this can be
        # done by removing the first value
        prog_unique = prog_unique[1:]
        prog_counts = prog_counts[1:]

    else:  # Assign an empty array if the snapshot is less than the earliest (000)
        prog_snap_haloIDs = np.array([], copy=False)
        internal_to_sim_haloID_prog = {}
        sim_to_internal_haloID_prog = {}
        prog_counts = []

    # =============== Descendant Snapshot ===============

    # Only look for descendant data if there is a descendant snapshot
    if desc_snap != None:

        desc_allpart_ids = E.read_array('SNAP', path, desc_snap, 'PartType' + str(part_type) + '/ParticleIDs',
                                        numThreads=8)
        grp_ids = E.read_array('PARTDATA', path, desc_snap, 'PartType' + str(part_type) + '/GroupNumber',
                               numThreads=8)
        subgrp_ids = E.read_array('PARTDATA', path, desc_snap, 'PartType' + str(part_type) + '/SubGroupNumber',
                                  numThreads=8)
        desc_part_ids = E.read_array('PARTDATA', path, desc_snap, 'PartType' + str(part_type) + '/ParticleIDs',
                                     numThreads=8)

        # Remove particles not associated to a subgroup
        desc_part_ids = desc_part_ids[subgrp_ids != 1073741824]
        grp_ids = grp_ids[subgrp_ids != 1073741824]
        subgrp_ids = subgrp_ids[subgrp_ids != 1073741824]
        desc_halo_ids = np.zeros(grp_ids.size, dtype=float)
        for (ind, g), sg in zip(enumerate(grp_ids), subgrp_ids):
            desc_halo_ids[ind] = float(str(int(g)) + '.' + str(int(sg) + 1))

        # Sort particle IDS
        desc_allpart_ids = np.sort(desc_allpart_ids)
        unsort_part_ids = desc_allpart_ids[:]
        sinds = np.argsort(desc_allpart_ids)
        desc_allpart_ids = desc_allpart_ids[sinds]

        sorted_index = np.searchsorted(desc_allpart_ids, desc_part_ids)

        yindex = np.take(sinds, sorted_index, mode="clip")
        mask = unsort_part_ids[yindex] != desc_part_ids

        result = np.ma.array(yindex, mask=mask)

        part_groups = desc_halo_ids[np.logical_not(result.mask)]
        parts_in_groups = result.data[np.logical_not(result.mask)]

        desc_snap_haloIDs = np.full(len(desc_allpart_ids), -2, dtype=int)
        internal_to_sim_haloID_desc = {}
        sim_to_internal_haloID_desc = {}
        internalID = -1
        for ind, desc in zip(parts_in_groups, part_groups):
            if desc in sim_to_internal_haloID_desc.keys():
                desc_snap_haloIDs[ind] = sim_to_internal_haloID_desc[desc]
            else:
                internalID += 1
                sim_to_internal_haloID_desc[desc] = internalID
                internal_to_sim_haloID_desc[internalID] = desc
                desc_snap_haloIDs[ind] = internalID

        # Get all the unique halo IDs in this snapshot and the number of times they appear
        desc_unique, desc_counts = np.unique(desc_snap_haloIDs, return_counts=True)

        # Remove single particle halos (ID=-2), since np.unique returns a sorted array this can be
        # done by removing the first value
        desc_unique = desc_unique[1:]
        desc_counts = desc_counts[1:]

    else:  # Assign an empty array if the snapshot is less than the earliest (000)
        desc_snap_haloIDs = np.array([], copy=False)
        internal_to_sim_haloID_desc = {}
        sim_to_internal_haloID_desc = {}
        desc_counts = []

    # =============== Find all Direct Progenitors And Descendant Of Halos In This Snapshot ===============

    # Initialise the progress
    progress = -1

    # Assign the number of halos for progress reporting
    size = len(halo_id_part_inds.keys())
    results = {}

    # Loop through all the halos in this snapshot
    for num, haloID in enumerate(halo_id_part_inds.keys()):

        # Print progress
        previous_progress = progress
        progress = int(num / size * 100)
        if progress != previous_progress:
            print('Graph progress: ', progress, '%', haloID, end='\r')

        if int(str(haloID).split('.')[1]) == 0:
            continue

        # =============== Current Halo ===============

        current_halo_pids = np.array(list(halo_id_part_inds[haloID]))

        # =============== Run The Direct Progenitor and Descendant Finder ===============

        # Run the progenitor/descendant finder
        results[haloID] = getLinks(current_halo_pids, prog_snap_haloIDs, desc_snap_haloIDs,
                          prog_counts, desc_counts)

    hdf = h5py.File(savepath + 'SubMgraph_' + snap + '_PartType' + str(part_type) +'.hdf5', 'w')

    for num, haloID in enumerate(results.keys()):

        (nprog, prog_haloids, prog_npart, prog_mass_contribution,
         ndesc, desc_haloids, desc_npart, desc_mass_contribution, current_halo_pids) = results[haloID]

        sim_prog_haloids = np.zeros_like(prog_haloids)
        for ind, prog in enumerate(prog_haloids):
            sim_prog_haloids[ind] = internal_to_sim_haloID_prog[prog]

        sim_desc_haloids = np.zeros_like(desc_haloids)
        for ind, desc in enumerate(desc_haloids):
            sim_desc_haloids[ind] = internal_to_sim_haloID_desc[desc]

        # Print progress
        previous_progress = progress
        progress = int(num / size * 100)
        if progress != previous_progress:
            print('Write progress: ', progress, '%', haloID, end='\r')

        # Write out the data produced
        halo = hdf.create_group(str(haloID))  # create halo group
        halo.attrs['nProg'] = nprog  # number of progenitors
        halo.attrs['nDesc'] = ndesc  # number of descendants
        halo.attrs['current_halo_nPart'] = current_halo_pids.size  # mass of the halo
        # halo.create_dataset('current_halo_partIDs', data=current_halo_pids, dtype=int,
        #                     compression='gzip')  # particle ids in this halo
        halo.create_dataset('prog_npart_contribution', data=prog_mass_contribution, dtype=int,
                            compression='gzip')  # Mass contribution
        halo.create_dataset('desc_npart_contribution', data=desc_mass_contribution, dtype=int,
                            compression='gzip')  # Mass contribution
        halo.create_dataset('Prog_nPart', data=prog_npart, dtype=int,
                            compression='gzip')  # number of particles in each progenitor
        halo.create_dataset('Desc_nPart', data=desc_npart, dtype=int,
                            compression='gzip')  # number of particles in each descendant
        halo.create_dataset('Prog_haloIDs', data=sim_prog_haloids, dtype=float,
                            compression='gzip')  # progenitor IDs
        halo.create_dataset('Desc_haloIDs', data=sim_desc_haloids, dtype=float,
                            compression='gzip')  # descendant IDs

    hdf.close()

regions = []
for reg in range(0, 40):
    if reg < 10:
        regions.append('0' + str(reg))
    else:
        regions.append(str(reg))

snaps = ['000_z015p000', '001_z014p000', '002_z013p000', '003_z012p000', '004_z011p000', '005_z010p000',
         '006_z009p000', '007_z008p000', '008_z007p000', '009_z006p000', '010_z005p000', '011_z004p770']
prog_snaps = [None, '000_z015p000', '001_z014p000', '002_z013p000', '003_z012p000', '004_z011p000', '005_z010p000',
              '006_z009p000', '007_z008p000', '008_z007p000', '009_z006p000', '010_z005p000']
desc_snaps = ['001_z014p000', '002_z013p000', '003_z012p000', '004_z011p000', '005_z010p000',
              '006_z009p000', '007_z008p000', '008_z007p000', '009_z006p000', '010_z005p000', '011_z004p770', None]

reg_snaps = []
for reg in regions:

    try:
        os.mkdir('/cosma/home/dp004/dc-rope1/FLARES/FLARES-1/MergerGraphs/GEAGLE_' + reg)
    except OSError:
        pass

    for snap, prog_snap, desc_snap in zip(snaps, prog_snaps, desc_snaps):

        reg_snaps.append((reg, snap, prog_snap, desc_snap))

if __name__ == '__main__':

    ind = int(sys.argv[1])
    print(reg_snaps[ind])

    mainDirectProgDesc(snap=reg_snaps[ind][1], prog_snap=reg_snaps[ind][2], desc_snap=reg_snaps[ind][3],
                       path='/cosma/home/dp004/dc-rope1/FLARES/FLARES-1/G-EAGLE_' + reg_snaps[ind][0] + '/data',
                       part_type=1,
                       savepath='/cosma/home/dp004/dc-rope1/FLARES/FLARES-1/MergerGraphs/GEAGLE_' 
                                + reg_snaps[ind][0] + '/')
