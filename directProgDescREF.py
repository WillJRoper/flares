#!/cosma/home/dp004/dc-rope1/.conda/envs/flares-env/bin/python
import numpy as np
import matplotlib
import h5py
import eagle_IO as E
import os
import gc
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


def mainDirectProgDesc(snap, prog_snap, desc_snap, path, part_type, rank, savepath='MergerGraphs/'):
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
    part_ids = np.uint64(E.read_array('SNAP', path, snap, 'PartType' + str(part_type) + '/ParticleIDs', numThreads=8))
    group_part_ids = np.uint64(E.read_array('PARTDATA', path, snap, 'PartType' + str(part_type) + '/ParticleIDs',
                                            numThreads=8))
    # print(internal_to_flares_part_ids.size)
    if rank == 0:
        halo_ids = E.read_array('PARTDATA', path, snap, 'PartType' + str(part_type) + '/GroupNumber', numThreads=8)
    elif rank == 1:
        grp_ids = E.read_array('PARTDATA', path, snap, 'PartType' + str(part_type) + '/GroupNumber', numThreads=8)
        subgrp_ids = E.read_array('PARTDATA', path, snap, 'PartType' + str(part_type) + '/SubGroupNumber', numThreads=8)
        halo_ids = np.zeros_like(grp_ids, dtype=float)
        for (ind, g), sg in zip(enumerate(grp_ids), subgrp_ids):
            halo_ids[ind] = float(str(int(g)) + '.' + str(int(sg) + 1))

        del grp_ids, subgrp_ids
        gc.collect()
    else:
        raise ValueError("Incompatible rank")

    set_group_part_ids = set(group_part_ids)
    print(len(part_ids), 'particles')
    print(len(group_part_ids), 'particles in halos')
    print(len(set(halo_ids)), 'halos')
    print(np.max(group_part_ids), np.min(group_part_ids))

    # ind_to_pid = np.full_like(part_ids, len(part_ids))
    # pid_to_ind = {}
    # for ind, pid in enumerate(part_ids):
    #     print(ind, pid)
    #     if pid in set_group_part_ids:
    #         ind_to_pid[ind] = pid
    #         pid_to_ind[pid] = ind
    #     # if ind % 10000000 == 0:
    #     #     print('Mapping particle IDs to index:', pid, 'to', ind, 'of', len(part_ids), end='\r')

    halo_id_part_inds = {}
    for ind, pid in enumerate(part_ids):
        print(ind, pid)
        if pid in set_group_part_ids:
            simid = halo_ids[group_part_ids == pid]
            print(simid)
            if int(str(simid).split('.')[rank]) == 2**30:
                continue
            halo_id_part_inds.setdefault(simid, set()).update({ind})

    # =============== Progenitor Snapshot ===============

    # Only look for descendant data if there is a descendant snapshot
    if prog_snap != None:

        if rank == 0:
            prog_halo_ids = E.read_array('PARTDATA', path, prog_snap, 'PartType' + str(part_type) + '/GroupNumber',
                                         numThreads=8)
        else:
            grp_ids = E.read_array('PARTDATA', path, prog_snap, 'PartType' + str(part_type) + '/GroupNumber', numThreads=8)
            subgrp_ids = E.read_array('PARTDATA', path, prog_snap, 'PartType' + str(part_type) + '/SubGroupNumber',
                                      numThreads=8)
            prog_halo_ids = np.zeros_like(grp_ids, dtype=float)
            for (ind, g), sg in zip(enumerate(grp_ids), subgrp_ids):
                prog_halo_ids[ind] = float(str(g) + '.' + str(sg + 1))

            del grp_ids, subgrp_ids
            gc.collect()
        
        prog_part_ids = E.read_array('PARTDATA', path, prog_snap, 'PartType' + str(part_type) + '/ParticleIDs',
                                     numThreads=8)
        
        prog_snap_haloIDs = np.full(len(pid_to_ind.keys()), -2, dtype=int)
        internal_to_sim_haloID_prog = {}
        sim_to_internal_haloID_prog = {}
        internalID = -1
        for pid, prog in zip(prog_part_ids, prog_halo_ids):
            if prog in sim_to_internal_haloID_prog.keys():
                prog_snap_haloIDs[pid_to_ind[pid]] = sim_to_internal_haloID_prog[prog]
            else:
                internalID += 1
                sim_to_internal_haloID_prog[prog] = internalID
                internal_to_sim_haloID_prog[internalID] = prog
                prog_snap_haloIDs[pid_to_ind[pid]] = internalID
            
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

        if rank == 0:
            desc_halo_ids = E.read_array('PARTDATA', path, desc_snap, 'PartType' + str(part_type) + '/GroupNumber',
                                         numThreads=8)
        else:
            grp_ids = E.read_array('PARTDATA', path, desc_snap, 'PartType' + str(part_type) + '/GroupNumber', numThreads=8)
            subgrp_ids = E.read_array('PARTDATA', path, desc_snap, 'PartType' + str(part_type) + '/SubGroupNumber',
                                      numThreads=8)
            desc_halo_ids = np.zeros_like(grp_ids, dtype=float)
            for (ind, g), sg in zip(enumerate(grp_ids), subgrp_ids):
                desc_halo_ids[ind] = float(str(g) + '.' + str(sg + 1))

            del grp_ids, subgrp_ids
            gc.collect()

        desc_part_ids = E.read_array('PARTDATA', path, desc_snap, 'PartType' + str(part_type) + '/ParticleIDs',
                                     numThreads=8)

        desc_snap_haloIDs = np.full(len(pid_to_ind.keys()), -2, dtype=int)
        internal_to_sim_haloID_desc = {}
        sim_to_internal_haloID_desc = {}
        internalID = -1
        for pid, desc in zip(desc_part_ids, desc_halo_ids):
            if desc in sim_to_internal_haloID_desc.keys():
                desc_snap_haloIDs[pid_to_ind[pid]] = sim_to_internal_haloID_desc[desc]
            else:
                internalID += 1
                sim_to_internal_haloID_desc[desc] = internalID
                internal_to_sim_haloID_desc[internalID] = desc
                desc_snap_haloIDs[pid_to_ind[pid]] = internalID

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

        if int(str(haloID).split('.')[1]) == 0:
            continue

        # =============== Current Halo ===============

        current_halo_pids = np.array(list(halo_id_part_inds[haloID]))

        # =============== Run The Direct Progenitor and Descendant Finder ===============

        # Run the progenitor/descendant finder
        results[haloID] = getLinks(current_halo_pids, prog_snap_haloIDs, desc_snap_haloIDs,
                          prog_counts, desc_counts)

    if rank == 0:

        hdf = h5py.File(savepath + 'Mgraph_' + snap + '_PartType' + str(part_type) +'.hdf5', 'w')

    else:

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


def ascend(a):

    sa = np.sort(a)
    start = sa[0]
    count = 0
    for i in sa:
        print(i)
        if i != start + count:
            print(i, count, start+count)
            return
        count += 1

    print('There are no gaps')

    return

pre_snaps = ['000_z020p000', '003_z008p988', '006_z005p971', '009_z004p485', '012_z003p017', '015_z002p012',
             '018_z001p259', '021_z000p736', '024_z000p366', '027_z000p101', '001_z015p132', '004_z008p075',
             '007_z005p487', '010_z003p984', '013_z002p478', '016_z001p737', '019_z001p004', '022_z000p615',
             '025_z000p271', '028_z000p000', '002_z009p993', '005_z007p050', '008_z005p037', '011_z003p528',
             '014_z002p237', '017_z001p487', '020_z000p865', '023_z000p503', '026_z000p183']

snaps = np.zeros(29, dtype=object)
for s in pre_snaps:
    ind = int(s.split('_')[0])
    snaps[ind] = s

snaps = list(snaps)
prog_snaps = snaps[:-1]
prog_snaps.insert(0, None)
desc_snaps = snaps[1:]
desc_snaps.append(None)
print(snaps)
print(prog_snaps)
print(desc_snaps)
path = '/cosma7/data//Eagle/ScienceRuns/Planck1/L0100N1504/PE/REFERENCE/data/'

if __name__ == '__main__':

    ind = int(sys.argv[1])
    print(snaps[ind])

    mainDirectProgDesc(snap=snaps[ind], prog_snap=prog_snaps[ind], desc_snap=desc_snaps[ind],
                       path=path, part_type=1, rank=1,
                       savepath='/cosma/home/dp004/dc-rope1/FLARES/FLARES-1/MergerGraphs/REF/')
