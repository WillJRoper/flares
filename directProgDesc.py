#!/cosma/home/dp004/dc-rope1/.conda/envs/flares-env/bin/python
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.colors import LogNorm
import h5py
import eagle_IO as E
import seaborn as sns
import pickle
import itertools
matplotlib.use('Agg')

sns.set_style('whitegrid')

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
        print(current_halo_pids)
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
        prog_npart = np.array([-1], copy=False, dtype=int)
        prog_haloids = np.array([-1], copy=False, dtype=int)
        prog_mass_contribution = np.array([-1], copy=False, dtype=int)
        preals = np.array([False], copy=False, dtype=bool)

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
        desc_npart = np.array([-1], copy=False, dtype=int)
        desc_haloids = np.array([-1], copy=False, dtype=int)
        desc_mass_contribution = np.array([-1], copy=False, dtype=int)

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
    # internal_to_flares_part_ids = E.read_array('PARTDATA', path, snap, 'PartType' + str(part_type) + '/ParticleIDs')
    if rank == 0:
        halo_ids = E.read_array('PARTDATA', path, snap, 'PartType' + str(part_type) + '/GroupNumber', numThreads=8)
    elif rank == 1:
        halo_ids = E.read_array('PARTDATA', path, snap, 'PartType' + str(part_type) + '/SubGroupNumber', numThreads=8)
    else:
        raise ValueError("Incompatible rank")

    max_sim_id = np.max(halo_ids[np.where(halo_ids != 2 ** 30)])

    halo_id_part_inds = {}
    sim_to_internal_haloid = np.full(int(max_sim_id) + 1, -2, dtype=int)
    internal_to_sim_haloid = np.zeros(len(np.unique(halo_ids)) - 1, dtype=int)
    internalIDcount = -1
    for ind, simid in enumerate(halo_ids):
        simid = int(simid)
        if simid == 2**30:
            continue
        if sim_to_internal_haloid[simid] == -2:
            internalIDcount += 1
            internalID = internalIDcount
            internal_to_sim_haloid[internalID] = simid
            sim_to_internal_haloid[simid] = internalID
            halo_id_part_inds.setdefault(internalID, set()).update({ind})
        else:
            internalID = sim_to_internal_haloid[simid]
            halo_id_part_inds.setdefault(internalID, set()).update({ind})
        print('Creating halo to contained particle mapping:', ind, 'of', len(halo_ids), end='\r')

    # =============== Progenitor Snapshot ===============

    # Only look for descendant data if there is a descendant snapshot
    if prog_snap != None:

        if rank == 0:
            prog_halo_ids = E.read_array('PARTDATA', path, prog_snap, 'PartType' + str(part_type) + '/GroupNumber',
                                         numThreads=8)
        else:
            prog_halo_ids = E.read_array('PARTDATA', path, prog_snap, 'PartType' + str(part_type) +
                                         '/SubGroupNumber', numThreads=8)

        max_sim_id = np.max(prog_halo_ids[np.where(prog_halo_ids != 2**30)])
        print(max_sim_id, len(np.unique(prog_halo_ids)), len(prog_halo_ids))

        sim_to_internal_haloid_prog = np.full(int(max_sim_id) + 1, -2, dtype=int)
        internal_to_sim_haloid_prog = np.zeros(len(np.unique(prog_halo_ids)) - 1, dtype=int)
        prog_snap_haloIDs = np.full_like(prog_halo_ids, -2)
        internalIDcount = -1
        for ind, simid in enumerate(prog_halo_ids):
            simid = int(simid)
            if simid == 2 ** 30:
                continue
            if sim_to_internal_haloid_prog[simid] == -2:
                internalIDcount += 1
                internalID = internalIDcount
                internal_to_sim_haloid_prog[internalID] = simid
                sim_to_internal_haloid_prog[simid] = internalID
                prog_snap_haloIDs[ind] = internalID
            else:
                internalID = sim_to_internal_haloid_prog[simid]
                prog_snap_haloIDs[ind] = internalID
            print('Mapping progenitor halos to internal ID:', ind, 'of', len(halo_ids), end='\r')
            
        # Get all the unique halo IDs in this snapshot and the number of times they appear
        prog_unique, prog_counts = np.unique(prog_snap_haloIDs, return_counts=True)

        # Remove single particle halos (ID=-2), since np.unique returns a sorted array this can be
        # done by removing the first value
        prog_unique = prog_unique[1:]
        prog_counts = prog_counts[1:]

    else:  # Assign an empty array if the snapshot is less than the earliest (000)
        prog_snap_haloIDs = np.array([], copy=False)
        internal_to_sim_haloid_prog = np.array([], copy=False)
        prog_counts = []

    # =============== Descendant Snapshot ===============

    # Only look for descendant data if there is a descendant snapshot
    if desc_snap != None:

        if rank == 0:
            desc_halo_ids = E.read_array('PARTDATA', path, desc_snap, 'PartType' + str(part_type) + '/GroupNumber',
                                         numThreads=8)
        else:
            desc_halo_ids = E.read_array('PARTDATA', path, desc_snap, 'PartType' + str(part_type) +
                                         '/SubGroupNumber', numThreads=8)

        max_sim_id = np.max(desc_halo_ids[np.where(desc_halo_ids != 2**30)])
        print(max_sim_id, len(np.unique(desc_halo_ids)))

        sim_to_internal_haloid_desc = np.full(int(max_sim_id) + 1, -2, dtype=int)
        internal_to_sim_haloid_desc = np.zeros(len(np.unique(desc_halo_ids)) - 1, dtype=int)
        desc_snap_haloIDs = np.full_like(desc_halo_ids, -2)
        internalIDcount = -1
        for ind, simid in enumerate(desc_halo_ids):
            simid = int(simid)
            if simid == 2 ** 30:
                continue
            if sim_to_internal_haloid_desc[simid] == -2:
                internalIDcount += 1
                internalID = internalIDcount
                internal_to_sim_haloid_desc[internalID] = simid
                sim_to_internal_haloid_desc[simid] = internalID
                desc_snap_haloIDs[ind] = internalID
            else:
                internalID = sim_to_internal_haloid_desc[simid]
                desc_snap_haloIDs[ind] = internalID
            print('Mapping descendant halos to internal ID:', ind, 'of', len(halo_ids), end='\r')

        # Get all the unique halo IDs in this snapshot and the number of times they appear
        desc_unique, desc_counts = np.unique(desc_snap_haloIDs, return_counts=True)

        # Remove single particle halos (ID=-2), since np.unique returns a sorted array this can be
        # done by removing the first value
        desc_unique = desc_unique[1:]
        desc_counts = desc_counts[1:]

    else:  # Assign an empty array if the snapshot is less than the earliest (000)
        desc_snap_haloIDs = np.array([], copy=False)
        internal_to_sim_haloid_desc = np.array([], copy=False)
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

        # =============== Current Halo ===============

        current_halo_pids = np.array(list(halo_id_part_inds[haloID]))

        # =============== Run The Direct Progenitor and Descendant Finder ===============

        # Run the progenitor/descendant finder
        results[haloID] = getLinks(current_halo_pids, prog_snap_haloIDs, desc_snap_haloIDs,
                          prog_counts, desc_counts)

        if rank == 0:

            hdf = h5py.File(savepath + 'Mgraph_' + snap + '_PartType' + part_type +'.hdf5', 'w')

        else:

            hdf = h5py.File(savepath + 'SubMgraph_' + snap + '_PartType' + part_type +'.hdf5', 'w')

        for num, haloID in enumerate(results.keys()):

            (nprog, prog_haloids, prog_npart, prog_mass_contribution,
             ndesc, desc_haloids, desc_npart, desc_mass_contribution, current_halo_pids) = results[haloID]
            
            simHaloID = internal_to_sim_haloid[int(haloID)]
            
            sim_prog_haloids = np.zeros_like(prog_haloids)
            for ind, prog in enumerate(prog_haloids):
                sim_prog_haloids[ind] = internal_to_sim_haloid_prog[prog]

            sim_desc_haloids = np.zeros_like(desc_haloids)
            for ind, desc in enumerate(desc_haloids):
                sim_desc_haloids[ind] = internal_to_sim_haloid_desc[desc]

            # Print progress
            previous_progress = progress
            progress = int(num / size * 100)
            if progress != previous_progress:
                print('Write progress: ', progress, '%', haloID, end='\r')

            # Write out the data produced
            halo = hdf.create_group(simHaloID)  # create halo group
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
            halo.create_dataset('Prog_haloIDs', data=sim_prog_haloids, dtype=int,
                                compression='gzip')  # progenitor IDs
            halo.create_dataset('Desc_haloIDs', data=sim_desc_haloids, dtype=int,
                                compression='gzip')  # descendant IDs

        hdf.close()

if __name__ == '__main__':
    mainDirectProgDesc(snap='001_z014p000', prog_snap='000_z015p000', desc_snap='002_z013p000',
                       path='/cosma/home/dp004/dc-rope1/FLARES/FLARES-1/G-EAGLE_38/data', part_type=0,
                       rank=0, savepath='/cosma/home/dp004/dc-rope1/cosma7/FLARES/MergerGraphs/')
