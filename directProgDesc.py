#!/cosma/home/dp004/dc-rope1/.conda/envs/flares-env/bin/python
import numpy as np
import matplotlib
import h5py
import eagle_IO as E
import os
import gc
import sys
matplotlib.use('Agg')


def dmgetLinks(current_halo_pids, prog_snap_haloIDs, desc_snap_haloIDs,
             prog_counts, desc_counts, part_type):
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

        if part_type == 1:
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

        if part_type == 1:
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


def getLinks(current_halo_pids, prog_snap_haloIDs, desc_snap_haloIDs,
             progpart_masses, descpart_masses, prog_ms, prog_ids, desc_ms, desc_ids):
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
        
        # Get progenitor halo masses
        for p in uniprog_haloids:
            if len(prog_ms[prog_ids == p]) == 2:
                print("the issue is here")
                print(uniprog_haloids)
                print(prog_ids[prog_ids == p])
        prog_masses = np.array([prog_ms[prog_ids == p] for p in uniprog_haloids]).flatten()
        
        # Get progenitor particle masses
        prog_partmass_contributed = progpart_masses[current_halo_pids]
        
        # Combine contribution to entire halo
        prog_mass_contribution = np.array([np.sum(prog_partmass_contributed[prog_haloids == p]) 
                                           for p in uniprog_haloids])
        prog_mass_contribution = prog_mass_contribution[np.where(prog_mass_contribution > 0.0)]

        # Find the number of progenitor halos from the size of the unique array
        nprog = uniprog_haloids.size

        # Sort the halo IDs and number of particles in each progenitor halo by their contribution to the
        # current halo (number of particles from the current halo in the progenitor or descendant)
        sorting_inds = prog_mass_contribution.argsort()[::-1]
        prog_npart = prog_masses[sorting_inds]
        prog_haloids = uniprog_haloids[sorting_inds]
        prog_mass_contribution = prog_mass_contribution[sorting_inds]

    # If there is no progenitor store Null values
    else:
        nprog = -1
        prog_npart = np.array([], copy=False, dtype=int)
        prog_haloids = np.array([], copy=False, dtype=int)
        prog_mass_contribution = np.array([], copy=False, dtype=int)

    # =============== Find Descendant IDs ===============

    # If descendant halos exist (i.e. The current snapshot ID is not 061, enforced in the main function)
    if desc_snap_haloIDs.size != 0:

        # Find the halo IDs of the current halo's particles in the descenitor snapshot by indexing the
        # descenitor snapshot's particle halo IDs array with the halo's particle IDs, this can be done
        # since the particle halo IDs array is sorted by particle ID.
        desc_haloids = desc_snap_haloIDs[current_halo_pids]

        # Find the unique halo IDs and the number of times each appears
        unidesc_haloids, unidesc_counts = np.unique(desc_haloids, return_counts=True)

        # Remove single particle halos (ID=-2), since np.unique returns a sorted array this can be
        # done by removing the first value.
        if unidesc_haloids[0] == -2:
            unidesc_haloids = unidesc_haloids[1:]
            unidesc_counts = unidesc_counts[1:]

        # Get descenitor halo masses
        desc_masses = np.array([desc_ms[desc_ids == d] for d in unidesc_haloids]).flatten()

        # Get descenitor particle masses
        desc_partmass_contributed = descpart_masses[current_halo_pids]

        # Combine contribution to entire halo
        desc_mass_contribution = np.array([np.sum(desc_partmass_contributed[desc_haloids == p])
                                           for p in unidesc_haloids])
        desc_mass_contribution = desc_mass_contribution[np.where(desc_mass_contribution > 0.0)]

        # Find the number of descenitor halos from the size of the unique array
        ndesc = unidesc_haloids.size

        # Sort the halo IDs and number of particles in each descenitor halo by their contribution to the
        # current halo (number of particles from the current halo in the descenitor or descendant)
        sorting_inds = desc_mass_contribution.argsort()[::-1]
        desc_npart = desc_masses[sorting_inds]
        desc_haloids = unidesc_haloids[sorting_inds]
        desc_mass_contribution = desc_mass_contribution[sorting_inds]

    # If there is no descendant snapshot store Null values
    else:
        ndesc = -1
        desc_npart = np.array([], copy=False, dtype=int)
        desc_haloids = np.array([], copy=False, dtype=int)
        desc_mass_contribution = np.array([], copy=False, dtype=int)

    return (nprog, prog_haloids, prog_npart, prog_mass_contribution,
            ndesc, desc_haloids, desc_npart, desc_mass_contribution,
            current_halo_pids)


def partDirectProgDesc(snap, prog_snap, desc_snap, path, part_type):
    """ A function which cycles through all halos in a snapshot finding and writing out the
    direct progenitor and descendant data.

    :param snapshot: The snapshot ID.
    :param halopath: The filepath to the halo finder HDF5 file.
    :param savepath: The filepath to the directory where the Merger Graph should be written out to.
    :param part_threshold: The mass (number of particles) threshold defining a halo.

    :return: None
    """

    # Extract particle IDs, if not using dark matter this must be all particles present at all 3 snapshots
    if part_type == 1:
        part_ids = E.read_array('SNAP', path, snap, 'PartType' + str(part_type) + '/ParticleIDs', numThreads=8)

        preprogpart_masses = np.array([])
        predescpart_masses = np.array([])
        preprog_gal_ms = np.array([])
        preprog_sub_ids = np.array([])
        predesc_gal_ms = np.array([])
        predesc_sub_ids = np.array([])
    else:
        # Get the particle IDs in each snapshot
        snap_part_ids = E.read_array('SNAP', path, snap, 'PartType' + str(part_type) + '/ParticleIDs', numThreads=8)
        
        if prog_snap != None:
            progsnap_part_ids = E.read_array('SNAP', path, prog_snap, 'PartType' + str(part_type) + '/ParticleIDs',
                                             numThreads=8)
            # Get halo IDs and halo data
            prog_subgrp_ids = E.read_array('SUBFIND', path, prog_snap, 'Subhalo/SubGroupNumber', numThreads=8)
            prog_grp_ids = E.read_array('SUBFIND', path, prog_snap, 'Subhalo/GroupNumber', numThreads=8)
            preprog_gal_ms = E.read_array('SUBFIND', path, prog_snap, 'Subhalo/ApertureMeasurements/Mass/030kpc',
                                          noH=False,
                                          physicalUnits=False, numThreads=8)[:, part_type]
            preprogpart_masses = E.read_array('PARTDATA', path, prog_snap, 'PartType' + str(part_type) + '/Mass',
                                              noH=False,
                                              physicalUnits=False, numThreads=8)

            preprog_sub_ids = np.zeros(prog_grp_ids.size, dtype=float)
            for (ind, g), sg in zip(enumerate(prog_grp_ids), prog_subgrp_ids):
                preprog_sub_ids[ind] = float(str(int(g)) + '.' + str(int(sg)))
                
        else:
            progsnap_part_ids = np.array([])
            preprog_sub_ids = np.array([])
            preprogpart_masses = np.array([])
            preprog_gal_ms = np.array([])
            
        if desc_snap != None:
            descsnap_part_ids = E.read_array('SNAP', path, desc_snap, 'PartType' + str(part_type) + '/ParticleIDs',
                                             numThreads=8)
            # Get halo IDs and halo data
            desc_subgrp_ids = E.read_array('SUBFIND', path, desc_snap, 'Subhalo/SubGroupNumber', numThreads=8)
            desc_grp_ids = E.read_array('SUBFIND', path, desc_snap, 'Subhalo/GroupNumber', numThreads=8)
            predesc_gal_ms = E.read_array('SUBFIND', path, desc_snap, 'Subhalo/ApertureMeasurements/Mass/030kpc',
                                          noH=False,
                                          physicalUnits=False, numThreads=8)[:, part_type]
            predescpart_masses = E.read_array('PARTDATA', path, desc_snap, 'PartType' + str(part_type) + '/Mass',
                                              noH=False,
                                              physicalUnits=False, numThreads=8)

            predesc_sub_ids = np.zeros(desc_grp_ids.size, dtype=float)
            for (ind, g), sg in zip(enumerate(desc_grp_ids), desc_subgrp_ids):
                predesc_sub_ids[ind] = float(str(int(g)) + '.' + str(int(sg)))
        else:
            descsnap_part_ids = np.array([])
            predesc_sub_ids = np.array([])
            predescpart_masses = np.array([])
            predesc_gal_ms = np.array([])

        # Combine the particle id arrays and only take unique values
        part_ids = np.unique(np.concatenate([snap_part_ids, progsnap_part_ids, descsnap_part_ids]))

    # =============== Current Snapshot ===============

    # Extract the halo IDs (group names/keys) contained within this snapshot
    group_part_ids = E.read_array('PARTDATA', path, snap, 'PartType' + str(part_type) + '/ParticleIDs',
                                  numThreads=8)
    grp_ids = E.read_array('PARTDATA', path, snap, 'PartType' + str(part_type) + '/GroupNumber', numThreads=8)
    subgrp_ids = E.read_array('PARTDATA', path, snap, 'PartType' + str(part_type) + '/SubGroupNumber', numThreads=8)

    # Remove particles not associated to a subgroup
    okinds = subgrp_ids != 1073741824
    group_part_ids = group_part_ids[okinds]
    grp_ids = grp_ids[okinds]
    subgrp_ids = subgrp_ids[okinds]
    halo_ids = np.zeros(grp_ids.size, dtype=float)
    for (ind, g), sg in zip(enumerate(grp_ids), subgrp_ids):
        halo_ids[ind] = float(str(int(g)) + '.' + str(int(sg)))

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

    del group_part_ids, halo_ids, subgrp_ids
    gc.collect()

    # =============== Progenitor Snapshot ===============

    # Only look for descendant data if there is a descendant snapshot
    if prog_snap != None:

        grp_ids = E.read_array('PARTDATA', path, prog_snap, 'PartType' + str(part_type) + '/GroupNumber',
                               numThreads=8)
        subgrp_ids = E.read_array('PARTDATA', path, prog_snap, 'PartType' + str(part_type) + '/SubGroupNumber',
                                  numThreads=8)
        prog_part_ids = E.read_array('PARTDATA', path, prog_snap, 'PartType' + str(part_type) + '/ParticleIDs',
                                     numThreads=8)

        # Remove particles not associated to a subgroup
        okinds = subgrp_ids != 1073741824
        prog_part_ids = prog_part_ids[okinds]
        grp_ids = grp_ids[okinds]
        subgrp_ids = subgrp_ids[okinds]
        if part_type != 1:
            preprogpart_masses = preprogpart_masses[okinds]

        prog_halo_ids = np.zeros(grp_ids.size, dtype=float)
        for (ind, g), sg in zip(enumerate(grp_ids), subgrp_ids):
            prog_halo_ids[ind] = float(str(int(g)) + '.' + str(int(sg)))

        sorted_index = np.searchsorted(part_ids, prog_part_ids)

        yindex = np.take(sinds, sorted_index, mode="clip")
        mask = unsort_part_ids[yindex] != prog_part_ids

        result = np.ma.array(yindex, mask=mask)

        part_groups = prog_halo_ids[np.logical_not(result.mask)]
        parts_in_groups = result.data[np.logical_not(result.mask)]
        
        # Map halo IDs to continuous intergers
        internal_to_sim_haloID_prog = {}
        sim_to_internal_haloID_prog = {}
        for i, p in enumerate(np.unique(part_groups)):
            internal_to_sim_haloID_prog[i] = p
            sim_to_internal_haloID_prog[p] = i

        if part_type != 1:
            prog_snap_haloIDs = np.full(len(part_ids), -2, dtype=int)
            progpart_masses = np.full(len(part_ids), -2, dtype=float)
            for ind, prog, m in zip(parts_in_groups, part_groups, preprogpart_masses):
                prog_snap_haloIDs[ind] = sim_to_internal_haloID_prog[prog]
                progpart_masses[ind] = m

            prog_sub_ids = []
            prog_gal_ms = []
            for prog, m in zip(preprog_sub_ids, preprog_gal_ms):
                try:
                    print(prog, sim_to_internal_haloID_prog[prog], m)
                    prog_sub_ids.append(sim_to_internal_haloID_prog[prog])
                    prog_gal_ms.append(m)
                except KeyError:
                    continue

            prog_sub_ids = np.array(prog_sub_ids)
            prog_gal_ms = np.array(prog_gal_ms)
            
        else:
            prog_gal_ms = np.array([], copy=False)
            prog_sub_ids = np.array([], copy=False)
            prog_snap_haloIDs = np.full(len(part_ids), -2, dtype=int)
            progpart_masses = np.full(len(part_ids), -2, dtype=float)
            for ind, prog in zip(parts_in_groups, part_groups):
                prog_snap_haloIDs[ind] = sim_to_internal_haloID_prog[prog]
            
        # Get all the unique halo IDs in this snapshot and the number of times they appear
        prog_unique, prog_counts = np.unique(prog_snap_haloIDs, return_counts=True)

        # Remove single particle halos (ID=-2), since np.unique returns a sorted array this can be
        # done by removing the first value
        prog_unique = prog_unique[1:]
        prog_counts = prog_counts[1:]

    else:  # Assign an empty array if the snapshot is less than the earliest (000)
        prog_snap_haloIDs = np.array([], copy=False)
        progpart_masses = np.array([], copy=False)
        prog_gal_ms = np.array([], copy=False)
        prog_sub_ids = np.array([], copy=False)
        internal_to_sim_haloID_prog = {}
        sim_to_internal_haloID_prog = {}
        prog_counts = []

    # =============== Descendant Snapshot ===============

    # Only look for descendant data if there is a descendant snapshot
    if desc_snap != None:

        grp_ids = E.read_array('PARTDATA', path, desc_snap, 'PartType' + str(part_type) + '/GroupNumber',
                               numThreads=8)
        subgrp_ids = E.read_array('PARTDATA', path, desc_snap, 'PartType' + str(part_type) + '/SubGroupNumber',
                                  numThreads=8)
        desc_part_ids = E.read_array('PARTDATA', path, desc_snap, 'PartType' + str(part_type) + '/ParticleIDs',
                                     numThreads=8)

        # Remove particles not associated to a subgroup
        okinds = subgrp_ids != 1073741824
        desc_part_ids = desc_part_ids[okinds]
        grp_ids = grp_ids[okinds]
        subgrp_ids = subgrp_ids[okinds]

        if part_type != 1:
            predescpart_masses = predescpart_masses[okinds]

        desc_halo_ids = np.zeros(grp_ids.size, dtype=float)
        for (ind, g), sg in zip(enumerate(grp_ids), subgrp_ids):
            desc_halo_ids[ind] = float(str(int(g)) + '.' + str(int(sg)))

        sorted_index = np.searchsorted(part_ids, desc_part_ids)

        yindex = np.take(sinds, sorted_index, mode="clip")
        mask = unsort_part_ids[yindex] != desc_part_ids

        result = np.ma.array(yindex, mask=mask)

        part_groups = desc_halo_ids[np.logical_not(result.mask)]
        parts_in_groups = result.data[np.logical_not(result.mask)]

        # Map halo IDs to continuous intergers
        internal_to_sim_haloID_desc = {}
        sim_to_internal_haloID_desc = {}
        for i, p in enumerate(np.unique(part_groups)):
            internal_to_sim_haloID_desc[i] = p
            sim_to_internal_haloID_desc[p] = i

        if part_type != 1:
            desc_snap_haloIDs = np.full(len(part_ids), -2, dtype=int)
            descpart_masses = np.full(len(part_ids), -2, dtype=float)
            for ind, desc, m in zip(parts_in_groups, part_groups, predescpart_masses):
                desc_snap_haloIDs[ind] = sim_to_internal_haloID_desc[desc]
                descpart_masses[ind] = m
                
            desc_sub_ids = []
            desc_gal_ms = []
            for desc, m in zip(predesc_sub_ids, predesc_gal_ms):
                try:
                    desc_sub_ids.append(sim_to_internal_haloID_desc[desc])
                    desc_gal_ms.append(m)
                except KeyError:
                    continue

            desc_sub_ids = np.array(desc_sub_ids, dtype=int)
            desc_gal_ms = np.array(desc_gal_ms, dtype=float)
                
        else:
            desc_gal_ms = np.array([], copy=False)
            desc_sub_ids = np.array([], copy=False)
            desc_snap_haloIDs = np.full(len(part_ids), -2, dtype=int)
            descpart_masses = np.full(len(part_ids), -2, dtype=float)
            for ind, desc in zip(parts_in_groups, part_groups):
                desc_snap_haloIDs[ind] = sim_to_internal_haloID_desc[desc]

        # Get all the unique halo IDs in this snapshot and the number of times they appear
        desc_unique, desc_counts = np.unique(desc_snap_haloIDs, return_counts=True)

        # Remove single particle halos (ID=-2), since np.unique returns a sorted array this can be
        # done by removing the first value
        desc_unique = desc_unique[1:]
        desc_counts = desc_counts[1:]

    else:  # Assign an empty array if the snapshot is less than the earliest (000)
        desc_snap_haloIDs = np.array([], copy=False)
        descpart_masses = np.array([], copy=False)
        desc_sub_ids = np.array([], copy=False)
        desc_gal_ms = np.array([], copy=False)
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
            print('Graph progress: ', progress, '%', haloID)

        # =============== Current Halo ===============

        current_halo_pids = np.array(list(halo_id_part_inds[haloID]))

        # =============== Run The Direct Progenitor and Descendant Finder ===============

        # Run the progenitor/descendant finder
        if part_type == 1:
            results[haloID] = dmgetLinks(current_halo_pids, prog_snap_haloIDs, desc_snap_haloIDs,
                              prog_counts, desc_counts, part_type)
        else:
            results[haloID] = getLinks(current_halo_pids, prog_snap_haloIDs, desc_snap_haloIDs,
                                       progpart_masses, descpart_masses, prog_gal_ms, prog_sub_ids,
                                       desc_gal_ms, desc_sub_ids)
            
    print('Processed', len(results.keys()), 'halos in snapshot', snap, 'of particle type', part_type)

    return results, internal_to_sim_haloID_desc, internal_to_sim_haloID_prog


def mainDirectProgDesc(snap, prog_snap, desc_snap, path, savepath='MergerGraphs/'):

    results_dm, internal_to_sim_haloID_desc_dm, internal_to_sim_haloID_prog_dm = partDirectProgDesc(snap, prog_snap,
                                                                                                    desc_snap, path,
                                                                                                    part_type=1)
    results_stars, internal_to_sim_haloID_desc_st, internal_to_sim_haloID_prog_st = partDirectProgDesc(snap, prog_snap,
                                                                                                       desc_snap, path,
                                                                                                       part_type=4)
    results_bh, internal_to_sim_haloID_desc_bh, internal_to_sim_haloID_prog_bh = partDirectProgDesc(snap, prog_snap,
                                                                                                    desc_snap, path,
                                                                                                    part_type=5)
    results_gas, internal_to_sim_haloID_desc_gas, internal_to_sim_haloID_prog_gas = partDirectProgDesc(snap, prog_snap,
                                                                                                       desc_snap, path,
                                                                                                       part_type=0)

    size = len(results_dm.keys())

    # Initialise the progress
    progress = -1

    hdf = h5py.File(savepath + 'SubMgraph_' + snap + '.hdf5', 'w')

    for num, haloID in enumerate(results_dm.keys()):

        (nprog, prog_haloids, prog_npart, prog_mass_contribution,
         ndesc, desc_haloids, desc_npart, desc_mass_contribution, current_halo_pids) = results_dm[haloID]

        sim_prog_haloids = np.zeros(len(prog_haloids), dtype=float)
        for ind, prog in enumerate(prog_haloids):
            sim_prog_haloids[ind] = internal_to_sim_haloID_prog_dm[prog]

        sim_desc_haloids = np.zeros(len(desc_haloids), dtype=float)
        for ind, desc in enumerate(desc_haloids):
            sim_desc_haloids[ind] = internal_to_sim_haloID_desc_dm[desc]

        # Print progress
        previous_progress = progress
        progress = int(num / size * 100)
        if progress != previous_progress:
            print('Write progress: ', progress, '%', haloID)

        # Αssign dark matter haloids for comparison to other particle types
        dm_sim_desc_haloids = sim_desc_haloids[:]
        dm_sim_prog_haloids = sim_prog_haloids[:]

        # Write out the data produced
        halo = hdf.create_group(str(haloID))  # create halo group
        halo.attrs['nProg'] = nprog  # number of progenitors
        halo.attrs['nDesc'] = ndesc  # number of descendants
        halo.attrs['current_halo_nPart'] = current_halo_pids.size  # mass of the halo
        # halo.create_dataset('current_halo_part_inds', data=current_halo_pids, dtype=int)  # particle ids in this halo
        halo.create_dataset('prog_npart_contribution', data=prog_mass_contribution, dtype=int)  # Mass contribution
        halo.create_dataset('desc_npart_contribution', data=desc_mass_contribution, dtype=int)  # Mass contribution
        halo.create_dataset('Prog_nPart', data=prog_npart, dtype=int)  # number of particles in each progenitor
        halo.create_dataset('Desc_nPart', data=desc_npart, dtype=int)  # number of particles in each descendant
        halo.create_dataset('Prog_haloIDs', data=sim_prog_haloids, dtype=float)  # progenitor IDs
        halo.create_dataset('Desc_haloIDs', data=sim_desc_haloids, dtype=float)  # descendant IDs

        # Write out star data
        if haloID in results_stars:

            (nprog, prog_haloids, prog_npart, prog_mass_contribution,
             ndesc, desc_haloids, desc_npart, desc_mass_contribution, current_halo_pids) = results_stars[haloID]

            sim_prog_haloids = np.zeros(len(prog_haloids), dtype=float)
            for ind, prog in enumerate(prog_haloids):
                sim_prog_haloids[ind] = internal_to_sim_haloID_prog_st[prog]

            sim_desc_haloids = np.zeros(len(desc_haloids), dtype=float)
            for ind, desc in enumerate(desc_haloids):
                sim_desc_haloids[ind] = internal_to_sim_haloID_desc_st[desc]

            # Assign values to the corresponding index for the dark matter progenitors
            star_prog_mass_contribution = np.zeros(len(dm_sim_prog_haloids), dtype=float)
            star_prog_mass = np.zeros(len(dm_sim_prog_haloids), dtype=float)
            for p, cont, mass in zip(sim_prog_haloids, prog_mass_contribution, prog_npart):
                if not p in dm_sim_prog_haloids:
                    continue
                ind = np.where(dm_sim_prog_haloids == p)
                star_prog_mass_contribution[ind] = cont
                star_prog_mass[ind] = mass
                
            # Assign values to the corresponding index for the dark matter descendents
            star_desc_mass_contribution = np.zeros(len(dm_sim_desc_haloids), dtype=float)
            star_desc_mass = np.zeros(len(dm_sim_desc_haloids), dtype=float)
            for p, cont, mass in zip(sim_desc_haloids, desc_mass_contribution, desc_npart):
                if not p in dm_sim_desc_haloids:
                    continue
                ind = np.where(dm_sim_desc_haloids == p)
                star_desc_mass_contribution[ind] = cont
                star_desc_mass[ind] = mass

            halo.create_dataset('prog_stellar_mass_contribution', data=star_prog_mass_contribution,
                                dtype=float)  # Mass contribution
            halo.create_dataset('desc_stellar_mass_contribution', data=star_desc_mass_contribution,
                                dtype=float)  # Mass contribution
            halo.create_dataset('Prog_stellar_mass', data=star_prog_mass,
                                dtype=float)  # number of particles in each progenitor
            halo.create_dataset('Desc_stellar_mass', data=star_desc_mass,
                                dtype=float)  # number of particles in each descendant
        else:

            halo.create_dataset('prog_stellar_mass_contribution', data=np.array([]), dtype=float)  # Mass contribution
            halo.create_dataset('desc_stellar_mass_contribution', data=np.array([]), dtype=float)  # Mass contribution
            halo.create_dataset('Prog_stellar_mass', data=np.array([]),
                                dtype=float)  # number of particles in each progenitor
            halo.create_dataset('Desc_stellar_mass', data=np.array([]),
                                dtype=float)  # number of particles in each descendant

        # Write out gas data
        if haloID in results_gas:

            (nprog, prog_haloids, prog_npart, prog_mass_contribution,
             ndesc, desc_haloids, desc_npart, desc_mass_contribution, current_halo_pids) = results_gas[haloID]

            sim_prog_haloids = np.zeros(len(prog_haloids), dtype=float)
            for ind, prog in enumerate(prog_haloids):
                sim_prog_haloids[ind] = internal_to_sim_haloID_prog_gas[prog]

            sim_desc_haloids = np.zeros(len(desc_haloids), dtype=float)
            for ind, desc in enumerate(desc_haloids):
                sim_desc_haloids[ind] = internal_to_sim_haloID_desc_gas[desc]

            # Assign values to the corresponding index for the dark matter progenitors
            gas_prog_mass_contribution = np.zeros(len(dm_sim_prog_haloids), dtype=float)
            gas_prog_mass = np.zeros(len(dm_sim_prog_haloids), dtype=float)
            print(sim_prog_haloids, prog_mass_contribution, prog_npart)
            for p, cont, mass in zip(sim_prog_haloids, prog_mass_contribution, prog_npart):
                if not p in dm_sim_prog_haloids:
                    continue
                ind = np.where(dm_sim_prog_haloids == p)
                print(ind, mass)
                gas_prog_mass_contribution[ind] = cont
                gas_prog_mass[ind] = mass

            # Assign values to the corresponding index for the dark matter descendents
            gas_desc_mass_contribution = np.zeros(len(dm_sim_desc_haloids), dtype=float)
            gas_desc_mass = np.zeros(len(dm_sim_desc_haloids), dtype=float)
            for p, cont, mass in zip(sim_desc_haloids, desc_mass_contribution, desc_npart):
                if not p in dm_sim_desc_haloids:
                    continue
                ind = np.where(dm_sim_desc_haloids == p)
                gas_desc_mass_contribution[ind] = cont
                gas_desc_mass[ind] = mass

            halo.create_dataset('prog_gas_mass_contribution', data=gas_prog_mass_contribution,
                                dtype=float)  # Mass contribution
            halo.create_dataset('desc_gas_mass_contribution', data=gas_desc_mass_contribution,
                                dtype=float)  # Mass contribution
            halo.create_dataset('Prog_gas_mass', data=gas_prog_mass,
                                dtype=float)  # number of particles in each progenitor
            halo.create_dataset('Desc_gas_mass', data=gas_desc_mass,
                                dtype=float)  # number of particles in each descendant
        else:

            halo.create_dataset('prog_gas_mass_contribution', data=np.array([]),
                                dtype=int)  # Mass contribution
            halo.create_dataset('desc_gas_mass_contribution', data=np.array([]),
                                dtype=int)  # Mass contribution
            halo.create_dataset('Prog_gas_mass', data=np.array([]),
                                dtype=int)  # number of particles in each progenitor
            halo.create_dataset('Desc_gas_mass', data=np.array([]),
                                dtype=int)  # number of particles in each descendant
            
        # Write out black hole data
        if haloID in results_bh:

            (nprog, prog_haloids, prog_npart, prog_mass_contribution,
             ndesc, desc_haloids, desc_npart, desc_mass_contribution, current_halo_pids) = results_bh[haloID]

            sim_prog_haloids = np.zeros(len(prog_haloids), dtype=float)
            for ind, prog in enumerate(prog_haloids):
                sim_prog_haloids[ind] = internal_to_sim_haloID_prog_bh[prog]

            sim_desc_haloids = np.zeros(len(desc_haloids), dtype=float)
            for ind, desc in enumerate(desc_haloids):
                sim_desc_haloids[ind] = internal_to_sim_haloID_desc_bh[desc]

            # Assign values to the corresponding index for the dark matter progenitors
            bh_prog_mass_contribution = np.zeros(len(dm_sim_prog_haloids), dtype=float)
            bh_prog_mass = np.zeros(len(dm_sim_prog_haloids), dtype=float)
            for p, cont, mass in zip(sim_prog_haloids, prog_mass_contribution, prog_npart):
                if not p in dm_sim_prog_haloids:
                    continue
                ind = np.where(dm_sim_prog_haloids == p)
                bh_prog_mass_contribution[ind] = cont
                bh_prog_mass[ind] = mass

            # Assign values to the corresponding index for the dark matter descendents
            bh_desc_mass_contribution = np.zeros(len(dm_sim_desc_haloids), dtype=float)
            bh_desc_mass = np.zeros(len(dm_sim_desc_haloids), dtype=float)
            for p, cont, mass in zip(sim_desc_haloids, desc_mass_contribution, desc_npart):
                if not p in dm_sim_desc_haloids:
                    continue
                ind = np.where(dm_sim_desc_haloids == p)
                bh_desc_mass_contribution[ind] = cont
                bh_desc_mass[ind] = mass

            halo.create_dataset('prog_bh_mass_contribution', data=bh_prog_mass_contribution,
                                dtype=float)  # Mass contribution
            halo.create_dataset('desc_bh_mass_contribution', data=bh_desc_mass_contribution,
                                dtype=float)  # Mass contribution
            halo.create_dataset('Prog_bh_mass', data=bh_prog_mass,
                                dtype=float)  # number of particles in each progenitor
            halo.create_dataset('Desc_bh_mass', data=bh_desc_mass,
                                dtype=float)  # number of particles in each descendant
        else:

            halo.create_dataset('prog_bh_mass_contribution', data=np.array([]),
                                dtype=int)  # Mass contribution
            halo.create_dataset('desc_bh_mass_contribution', data=np.array([]),
                                dtype=int)  # Mass contribution
            halo.create_dataset('Prog_bh_mass', data=np.array([]),
                                dtype=int)  # number of particles in each progenitor
            halo.create_dataset('Desc_bh_mass', data=np.array([]),
                                dtype=int)  # number of particles in each descendant


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
for reg in reversed(regions):

    try:
        os.mkdir('/cosma/home/dp004/dc-rope1/FLARES/FLARES-1/MergerGraphs/GEAGLE_' + reg)
    except OSError:
        pass

    for snap, prog_snap, desc_snap in zip(snaps, prog_snaps, desc_snaps):

        reg_snaps.append((reg, prog_snap, snap, desc_snap))

if __name__ == '__main__':

    ind = int(sys.argv[1])
    print(ind)
    print(reg_snaps[ind])

    mainDirectProgDesc(snap=reg_snaps[ind][2], prog_snap=reg_snaps[ind][1], desc_snap=reg_snaps[ind][3],
                       path='/cosma/home/dp004/dc-rope1/FLARES/FLARES-1/G-EAGLE_' + reg_snaps[ind][0] + '/data',
                       savepath='/cosma/home/dp004/dc-rope1/FLARES/FLARES-1/MergerGraphs/GEAGLE_'
                                + reg_snaps[ind][0] + '/')
