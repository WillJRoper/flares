#!/cosma/home/dp004/dc-rope1/.conda/envs/flares-env/bin/python
import numpy as np
import matplotlib
import h5py
import eagle_IO.eagle_IO as E
import os
import gc
import sys

matplotlib.use('Agg')


def get_current_part_IDs(path, snap, part_type, part_ids):

    # Extract the halo IDs (group names/keys) contained within this snapshot
    try:
        group_part_ids = E.read_array('PARTDATA', path, snap, 'PartType' + str(part_type) + '/ParticleIDs',
                                      numThreads=8)
        grp_ids = E.read_array('PARTDATA', path, snap, 'PartType' + str(part_type) + '/GroupNumber', numThreads=8)
        subgrp_ids = E.read_array('PARTDATA', path, snap, 'PartType' + str(part_type) + '/SubGroupNumber', numThreads=8)
    except ValueError:
        print("There were no particle IDs in", path, snap)
        group_part_ids = np.array([])
        grp_ids = np.array([])
        subgrp_ids = np.array([])

    # DEBUG DEBUG DEBUG DEBUG DEBUG DEBUG DEBUG DEBUG DEBUG DEBUG DEBUG DEBUG DEBUG DEBUG DEBUG DEBUG DEBUG DEBUG
    okinds = group_part_ids < 1407374870714
    group_part_ids = group_part_ids[okinds]
    grp_ids = grp_ids[okinds]
    subgrp_ids = subgrp_ids[okinds]
    # DEBUG DEBUG DEBUG DEBUG DEBUG DEBUG DEBUG DEBUG DEBUG DEBUG DEBUG DEBUG DEBUG DEBUG DEBUG DEBUG DEBUG DEBUG

    # Remove particles not associated to a subgroup
    okinds = subgrp_ids != 1073741824
    group_part_ids = group_part_ids[okinds]
    grp_ids = grp_ids[okinds]
    subgrp_ids = subgrp_ids[okinds]
    halo_ids = np.zeros(grp_ids.size, dtype=float)
    for (ind, g), sg in zip(enumerate(grp_ids), subgrp_ids):
        halo_ids[ind] = float(str(int(g)) + '.%05d' % int(sg))

    # Sort particle IDS
    part_ids = np.sort(part_ids)
    unsort_part_ids = np.copy(part_ids)
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

    return halo_id_part_inds, sinds, unsort_part_ids


def get_part_halo_data(path, snap, prog_snap, desc_snap, part_type):

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
        try:
            snap_part_ids = E.read_array('SNAP', path, snap, 'PartType' + str(part_type) + '/ParticleIDs',
                                         numThreads=8)
        except ValueError:
            snap_part_ids = np.array([])

        if prog_snap != None:
            try:
                progsnap_part_ids = E.read_array('SNAP', path, prog_snap, 'PartType' + str(part_type) + '/ParticleIDs',
                                                 numThreads=8)
            except ValueError:
                progsnap_part_ids = np.array([])

            # Get halo IDs and halo data
            prog_subgrp_ids = E.read_array('SUBFIND', path, prog_snap, 'Subhalo/SubGroupNumber', numThreads=8)
            prog_grp_ids = E.read_array('SUBFIND', path, prog_snap, 'Subhalo/GroupNumber', numThreads=8)
            preprog_gal_ms = E.read_array('SUBFIND', path, prog_snap, 'Subhalo/ApertureMeasurements/Mass/030kpc',
                                          noH=False,
                                          physicalUnits=False, numThreads=8)[:, part_type]
            try:
                preprogpart_masses = E.read_array('PARTDATA', path, prog_snap, 'PartType' + str(part_type) + '/Mass',
                                                  noH=False,
                                                  physicalUnits=False, numThreads=8)
            except ValueError:
                preprogpart_masses = np.array([])

            preprog_sub_ids = np.zeros(prog_grp_ids.size, dtype=float)
            for (ind, g), sg in zip(enumerate(prog_grp_ids), prog_subgrp_ids):
                preprog_sub_ids[ind] = float(str(int(g)) + '.%05d' % int(sg))

        else:
            progsnap_part_ids = np.array([])
            preprog_sub_ids = np.array([])
            preprogpart_masses = np.array([])
            preprog_gal_ms = np.array([])

        if desc_snap != None:
            try:
                descsnap_part_ids = E.read_array('SNAP', path, desc_snap, 'PartType' + str(part_type) + '/ParticleIDs',
                                                 numThreads=8)
            except ValueError:
                descsnap_part_ids = np.array([])

            # Get halo IDs and halo data
            desc_subgrp_ids = E.read_array('SUBFIND', path, desc_snap, 'Subhalo/SubGroupNumber', numThreads=8)
            desc_grp_ids = E.read_array('SUBFIND', path, desc_snap, 'Subhalo/GroupNumber', numThreads=8)
            predesc_gal_ms = E.read_array('SUBFIND', path, desc_snap, 'Subhalo/ApertureMeasurements/Mass/030kpc',
                                          noH=False,
                                          physicalUnits=False, numThreads=8)[:, part_type]
            try:
                predescpart_masses = E.read_array('PARTDATA', path, desc_snap, 'PartType' + str(part_type) + '/Mass',
                                                  noH=False,
                                                  physicalUnits=False, numThreads=8)
            except ValueError:
                predescpart_masses = np.array([])

            predesc_sub_ids = np.zeros(desc_grp_ids.size, dtype=float)
            for (ind, g), sg in zip(enumerate(desc_grp_ids), desc_subgrp_ids):
                predesc_sub_ids[ind] = float(str(int(g)) + '.%05d' % int(sg))
        else:
            descsnap_part_ids = np.array([])
            predesc_sub_ids = np.array([])
            predescpart_masses = np.array([])
            predesc_gal_ms = np.array([])

        # Combine the particle id arrays and only take unique values
        part_ids = np.unique(np.concatenate([snap_part_ids, progsnap_part_ids, descsnap_part_ids]))

    return (part_ids, preprogpart_masses, predescpart_masses, preprog_gal_ms, preprog_sub_ids,
            predesc_gal_ms, predesc_sub_ids)


def get_direct_IDs(path, snap, sinds, unsort_part_ids, part_ids, part_type, predirectpart_masses, 
                   predirect_sub_ids, predirect_gal_ms):

    try:
        grp_ids = E.read_array('PARTDATA', path, snap, 'PartType' + str(part_type) + '/GroupNumber',
                               numThreads=8)
        subgrp_ids = E.read_array('PARTDATA', path, snap, 'PartType' + str(part_type) + '/SubGroupNumber',
                                  numThreads=8)
        direct_part_ids = E.read_array('PARTDATA', path, snap, 'PartType' + str(part_type) + '/ParticleIDs',
                                       numThreads=8)
    except ValueError:
        grp_ids = np.array([])
        subgrp_ids = np.array([])
        direct_part_ids = np.array([])

    # Remove particles not associated to a subgroup
    okinds = subgrp_ids != 1073741824
    direct_part_ids = direct_part_ids[okinds]
    grp_ids = grp_ids[okinds]
    subgrp_ids = subgrp_ids[okinds]

    if part_type == 1:
        print("in direct read ins", snap)
        print(direct_part_ids.shape)
        print(grp_ids.shape)
        print(subgrp_ids.shape)

    if part_type != 1:
        predirectpart_masses = predirectpart_masses[okinds]

    direct_halo_ids = np.zeros(grp_ids.size, dtype=float)
    for (ind, g), sg in zip(enumerate(grp_ids), subgrp_ids):
        direct_halo_ids[ind] = float(str(int(g)) + '.%05d' % int(sg))

    if part_type == 1:
        print("in direct pre particle matching")
        print(direct_halo_ids.shape)
        print(sinds.shape)
        print(part_ids.shape)
        print(unsort_part_ids.shape)
        print(part_ids)
        print(direct_part_ids)

    sorted_index = np.searchsorted(part_ids, direct_part_ids)
    yindex = np.take(sinds, sorted_index, mode="clip")
    mask = unsort_part_ids[yindex] != direct_part_ids
    result = np.ma.array(yindex, mask=mask)

    part_groups = direct_halo_ids[np.logical_not(result.mask)]
    parts_in_groups = result.data[np.logical_not(result.mask)]

    # Map halo IDs to continuous intergers
    internal_to_sim_haloID_direct = {}
    sim_to_internal_haloID_direct = {}
    for i, p in enumerate(np.unique(part_groups)):
        internal_to_sim_haloID_direct[i] = p
        sim_to_internal_haloID_direct[p] = i

    if part_type == 1:
        print("post matching")
        print(part_groups.shape)
        print(parts_in_groups.shape)
        print(len(sim_to_internal_haloID_direct))

    if part_type != 1:
        direct_snap_haloIDs = np.full(len(part_ids), -2, dtype=int)
        directpart_masses = np.full(len(part_ids), -2, dtype=float)
        for ind, direct, m in zip(parts_in_groups, part_groups, predirectpart_masses):
            direct_snap_haloIDs[ind] = sim_to_internal_haloID_direct[direct]
            directpart_masses[ind] = m

        direct_sub_ids = []
        direct_gal_ms = []
        for direct, m in zip(predirect_sub_ids, predirect_gal_ms):
            try:
                direct_sub_ids.append(sim_to_internal_haloID_direct[direct])
                direct_gal_ms.append(m)
            except KeyError:
                continue

        direct_sub_ids = np.array(direct_sub_ids)
        direct_gal_ms = np.array(direct_gal_ms)

    else:
        print("part type is 1")
        direct_gal_ms = np.array([], copy=False)
        direct_sub_ids = np.array([], copy=False)
        directpart_masses = np.array([], copy=False)
        direct_snap_haloIDs = np.full(len(part_ids), -2, dtype=int)
        for ind, direct in zip(parts_in_groups, part_groups):
            print(ind, direct)
            direct_snap_haloIDs[ind] = sim_to_internal_haloID_direct[direct]

    # Get all the unique halo IDs in this snapshot and the number of times they appear
    direct_unique, direct_counts = np.unique(direct_snap_haloIDs, return_counts=True)

    # Remove single particle halos (ID=-2), since np.unique returns a sorted array this can be
    # done by removing the first value
    direct_unique = direct_unique[1:]
    direct_counts = direct_counts[1:]
    
    return (direct_snap_haloIDs, direct_counts, directpart_masses, direct_sub_ids, 
            direct_gal_ms, internal_to_sim_haloID_direct)


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

    # Extract raw particle and halo data for each snapshot
    data = get_part_halo_data(path, snap, prog_snap, desc_snap, part_type)
    part_ids = data[0]
    preprogpart_masses = data[1]
    predescpart_masses = data[2]
    preprog_gal_ms = data[3]
    preprog_sub_ids = data[4]
    predesc_gal_ms = data[5]
    predesc_sub_ids = data[6]
    if part_type == 1:
        print(part_ids.shape)
    # If no part IDs exist exit
    if len(part_ids) == 0:
        return {}, {}, {}

    # =============== Current Snapshot ===============

    # Get particle IDs for each halo in the current snapshot
    halo_id_part_inds, sinds, unsort_part_ids = get_current_part_IDs(path, snap, part_type, part_ids)
    if part_type == 1:
        print(unsort_part_ids.shape)
        print(part_ids.shape)
        print(len(halo_id_part_inds))
    # =============== Progenitor Snapshot ===============

    # Only look for descendant data if there is a descendant snapshot
    if prog_snap != None:

        id_data = get_direct_IDs(path, prog_snap, sinds, unsort_part_ids, part_ids, part_type, preprogpart_masses,
                                 preprog_sub_ids, preprog_gal_ms)
        prog_snap_haloIDs = id_data[0]
        prog_counts = id_data[1]
        progpart_masses = id_data[2]
        prog_sub_ids = id_data[3]
        prog_gal_ms = id_data[4]
        internal_to_sim_haloID_prog = id_data[5]

    else:  
        # Assign an empty array if the snapshot is less than the earliest (000)
        prog_snap_haloIDs = np.array([], copy=False)
        prog_counts = np.array([], copy=False)
        progpart_masses = np.array([], copy=False)
        prog_sub_ids = np.array([], copy=False)
        prog_gal_ms = np.array([], copy=False)
        internal_to_sim_haloID_prog = {}
    print(np.unique(prog_snap_haloIDs))
    print(prog_counts)
    # =============== Descendant Snapshot ===============

    # Only look for descendant data if there is a descendant snapshot
    if desc_snap != None:

        id_data = get_direct_IDs(path, desc_snap, sinds, unsort_part_ids, part_ids, part_type, predescpart_masses,
                                 predesc_sub_ids, predesc_gal_ms)
        desc_snap_haloIDs = id_data[0]
        desc_counts = id_data[1]
        descpart_masses = id_data[2]
        desc_sub_ids = id_data[3]
        desc_gal_ms = id_data[4]
        internal_to_sim_haloID_desc = id_data[5]

    else:  
        # Assign an empty array if the snapshot is less than the earliest (000)
        desc_snap_haloIDs = np.array([], copy=False)
        desc_counts = np.array([], copy=False)
        descpart_masses = np.array([], copy=False)
        desc_sub_ids = np.array([], copy=False)
        desc_gal_ms = np.array([], copy=False)
        internal_to_sim_haloID_desc = {}
    print(np.unique(desc_snap_haloIDs))
    # Clean up arrays that are no longer needed
    del sinds, unsort_part_ids, part_ids
    gc.collect()

    # =============== Find all Direct Progenitors And Descendant Of Halos In This Snapshot ===============
    
    # Initialise dictionary for results
    results = {}

    # Loop through all the halos in this snapshot
    for num, haloID in enumerate(halo_id_part_inds.keys()):

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

    # Get the direct progenitors and descendents for the dark matter halos
    results_dm, internal_to_sim_haloID_desc_dm, internal_to_sim_haloID_prog_dm = partDirectProgDesc(snap, prog_snap,
                                                                                                    desc_snap, path,
                                                                                                    part_type=1)

    # Open HDF5 file
    hdf = h5py.File(savepath + 'SubMgraph_' + snap + '.hdf5', 'w')

    # Initialise list to store haloIDs and dictionaries to store progenitors and descendents
    halo_ids_lst = []
    dmprogs = {}
    dmdescs = {}

    # Loop over dark matter results writing them out
    for num, haloID in enumerate(results_dm):

        (nprog, prog_haloids, prog_npart, prog_mass_contribution,
         ndesc, desc_haloids, desc_npart, desc_mass_contribution, current_halo_pids) = results_dm[haloID]
        # print(nprog, ndesc)
        if nprog < 1 and ndesc < 1 or len(current_halo_pids) < 20:
            continue

        sim_prog_haloids = np.zeros(len(prog_haloids), dtype=float)
        for ind, prog in enumerate(prog_haloids):
            sim_prog_haloids[ind] = internal_to_sim_haloID_prog_dm[prog]

        sim_desc_haloids = np.zeros(len(desc_haloids), dtype=float)
        for ind, desc in enumerate(desc_haloids):
            sim_desc_haloids[ind] = internal_to_sim_haloID_desc_dm[desc]

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

        # Append this ID to the halo ID list
        halo_ids_lst.append(haloID)

        # Store the progenitors and descendents
        dmprogs[haloID] = sim_prog_haloids
        dmdescs[haloID] = sim_desc_haloids

    # Clear memory of unsused dictionaries
    del results_dm, internal_to_sim_haloID_desc_dm, internal_to_sim_haloID_prog_dm
    gc.collect()

    for ptype in [4, 5, 0]:

        results, internal_to_sim_haloID_desc, internal_to_sim_haloID_prog = partDirectProgDesc(snap, prog_snap,
                                                                                               desc_snap, path,
                                                                                               part_type=ptype)

        for num, haloID in enumerate(halo_ids_lst):

            # Write out star data
            if haloID in results:

                # Αssign dark matter haloids for comparison to other particle types
                dm_sim_desc_haloids = dmdescs[haloID]
                dm_sim_prog_haloids = dmprogs[haloID]

                (nprog, prog_haloids, prog_npart, prog_mass_contribution,
                 ndesc, desc_haloids, desc_npart, desc_mass_contribution, current_halo_pids) = results[haloID]

                sim_prog_haloids = np.zeros(len(prog_haloids), dtype=float)
                for ind, prog in enumerate(prog_haloids):
                    sim_prog_haloids[ind] = internal_to_sim_haloID_prog[prog]

                sim_desc_haloids = np.zeros(len(desc_haloids), dtype=float)
                for ind, desc in enumerate(desc_haloids):
                    sim_desc_haloids[ind] = internal_to_sim_haloID_desc[desc]

                # Assign values to the corresponding index for the dark matter progenitors
                out_prog_mass_contribution = np.zeros(len(dm_sim_prog_haloids), dtype=float)
                out_prog_mass = np.zeros(len(dm_sim_prog_haloids), dtype=float)
                for p, cont, mass in zip(sim_prog_haloids, prog_mass_contribution, prog_npart):
                    if not p in dm_sim_prog_haloids:
                        continue
                    ind = np.where(dm_sim_prog_haloids == p)
                    out_prog_mass_contribution[ind] = cont
                    out_prog_mass[ind] = mass

                # Assign values to the corresponding index for the dark matter descendents
                out_desc_mass_contribution = np.zeros(len(dm_sim_desc_haloids), dtype=float)
                out_desc_mass = np.zeros(len(dm_sim_desc_haloids), dtype=float)
                for d, cont, mass in zip(sim_desc_haloids, desc_mass_contribution, desc_npart):
                    if not d in dm_sim_desc_haloids:
                        continue
                    ind = np.where(dm_sim_desc_haloids == d)
                    out_desc_mass_contribution[ind] = cont
                    out_desc_mass[ind] = mass

                halo = hdf[str(haloID)]

                if ptype == 4:

                    halo.create_dataset('prog_stellar_mass_contribution', data=out_prog_mass_contribution,
                                        dtype=float)  # Mass contribution
                    halo.create_dataset('desc_stellar_mass_contribution', data=out_desc_mass_contribution,
                                        dtype=float)  # Mass contribution
                    halo.create_dataset('Prog_stellar_mass', data=out_prog_mass,
                                        dtype=float)  # number of particles in each progenitor
                    halo.create_dataset('Desc_stellar_mass', data=out_desc_mass,
                                        dtype=float)  # number of particles in each descendant

                if ptype == 0:

                    halo.create_dataset('prog_gas_mass_contribution', data=out_prog_mass_contribution,
                                        dtype=float)  # Mass contribution
                    halo.create_dataset('desc_gas_mass_contribution', data=out_desc_mass_contribution,
                                        dtype=float)  # Mass contribution
                    halo.create_dataset('Prog_gas_mass', data=out_prog_mass,
                                        dtype=float)  # number of particles in each progenitor
                    halo.create_dataset('Desc_gas_mass', data=out_desc_mass,
                                        dtype=float)  # number of particles in each descendant

                if ptype == 5:

                    halo.create_dataset('prog_bh_mass_contribution', data=out_prog_mass_contribution,
                                        dtype=float)  # Mass contribution
                    halo.create_dataset('desc_bh_mass_contribution', data=out_desc_mass_contribution,
                                        dtype=float)  # Mass contribution
                    halo.create_dataset('Prog_bh_mass', data=out_prog_mass,
                                        dtype=float)  # number of particles in each progenitor
                    halo.create_dataset('Desc_bh_mass', data=out_desc_mass,
                                        dtype=float)  # number of particles in each descendant

            else:

                halo = hdf[str(haloID)]

                if ptype == 4:

                    halo.create_dataset('prog_stellar_mass_contribution', data=np.array([]),
                                        dtype=float)  # Mass contribution
                    halo.create_dataset('desc_stellar_mass_contribution', data=np.array([]),
                                        dtype=float)  # Mass contribution
                    halo.create_dataset('Prog_stellar_mass', data=np.array([]),
                                        dtype=float)  # number of particles in each progenitor
                    halo.create_dataset('Desc_stellar_mass', data=np.array([]),
                                        dtype=float)  # number of particles in each descendant

                if ptype == 0:

                    halo.create_dataset('prog_gas_mass_contribution', data=np.array([]),
                                        dtype=int)  # Mass contribution
                    halo.create_dataset('desc_gas_mass_contribution', data=np.array([]),
                                        dtype=int)  # Mass contribution
                    halo.create_dataset('Prog_gas_mass', data=np.array([]),
                                        dtype=int)  # number of particles in each progenitor
                    halo.create_dataset('Desc_gas_mass', data=np.array([]),
                                        dtype=int)  # number of particles in each descendant

                if ptype == 5:

                    halo.create_dataset('prog_bh_mass_contribution', data=np.array([]),
                                        dtype=int)  # Mass contribution
                    halo.create_dataset('desc_bh_mass_contribution', data=np.array([]),
                                        dtype=int)  # Mass contribution
                    halo.create_dataset('Prog_bh_mass', data=np.array([]),
                                        dtype=int)  # number of particles in each progenitor
                    halo.create_dataset('Desc_bh_mass', data=np.array([]),
                                        dtype=int)  # number of particles in each descendant

        del results, internal_to_sim_haloID_desc, internal_to_sim_haloID_prog
        gc.collect()

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
                       path=path,
                       savepath='/cosma/home/dp004/dc-rope1/FLARES/FLARES-1/MergerGraphs/REF/')
