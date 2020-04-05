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


def img_main(path, snap, reg, npart_lim=10**3):

    # Get the redshift
    z_str = snap.split('z')[1].split('p')
    z = float(z_str[0] + '.' + z_str[1])

    # Define stellar particle type
    part_type = 4

    # Load all necessary arrays
    subgrp_ids = E.read_array('PARTDATA', path, snap, 'PartType' + str(part_type) + '/SubGroupNumber', numThreads=8)
    all_poss = E.read_array('SNAP', path, snap, 'PartType' + str(part_type) + '/Coordinates', noH=True, numThreads=8)
    part_ids = E.read_array('SNAP', path, snap, 'PartType' + str(part_type) + '/ParticleIDs', numThreads=8)
    gal_sml = E.read_array('SNAP', path, snap, 'PartType' + str(part_type) + '/SmoothingLength', noH=True, numThreads=8)
    group_part_ids = E.read_array('PARTDATA', path, snap, 'PartType' + str(part_type) + '/ParticleIDs', numThreads=8)
    grp_ids = E.read_array('PARTDATA', path, snap, 'PartType' + str(part_type) + '/GroupNumber', numThreads=8)
    gal_ids = E.read_array('SUBFIND', path, snap, 'Subhalo/SubGroupNumber', numThreads=8)
    gal_gids = E.read_array('SUBFIND', path, snap, 'Subhalo/GroupNumber', numThreads=8)
    gal_cops = E.read_array('SUBFIND', path, snap, 'Subhalo/CentreOfPotential', noH=True, numThreads=8)
    halo_ids = np.zeros_like(grp_ids, dtype=float)
    for (ind, g), sg in zip(enumerate(grp_ids), subgrp_ids):
        halo_ids[ind] = float(str(g) + '.' + str(sg + 1))

    # Get centre of potentials
    gal_cop = {}
    for cop, g, sg in zip(gal_cops, gal_gids, gal_ids):
        gal_cop[float(str(g) + '.' + str(sg + 1))] = cop

    # Translate ID into indices
    ind_to_pid = {}
    pid_to_ind = {}
    for ind, pid in enumerate(part_ids):
        if pid in group_part_ids:
            ind_to_pid[ind] = pid
            pid_to_ind[pid] = ind

    # Get the IDs above the npart threshold
    ids, counts = np.unique(halo_ids, return_counts=True)
    ids = set(ids[counts > npart_lim])

    # Get the particles in the halos
    halo_id_part_inds = {}
    for pid, simid in zip(group_part_ids, halo_ids):
        if simid not in ids:
            continue
        if int(str(simid).split('.')[1]) == 2**30:
            continue
        try:
            halo_id_part_inds.setdefault(simid, set()).update({pid_to_ind[pid]})
        except KeyError:
            ind_to_pid[len(part_ids) + 1] = pid
            pid_to_ind[pid] = len(part_ids) + 1
            halo_id_part_inds.setdefault(simid, set()).update({pid_to_ind[pid]})

    del group_part_ids, halo_ids, pid_to_ind, ind_to_pid, subgrp_ids, part_ids

    gc.collect()

    print('There are', len(ids), 'galaxies above the cutoff')

    # If there are no galaxies exit
    if len(ids) == 0:
        return

    # Load data for luminosities
    a_born = E.read_array('SNAP', path, snap, 'PartType4/StellarFormationTime', noH=True, numThreads=8)
    metallicities = E.read_array('SNAP', path, snap, 'PartType4/SmoothedMetallicity', noH=True, numThreads=8)
    masses = E.read_array('SNAP', path, snap, 'PartType4/Mass', noH=True, numThreads=8) * 10**10

    # Calculate ages
    ages = calc_ages(z, a_born)

    # Get the position of each of these galaxies
    gal_ages = {}
    gal_mets = {}
    gal_ms = {}
    gal_smls = {}
    all_gal_poss = {}
    means = {}
    for id in ids:
        parts = list(halo_id_part_inds[id])
        all_gal_poss[id] = all_poss[parts, :]
        gal_ages[id] = ages[parts]
        gal_mets[id] = metallicities[parts]
        gal_ms[id] = masses[parts]
        gal_smls[id] = gal_sml[parts]
        means[id] = gal_cop[id]

    print('Got galaxy properties')

    del ages, all_poss, metallicities, masses, gal_sml, halo_id_part_inds, a_born

    gc.collect()

    # Get gas particle information
    gsubgrp_ids = E.read_array('PARTDATA', path, snap, 'PartType0/SubGroupNumber', numThreads=8)
    gas_all_poss = E.read_array('SNAP', path, snap, 'PartType0/Coordinates', noH=True, numThreads=8)
    gpart_ids = E.read_array('SNAP', path, snap, 'PartType0/ParticleIDs', numThreads=8)
    ggroup_part_ids = E.read_array('PARTDATA', path, snap, 'PartType0/ParticleIDs', numThreads=8)
    ggrp_ids = E.read_array('PARTDATA', path, snap, 'PartType0/GroupNumber', numThreads=8)
    gas_metallicities = E.read_array('SNAP', path, snap, 'PartType0/SmoothedMetallicity', noH=True, numThreads=8)
    gas_smooth_ls = E.read_array('SNAP', path, snap, 'PartType0/SmoothingLength', noH=True, numThreads=8)
    gas_masses = E.read_array('SNAP', path, snap, 'PartType0/Mass', noH=True, numThreads=8) * 10**10
    ghalo_ids = np.zeros_like(ggrp_ids, dtype=float)
    for (ind, g), sg in zip(enumerate(ggrp_ids), gsubgrp_ids):
        ghalo_ids[ind] = float(str(g) + '.' + str(sg + 1))

    print('Got halo IDs')
    set_ggroup_part_ids = set(ggroup_part_ids)
    # Translate ID into indices
    gind_to_pid = {}
    gpid_to_ind = {}
    for ind, pid in enumerate(gpart_ids):
        if pid in set_ggroup_part_ids:
            gind_to_pid[ind] = pid
            gpid_to_ind[pid] = ind

    print('Made ID dicts')

    # Get the particles in the halos
    ghalo_id_part_inds = {}
    for pid, simid in zip(ggroup_part_ids, ghalo_ids):
        if simid not in ids:
            continue
        if int(str(simid).split('.')[1]) == 2**30:
            continue
        try:
            ghalo_id_part_inds.setdefault(simid, set()).update({gpid_to_ind[pid]})
        except KeyError:
            gind_to_pid[len(gpart_ids) + 1] = pid
            gpid_to_ind[pid] = len(gpart_ids) + 1
            ghalo_id_part_inds.setdefault(simid, set()).update({gpid_to_ind[pid]})

    print('Got particle IDs')

    del ggroup_part_ids, ghalo_ids, gpid_to_ind, gind_to_pid, gsubgrp_ids, gpart_ids

    gc.collect()

    # Get the position of each of these galaxies
    gas_mets = {}
    gas_ms = {}
    gas_smls = {}
    all_gas_poss = {}
    for id in ids:
        gparts = list(ghalo_id_part_inds[id])
        all_gas_poss[id] = gas_all_poss[gparts, :]
        gas_mets[id] = gas_metallicities[gparts]
        gas_ms[id] = gas_masses[gparts]
        gas_smls[id] = gas_smooth_ls[gparts]

    print('Got gas properties')

    del gas_smooth_ls, gas_masses, gas_metallicities, gas_all_poss, ghalo_id_part_inds

    gc.collect()

    save_dict = {'gal_ages': gal_ages, 'gal_mets': gal_mets, 'gal_ms': gal_ms, 'gas_mets': gas_mets,
                 'gas_ms': gas_ms, 'gal_smls': gal_smls, 'gas_smls': gas_smls, 'all_gas_poss': all_gas_poss,
                 'all_gal_poss': all_gal_poss, 'means': means}

    with open('UVimg_data/stellardata_reg' + reg + '_snap' + snap + '_npartgreaterthan'
              + str(npart_lim) + '.pck', 'wb') as pfile1:
        pickle.dump(save_dict, pfile1)


regions = []
for reg in range(0, 40):
    if reg < 10:
        regions.append('0' + str(reg))
    else:
        regions.append(str(reg))

snaps = ['000_z015p000', '001_z014p000', '002_z013p000', '003_z012p000', '004_z011p000', '005_z010p000',
         '006_z009p000', '007_z008p000', '008_z007p000', '009_z006p000', '010_z005p000', '011_z004p770']

reg_snaps = []
for reg in regions:

    try:
        os.mkdir('/cosma/home/dp004/dc-rope1/FLARES/FLARES-1/MergerGraphs/GEAGLE_' + reg)
    except OSError:
        pass

    for snap in snaps:

        reg_snaps.append((reg, snap))

if __name__ == '__main__':

    ind = int(sys.argv[1])
    print(reg_snaps[ind])
    reg, snap = reg_snaps[ind]

    path = '/cosma/home/dp004/dc-rope1/FLARES/FLARES-1/G-EAGLE_' + reg + '/data'

    try:
        img_main(path, snap, reg, npart_lim=10**2)
    except KeyError:
        print('KeyError')
    except OSError:
        print('OSError')