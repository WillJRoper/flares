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

    # Remove particles not in a subgroup
    nosub_mask = subgrp_ids != 1073741824
    all_poss = all_poss[nosub_mask, :]
    gal_sml = gal_sml[nosub_mask]
    grp_ids = grp_ids[nosub_mask]
    subgrp_ids = subgrp_ids[nosub_mask]
    a_born = a_born[nosub_mask]
    metallicities = metallicities[nosub_mask]
    masses = masses[nosub_mask]

    # Calculate ages
    ages = calc_ages(z, a_born)

    # Get the position of each of these galaxies
    gal_ages = {}
    gal_mets = {}
    gal_ms = {}
    gal_smls = {}
    all_gal_poss = {}
    means = {}
    print(np.unique(grp_ids))
    print(np.unique(subgrp_ids, return_counts=True))
    for g, sg, cop in zip(gal_gids, gal_ids, gal_cops):
        mask = (grp_ids == g) & (subgrp_ids == sg)
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
    gas_all_poss = E.read_array('PARTDATA', path, snap, 'PartType0/Coordinates', noH=True, physicalUnits=True, numThreads=8)
    ggrp_ids = E.read_array('PARTDATA', path, snap, 'PartType0/GroupNumber', numThreads=8)
    gsubgrp_ids = E.read_array('PARTDATA', path, snap, 'PartType0/SubGroupNumber', numThreads=8)
    gas_metallicities = E.read_array('PARTDATA', path, snap, 'PartType0/SmoothedMetallicity', noH=True, physicalUnits=True, numThreads=8)
    gas_smooth_ls = E.read_array('PARTDATA', path, snap, 'PartType0/SmoothingLength', noH=True, physicalUnits=True, numThreads=8)
    gas_masses = E.read_array('PARTDATA', path, snap, 'PartType0/Mass', noH=True, physicalUnits=True, numThreads=8) * 10**10

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
    print(np.unique(ggrp_ids))
    print(np.unique(gsubgrp_ids, return_counts=True))
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

if 'UVimg_data/stellardata_reg' + reg + '_snap' + snap + '.pck' in files:
    pass
else:
    get_main(path, snap, reg)
# get_main(path, snap, reg)
