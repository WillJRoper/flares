#!/cosma/home/dp004/dc-rope1/.conda/envs/flares-env/bin/python
import numpy as np
import matplotlib
import eagle_IO.eagle_IO as E
import numba as nb
import astropy.units as u
import gc
import os
import sys
from utilities import calc_ages, get_Z_LOS
import h5py
from astropy.cosmology import Planck13 as cosmo
os.environ['FLARE'] = '/cosma7/data/dp004/dc-wilk2/flare'
import FLARE.filters
from SynthObs.SED import models
matplotlib.use('Agg')


# Define SED model
model = models.define_model('BPASSv2.2.1.binary/ModSalpeter_300',
                            path_to_SPS_grid = FLARE.FLARE_dir + '/data/SPS/nebular/3.0/') # DEFINE SED GRID -
model.dust_ISM = ('simple', {'slope': -1.0})
model.dust_BC = ('simple', {'slope': -1.0})
filters = FLARE.filters.NIRCam
F = FLARE.filters.add_filters(filters, new_lam = model.lam)


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


def get_lumins(gal_poss, gal_ms, gal_ages, gal_mets, gas_mets, gas_poss, gas_ms, gas_sml,
               lkernel, kbins, conv, model, F, i, j, f):

    # Define dimensions array
    if i == 0 and j == 1:
        k = 2
    elif i == 0 and j == 2:
        k = 1
    else:
        k = 0
    dimens = np.array([i, j, k])

    gal_met_surfden = get_Z_LOS(gal_poss, gas_poss, gas_ms, gas_mets, gas_sml, dimens, lkernel, kbins, conv)

    # Calculate optical depth of ISM and birth cloud
    tauVs_ISM = (10 ** 5.2) * gal_met_surfden
    tauVs_BC = 2.0 * (gal_mets / 0.01)

    # Extract the flux in nanoJansky
    L = (models.generate_Fnu_array(model, gal_ms, gal_ages, gal_mets, tauVs_ISM,
                                   tauVs_BC, F, 'JWST.NIRCAM.' + f) * u.nJy).decompose()

    return L


@nb.njit(nogil=True, parallel=True)
def calc_light_mass_rad(poss, ls):

    # Get galaxy particle indices
    rs = np.sqrt(poss[:, 0]**2 + poss[:, 1]**2 + poss[:, 2]**2)

    # Sort the radii and masses
    sinds = np.argsort(rs)
    rs = rs[sinds]
    ls = ls[sinds]
    ls = ls[rs < 30/1e3]
    rs = rs[rs < 30/1e3]

    if len(ls) < 20:
        return -9999, -9999

    # Get the cumalative sum of masses
    l_profile = np.cumsum(ls)

    # Get the total mass and half the total mass
    tot_l = l_profile[-1]
    half_l = tot_l / 2

    # Get the half mass radius particle
    hmr_ind = np.argmin(np.abs(l_profile - half_l))
    hmr = rs[hmr_ind]

    return hmr


def get_main(path, snap, savepath):

    # Get the redshift
    z_str = snap.split('z')[1].split('p')
    z = float(z_str[0] + '.' + z_str[1])

    model.create_Fnu_grid(F, z, cosmo)

    kinp = np.load('/cosma/home/dp004/dc-rope1/cosma7/FLARES/flares/los_extinction/kernel_sph-anarchy.npz',
                   allow_pickle=True)
    lkernel = kinp['kernel']
    header = kinp['header']
    kbins = header.item()['bins']

    conv = (u.solMass / u.Mpc ** 2).to(u.g / u.cm ** 2)

    # Load all necessary arrays
    all_poss = E.read_array('PARTDATA', path, snap, 'PartType4/Coordinates', noH=True,
                            physicalUnits=True, numThreads=8)
    gal_sml = E.read_array('PARTDATA', path, snap, 'PartType4/SmoothingLength', noH=True,
                           physicalUnits=True, numThreads=8)
    subgrp_ids = E.read_array('PARTDATA', path, snap, 'PartType4/SubGroupNumber', numThreads=8)
    subfind_grp_ids = E.read_array('PARTDATA', path, snap, 'PartType4/GroupNumber', numThreads=8)
    subfind_subgrp_ids = E.read_array('PARTDATA', path, snap, 'PartType4/SubGroupNumber', numThreads=8)
    gal_cops = E.read_array('SUBFIND', path, snap, 'Subhalo/CentreOfPotential', noH=True,
                            physicalUnits=True, numThreads=8)
    pre_gal_ms = E.read_array('SUBFIND', path, snap, 'Subhalo/ApertureMeasurements/Mass/030kpc', noH=True,
                            physicalUnits=True, numThreads=8)[:, 4] * 10**10

    # Load data for luminosities
    a_born = E.read_array('PARTDATA', path, snap, 'PartType4/StellarFormationTime', noH=True,
                          physicalUnits=True, numThreads=8)
    metallicities = E.read_array('PARTDATA', path, snap, 'PartType4/SmoothedMetallicity', noH=True,
                                 physicalUnits=True, numThreads=8)
    masses = E.read_array('PARTDATA', path, snap, 'PartType4/Mass', noH=True, physicalUnits=True,
                          numThreads=8) * 10**10

    # Remove particles not in a subgroup
    okinds = np.logical_and(subfind_subgrp_ids != 1073741824, pre_gal_ms > 1e8)
    subfind_grp_ids = subfind_grp_ids[okinds]
    subfind_subgrp_ids = subfind_subgrp_ids[okinds]
    gal_cops = gal_cops[okinds]
    nosub_mask = subgrp_ids != 1073741824
    all_poss = all_poss[nosub_mask, :]
    gal_sml = gal_sml[nosub_mask]
    a_born = a_born[nosub_mask]
    metallicities = metallicities[nosub_mask]
    masses = masses[nosub_mask]

    # Convert IDs to float(groupNumber.SubGroupNumber) format, i.e. group 1 subgroup 11 = 1.00011
    halo_ids = np.zeros(subfind_grp_ids.size, dtype=float)
    for (ind, g), sg in zip(enumerate(subfind_grp_ids), subfind_subgrp_ids):
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
    gas_metallicities = gas_metallicities[nosub_mask]
    gas_smooth_ls = gas_smooth_ls[nosub_mask]
    gas_masses = gas_masses[nosub_mask]

    # Get particle indices
    ghalo_part_inds = get_subgroup_part_inds(path, snap, part_type=0, all_parts=False, sorted=False)

    # Get the position of each of these galaxies
    gas_mets = {}
    gas_ms = {}
    gas_smls = {}
    all_gas_poss = {}
    for id in halo_ids:
        try:
            mask = ghalo_part_inds[id]
        except KeyError:
            print("Galaxy", id, "has no gas")
            continue
        all_gas_poss[id] = gas_all_poss[mask, :]
        gas_mets[id] = gas_metallicities[mask]
        gas_ms[id] = gas_masses[mask]
        gas_smls[id] = gas_smooth_ls[mask]

    print('Got particle IDs')

    del gsubgrp_ids, gas_all_poss, gas_metallicities, gas_smooth_ls, gas_masses

    gc.collect()

    print('Got gas properties')

    # Open the HDF5 file
    hdf = h5py.File(savepath + 'ObsWebbLumins_' + snap + '.hdf5', 'w')
    hdf.create_dataset('orientation', data=[(0, 1), (1, 2), (0, 2)])  # Mass contribution
    hdf.create_dataset('galaxy_ids', data=halo_ids)  # galaxy ids

    # Loop over filters
    for f in FLARE.filters.NIRCam:

        print("Extracting data for filter:", f)

        # Create images for these galaxies
        hls = np.zeros((len(gal_ages), 3))
        ms = np.zeros((len(gal_ages), 3))
        tot_l = np.zeros((len(gal_ages), 3))
        for ind1, (i, j) in enumerate([(0, 1), (1, 2), (0, 2)]):
            for ind2, id in enumerate(halo_ids):

                # Get the luminosities
                try:
                    ls = get_lumins(all_gal_poss[id] - means[id], gal_ms[id], gal_ages[id], gal_mets[id], gas_mets[id],
                                    all_gas_poss[id] - means[id], gas_ms[id], gas_smls[id], lkernel, kbins, conv, model,
                                    F, i, j, f)

                    # Compute half mass radii
                    hls[ind2, ind1] = calc_light_mass_rad(all_gal_poss[id] - means[id], ls)
                    ms[ind, ind1] = np.sum(gal_ms[id]) / 10**10
                    tot_l[ind, ind1] = np.sum(ls)

                except KeyError:
                    print("Galaxy", id, "did not appear in one of the dictionaries")
                    hls[ind2, ind1] = -9999
                    ms[ind, ind1] = -9999
                    tot_l[ind, ind1] = -9999
                    continue

        # Write out the results for this filter
        filt = hdf.create_group(f)  # create halo group
        filt.create_dataset('half_lift_rad', data=hls, dtype=int)  # Half light radius [Mpc]
        filt.create_dataset('Aperture_Mass_30kpc', data=ms, dtype=int)  # Aperture mass [Msun * 10*10]
        filt.create_dataset('Aperture_Luminosity_30kpc', data=tot_l, dtype=int)  # Aperture Luminosity [nJy]

    hdf.close()




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

    for snap in snaps:

        reg_snaps.append((reg, snap))

ind = int(sys.argv[1])
print(reg_snaps[ind])
reg, snap = reg_snaps[ind]

path = '/cosma/home/dp004/dc-rope1/FLARES/FLARES-1/G-EAGLE_' + reg + '/data'
savepath = '/cosma/home/dp004/dc-rope1/FLARES/FLARES-1/WebbData/GEAGLE_' + reg_snaps[ind][0] + '/'

get_main(path, snap, savepath)
