#!/cosma/home/dp004/dc-rope1/.conda/envs/flares-env/bin/python
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import eagle_IO.eagle_IO as E
import numba as nb
import astropy.units as u
import gc
import os
import sys
from utilities import calc_ages, get_Z_LOS
from photutils import CircularAperture, aperture_photometry
from scipy.interpolate import interp1d
import h5py
import warnings
from astropy.cosmology import Planck13 as cosmo
os.environ['FLARE'] = '/cosma7/data/dp004/dc-wilk2/flare'
import FLARE.filters
from SynthObs.SED import models
matplotlib.use('Agg')
warnings.filterwarnings('ignore')


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


def get_lumins(gal_poss, gal_ini_ms, gal_ages, gal_mets, gas_mets, gas_poss, gas_ms, gas_sml,
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
    tauVs_ISM = 0.0063 * gal_met_surfden
    tauVs_BC = 1.25 * (gal_mets / 0.01)

    # Extract the flux in erg s^-1 Hz^-1
    if f.split(".")[0] == 'FAKE':
        L = (models.generate_Lnu_array(model, gal_ini_ms, gal_ages, gal_mets, tauVs_ISM,
                                       tauVs_BC, F, f, fesc=0, log10t_BC=7))
    else:
        L = (models.generate_Fnu_array(model, gal_ini_ms, gal_ages, gal_mets, tauVs_ISM,
                                       tauVs_BC, F, f, fesc=0, log10t_BC=7))

    return L


@nb.njit(nogil=True, parallel=True)
def make_soft_img(pos, Ndim, i, j, imgrange, ls, smooth):

    # Define x and y positions for the gaussians
    Gx, Gy = np.meshgrid(np.linspace(imgrange[0][0], imgrange[0][1], Ndim),
                         np.linspace(imgrange[1][0], imgrange[1][1], Ndim))

    # Initialise the image array
    gsmooth_img = np.zeros((Ndim, Ndim))

    # Loop over each star computing the smoothed gaussian distribution for this particle
    for x, y, l, sml in zip(pos[:, i], pos[:, j], ls, smooth):

        # Compute the image
        g = np.exp(-(((Gx - x) ** 2 + (Gy - y) ** 2) / (2.0 * sml ** 2)))

        # Get the sum of the gaussian
        gsum = np.sum(g)

        # If there are stars within the image in this gaussian add it to the image array
        if gsum > 0:
            gsmooth_img += g * l / gsum

    # img, xedges, yedges = np.histogram2d(pos[:, i], pos[:, j], bins=nbin, range=imgrange, weights=ls)

    return gsmooth_img


@nb.jit(nogil=True, parallel=True)
def get_img_hlr(img, apertures, tot_l, app_rs, res, csoft):

    # Apply the apertures
    phot_table = aperture_photometry(img, apertures, method='subpixel', subpixels=5)

    # Extract the aperture luminosities
    row = np.lib.recfunctions.structured_to_unstructured(np.array(phot_table[0]))
    lumins = row[3:]

    # Get half the total luminosity
    half_l = tot_l / 2

    # Interpolate to increase resolution
    func = interp1d(app_rs, lumins, kind="linear")
    interp_rs = np.linspace(0.001, res / 4, 10000) * csoft
    interp_lumins = func(interp_rs)

    # Get the half mass radius particle
    hmr_ind = np.argmin(np.abs(interp_lumins - half_l))
    hmr = interp_rs[hmr_ind]

    return hmr


@nb.njit(nogil=True, parallel=True)
def calc_rad(poss, i, j):

    # Get galaxy particle indices
    rs = np.sqrt(poss[:, i]**2 + poss[:, j]**2)

    return rs


@nb.njit(nogil=True, parallel=True)
def calc_3drad(poss):

    # Get galaxy particle indices
    rs = np.sqrt(poss[:, 0]**2 + poss[:, 1]**2 + poss[:, 2]**2)

    return rs


def lumin_weighted_centre(poss, ls):

    cent = np.average(poss, axis=0, weights=ls)

    return cent


@nb.njit(nogil=True, parallel=True)
def calc_light_mass_rad(ls, poss, i, j, ms):

    rs = calc_rad(poss, i, j)

    # Sort the radii and masses
    sinds = np.argsort(rs)
    rs = rs[sinds]
    ls = ls[sinds]
    okinds = rs <= 0.03
    rs = rs[okinds]
    ls = ls[okinds]
    ms = ms[okinds]

    # Get the cumalative sum of masses
    l_profile = np.cumsum(ls)

    # Get the total mass and half the total mass
    tot_l = np.sum(ls)
    half_l = tot_l / 2

    # Get the half mass radius particle
    hmr_ind = np.argmin(np.abs(l_profile - half_l))
    hmr = rs[hmr_ind]

    return hmr, tot_l, np.sum(ms) / 10**10


def get_main(path, snap, savepath, filters, F, model, filename):

    # Get the redshift
    z_str = snap.split('z')[1].split('p')
    z = float(z_str[0] + '.' + z_str[1])

    kinp = np.load('/cosma/home/dp004/dc-rope1/cosma7/FLARES/flares/los_extinction/kernel_sph-anarchy.npz',
                   allow_pickle=True)
    lkernel = kinp['kernel']
    header = kinp['header']
    kbins = header.item()['bins']

    conv = (u.solMass / u.Mpc ** 2).to(u.solMass / u.pc ** 2)

    # Load all necessary arrays
    subfind_grp_ids = E.read_array('SUBFIND', path, snap, 'Subhalo/GroupNumber', numThreads=8)
    subfind_subgrp_ids = E.read_array('SUBFIND', path, snap, 'Subhalo/SubGroupNumber', numThreads=8)
    gal_cops = E.read_array('SUBFIND', path, snap, 'Subhalo/CentreOfPotential', noH=True,
                            physicalUnits=True, numThreads=8)
    all_gal_ns = E.read_array('SUBFIND', path, snap, 'Subhalo/SubLengthType', numThreads=8)
    all_gal_ms = E.read_array('SUBFIND', path, snap, 'Subhalo/ApertureMeasurements/Mass/030kpc', numThreads=8) * 10**10

    # Remove particles not in a subgroup
    okinds = np.logical_and(subfind_subgrp_ids != 1073741824,
                            np.logical_and((all_gal_ns[:, 4] + all_gal_ns[:, 0]) >= 100,
                                           np.logical_and(all_gal_ms[:, 4] >= 1e8, all_gal_ms[:, 0] > 0)
                                           )
                            )
    subfind_grp_ids = subfind_grp_ids[okinds]
    subfind_subgrp_ids = subfind_subgrp_ids[okinds]
    gal_cops = gal_cops[okinds]

    # Convert IDs to float(groupNumber.SubGroupNumber) format, i.e. group 1 subgroup 11 = 1.00011
    halo_ids = np.zeros(subfind_grp_ids.size, dtype=float)
    for (ind, g), sg in zip(enumerate(subfind_grp_ids), subfind_subgrp_ids):
        halo_ids[ind] = float(str(int(g)) + '.%05d' % int(sg))

    star_halo_ids = np.copy(halo_ids)

    try:
        # Load data for luminosities
        all_poss = E.read_array('PARTDATA', path, snap, 'PartType4/Coordinates', noH=True,
                                physicalUnits=True, numThreads=8)
        gal_sml = E.read_array('PARTDATA', path, snap, 'PartType4/SmoothingLength', noH=True,
                               physicalUnits=True, numThreads=8)
        a_born = E.read_array('PARTDATA', path, snap, 'PartType4/StellarFormationTime', noH=True,
                              physicalUnits=True, numThreads=8)
        metallicities = E.read_array('PARTDATA', path, snap, 'PartType4/SmoothedMetallicity', noH=True,
                                     physicalUnits=True, numThreads=8)
        masses = E.read_array('PARTDATA', path, snap, 'PartType4/InitialMass', noH=True, physicalUnits=True,
                              numThreads=8) * 10 ** 10
        app_masses = E.read_array('PARTDATA', path, snap, 'PartType4/Mass', noH=True, physicalUnits=True,
                              numThreads=8) * 10 ** 10

        # Calculate ages
        ages = calc_ages(z, a_born)

        grp_ids = E.read_array('PARTDATA', path, snap, 'PartType4/GroupNumber', noH=True,
                               physicalUnits=True, verbose=False, numThreads=8)

        subgrp_ids = E.read_array('PARTDATA', path, snap, 'PartType4/SubGroupNumber', noH=True,
                                  physicalUnits=True, verbose=False, numThreads=8)

        part_ids = E.read_array('PARTDATA', path, snap, 'PartType4/ParticleIDs', noH=True,
                                physicalUnits=True, verbose=False, numThreads=8)
    except OSError:
        return
    except KeyError:
        return

    # A copy of this array is needed for the extraction method
    group_part_ids = np.copy(part_ids)

    print("There are", len(subgrp_ids), "particles")

    # Convert IDs to float(groupNumber.SubGroupNumber) format, i.e. group 1 subgroup 11 = 1.00011
    part_halo_ids = np.zeros(grp_ids.size, dtype=float)
    for (ind, g), sg in zip(enumerate(grp_ids), subgrp_ids):
        part_halo_ids[ind] = float(str(int(g)) + '.%05d' % int(sg))

    okinds = np.isin(part_halo_ids, star_halo_ids)
    part_halo_ids = part_halo_ids[okinds]
    part_ids = part_ids[okinds]
    group_part_ids = group_part_ids[okinds]
    all_poss = all_poss[okinds]
    gal_sml = gal_sml[okinds]
    ages = ages[okinds]
    metallicities = metallicities[okinds]
    masses = masses[okinds]
    app_masses = app_masses[okinds]

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

    # Get the position of each of these galaxies
    gal_ages = {}
    gal_mets = {}
    gal_ini_ms = {}
    gal_app_ms = {}
    gal_smls = {}
    all_gal_poss = {}
    means = {}
    for id, cop in zip(star_halo_ids, gal_cops):
        mask = halo_part_inds[id]
        all_gal_poss[id] = all_poss[mask, :]
        gal_ages[id] = ages[mask]
        gal_mets[id] = metallicities[mask]
        gal_ini_ms[id] = masses[mask]
        gal_app_ms[id] = app_masses[mask]
        gal_smls[id] = gal_sml[mask]
        means[id] = cop

    print('There are', len(gal_ages.keys()), 'galaxies')

    del ages, all_poss, metallicities, masses, gal_sml, a_born, halo_ids

    gc.collect()

    # If there are no galaxies exit
    if len(gal_ages.keys()) == 0:
        return
    
    print('Got galaxy properties')
    
    # Get gas particle information
    gas_all_poss = E.read_array('PARTDATA', path, snap, 'PartType0/Coordinates', noH=True, physicalUnits=True,
                                numThreads=8)
    gas_metallicities = E.read_array('PARTDATA', path, snap, 'PartType0/SmoothedMetallicity', noH=True,
                                     physicalUnits=True, numThreads=8)
    gas_smooth_ls = E.read_array('PARTDATA', path, snap, 'PartType0/SmoothingLength', noH=True, physicalUnits=True,
                                 numThreads=8)
    gas_masses = E.read_array('PARTDATA', path, snap, 'PartType0/Mass', noH=True, physicalUnits=True,
                              numThreads=8) * 10**10

    grp_ids = E.read_array('PARTDATA', path, snap, 'PartType0/GroupNumber', noH=True,
                           physicalUnits=True, verbose=False, numThreads=8)

    subgrp_ids = E.read_array('PARTDATA', path, snap, 'PartType0/SubGroupNumber', noH=True,
                              physicalUnits=True, verbose=False, numThreads=8)

    part_ids = E.read_array('PARTDATA', path, snap, 'PartType0/ParticleIDs', noH=True,
                            physicalUnits=True, verbose=False, numThreads=8)

    # A copy of this array is needed for the extraction method
    group_part_ids = np.copy(part_ids)

    print("There are", len(subgrp_ids), "particles")

    # Convert IDs to float(groupNumber.SubGroupNumber) format, i.e. group 1 subgroup 11 = 1.00011
    part_halo_ids = np.zeros(grp_ids.size, dtype=float)
    for (ind, g), sg in zip(enumerate(grp_ids), subgrp_ids):
        part_halo_ids[ind] = float(str(int(g)) + '.%05d' % int(sg))

    okinds = np.isin(part_halo_ids, star_halo_ids)
    part_halo_ids = part_halo_ids[okinds]
    part_ids = part_ids[okinds]
    group_part_ids = group_part_ids[okinds]
    gas_all_poss = gas_all_poss[okinds]
    gas_metallicities = gas_metallicities[okinds]
    gas_masses = gas_masses[okinds]
    gas_smooth_ls = gas_smooth_ls[okinds]

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

    # Get the position of each of these galaxies
    gas_mets = {}
    gas_ms = {}
    gas_smls = {}
    all_gas_poss = {}
    for id in star_halo_ids:
        mask = halo_part_inds[id]
        all_gas_poss[id] = gas_all_poss[mask, :]
        gas_mets[id] = gas_metallicities[mask]
        gas_ms[id] = gas_masses[mask]
        gas_smls[id] = gas_smooth_ls[mask]

    print('Got particle IDs')

    del gas_all_poss, gas_metallicities, gas_smooth_ls, gas_masses

    gc.collect()

    print('Got gas properties')

    # ======================== Set up images ========================

    # Define comoving softening length in pMpc
    csoft = 0.001802390 / 0.6777 / (1 + z)

    # Define width
    ini_width = 0.07

    # Compute the resolution
    ini_res = ini_width / csoft
    res = int(np.ceil(ini_res))

    # Compute the new width
    width = csoft * res

    print(width, res)

    # Define range and extent for the images
    imgrange = ((-width / 2, width / 2), (-width / 2, width / 2))
    imgextent = [-width / 2, width / 2, -width / 2, width / 2]

    # Set up aperture objects
    positions = [(res / 2, res / 2)]
    app_radii = np.linspace(0.001, res / 4, 100)
    apertures = [CircularAperture(positions, r=r) for r in app_radii]
    app_radii *= csoft

    # Open the HDF5 file
    hdf = h5py.File(savepath + filename + snap + '.hdf5', 'w')
    hdf.create_dataset('orientation', data=[(0, 1), (1, 2), (0, 2)])
    hdf.create_dataset('galaxy_ids', data=star_halo_ids)  # galaxy ids

    hdf.close()

    # Loop over filters
    for f in filters:

        if f.split(".")[0] == 'FAKE':
            model.create_Lnu_grid(F)  # create new L grid for each filter. In units of erg/s/Hz
        else:
            model.create_Fnu_grid(F, z, cosmo)

        print("Extracting data for filter:", f)

        # Create images for these galaxies
        hls = np.zeros((len(star_halo_ids), 3))
        img_hlrs = np.zeros((len(star_halo_ids), 3))
        ms = np.zeros((len(star_halo_ids), 3))
        tot_l = np.zeros((len(star_halo_ids), 3))
        imgs = np.zeros((len(star_halo_ids), res, res, 3))
        for ind1, (i, j) in enumerate([(0, 1), (1, 2), (0, 2)]):
            for ind2, id in enumerate(star_halo_ids):

                # Get the luminosities
                try:
                    centd_star_pos = all_gal_poss[id] - means[id]
                    centd_gas_pos = all_gas_poss[id] - means[id]
                    # star_rs = calc_3drad(centd_star_pos)
                    # gas_rs = calc_3drad(centd_gas_pos)
                    # okinds_star = star_rs <= 0.03
                    # okinds_gas = gas_rs <= 0.03
                    # centd_star_pos = centd_star_pos[okinds_star]
                    # centd_gas_pos = centd_gas_pos[okinds_gas]
                    # ls = get_lumins(centd_star_pos, gal_ini_ms[id][okinds_star], gal_ages[id][okinds_star],
                    #                 gal_mets[id][okinds_star], gas_mets[id][okinds_gas], centd_gas_pos,
                    #                 gas_ms[id][okinds_gas], gas_smls[id][okinds_gas], lkernel, kbins,
                    #                 conv, model, F, i, j, f)
                    ls = get_lumins(centd_star_pos, gal_ini_ms[id], gal_ages[id], gal_mets[id],
                                    gas_mets[id], centd_gas_pos, gas_ms[id], gas_smls[id], lkernel, kbins,
                                    conv, model, F, i, j, f)

                    # # Centre on luminosity weighted centre
                    # cent = lumin_weighted_centre(centd_star_pos, ls)
                    # centd_star_pos = centd_star_pos - cent

                    # Compute half mass radii
                    try:
                        hls[ind2, ind1], tot_l[ind2, ind1], ms[ind2, ind1] = calc_light_mass_rad(ls, centd_star_pos,
                                                                                                 i, j, gal_app_ms[id])
                    except ValueError:
                        print("Galaxy", id, "had no stars within 30 kpc of COP")
                        continue

                    # Make an image at softening length
                    img = make_soft_img(centd_star_pos, res, i, j, imgrange, ls, gal_smls[id])
                    imgs[ind2, :, :, ind1] = img

                    # Get the image half light radius
                    img_hlrs[ind2, ind1] = get_img_hlr(img, apertures, tot_l[ind2, ind1], app_radii, res, csoft)

                except KeyError:
                    print("Galaxy", id, "Does not appear in the dictionaries")
                    continue

        # Open the HDF5 file
        hdf = h5py.File(savepath + filename + snap + '.hdf5', 'r+')

        # Write out the results for this filter
        filt = hdf.create_group(f)  # create halo group
        filt.create_dataset('particle_half_light_rad', data=hls, dtype=float,
                            compression="gzip")  # Half light radius [Mpc]
        filt.create_dataset('Image_half_light_rad', data=img_hlrs, dtype=float,
                            compression="gzip")  # Half light radius [Mpc]
        filt.create_dataset('Images', data=imgs, dtype=float, compression="gzip")  # Images in erg/s/Hz
        filt.create_dataset('Aperture_Mass_30kpc', data=ms, dtype=float,
                            compression="gzip")  # Aperture mass [Msun * 10*10]
        filt.create_dataset('Aperture_Luminosity_30kpc', data=tot_l, dtype=float,
                            compression="gzip")  # Aperture Luminosity [nJy]

        hdf.close()

        # H, binedges = np.histogram(img_hlrs[:, 0] / csoft, bins=100)
        # binwid = binedges[1] - binedges[0]
        # bin_cents = binedges[1:] - binwid / 2
        #
        # fig = plt.figure()
        # ax = fig.add_subplot(111)
        #
        # ax.bar(bin_cents, H, width=binwid, label="Image")
        #
        # H, binedges = np.histogram(hls[:, 0] / csoft, bins=100)
        # binwid = binedges[1] - binedges[0]
        # bin_cents = binedges[1:] - binwid / 2
        #
        # ax.plot(bin_cents, H, label="Particle", color='r')
        #
        # ax.set_xlabel("$R_{1/2}/\epsilon$")
        # ax.set_ylabel("$N$")
        #
        # handles, labels = ax.get_legend_handles_labels()
        # ax.legend(handles, labels)
        #
        # fig.savefig("plots/imghalflightrads_" + f.split(".")[-1] + ".png", bbox_inches="tight")


# Define SED model
model = models.define_model('BPASSv2.2.1.binary/Chabrier_300')  # define SED grid
model.dust_ISM = ('simple', {'slope': -1.0})
model.dust_BC = ('simple', {'slope': -1.0})
# filters = FLARE.filters.NIRCam
# filename = 'ObsWebbLumins_'
filters = ['FAKE.TH.' + f for f in ['FUV', 'NUV', 'V']]
filename = "RestUV"
F = FLARE.filters.add_filters(filters, new_lam=model.lam)
print(filters)

regions = []
for reg in range(0, 40):
    if reg < 10:
        regions.append('0' + str(reg))
    else:
        regions.append(str(reg))
regions.reverse()
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

get_main(path, snap, savepath, filters, F, model, filename=filename)
