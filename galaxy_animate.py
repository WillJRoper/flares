#!/cosma/home/dp004/dc-rope1/.conda/envs/flares-env/bin/python
import matplotlib
matplotlib.use('Agg')
import numpy as np
from sphviewer.tools import cmaps, Blend, QuickView
import matplotlib as ml
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.spatial import ConvexHull
import eagle_IO as E
import sys
import os


def _sphere(coords, a, b, c, r):

    # Equation of a sphere
    x, y, z = coords[:, 0], coords[:, 1], coords[:, 2]

    return (x - a) ** 2 + (y - b) ** 2 + (z - c) ** 2 - r ** 2


def spherical_region(sim, snap):
    """
    Inspired from David Turner's suggestion
    """

    dm_cood = E.read_array('PARTDATA', sim, snap, '/PartType1/Coordinates',
                           noH=True, physicalUnits=False, numThreads=4)  # dm particle coordinates

    hull = ConvexHull(dm_cood)

    cen = [np.median(dm_cood[:, 0]), np.median(dm_cood[:, 1]), np.median(dm_cood[:,2])]
    pedge = dm_cood[hull.vertices]  #edge particles
    y_obs = np.zeros(len(pedge))
    p0 = np.append(cen, 15 / 0.677)

    popt, pcov = curve_fit(_sphere, pedge, y_obs, p0, method='lm', sigma=np.ones(len(pedge)) * 0.001)
    dist = np.sqrt(np.sum((pedge-popt[:3])**2, axis=1))
    centre, radius, mindist = popt[:3], popt[3], np.min(dist)

    return centre, radius, mindist


def get_normalised_image(img, vmin=None, vmax=None):

    if vmin == None:
        vmin = np.min(img)
    if vmax == None:
        vmax = np.max(img)

    img = np.clip(img, vmin, vmax)
    img = (img - vmin) / (vmax - vmin)

    return img


def get_galaxy_data(g, sg, path, snap, soft):

    # Load all necessary arrays
    dm_all_poss = E.read_array('PARTDATA', path, snap, 'PartType1/Coordinates', noH=True,
                               physicalUnits=True, numThreads=8)
    dmgrp_ids = E.read_array('PARTDATA', path, snap, 'PartType1/GroupNumber', numThreads=8)
    dmsubgrp_ids = E.read_array('PARTDATA', path, snap, 'PartType1/SubGroupNumber', numThreads=8)

    # Get gas particle information
    gas_all_poss = E.read_array('PARTDATA', path, snap, 'PartType0/Coordinates', noH=True, physicalUnits=True,
                                numThreads=8)
    ggrp_ids = E.read_array('PARTDATA', path, snap, 'PartType0/GroupNumber', numThreads=8)
    gsubgrp_ids = E.read_array('PARTDATA', path, snap, 'PartType0/SubGroupNumber', numThreads=8)
    gas_masses = E.read_array('PARTDATA', path, snap, 'PartType0/Mass', noH=True, physicalUnits=True,
                              numThreads=8) * 10 ** 10
    gas_smooth_ls = E.read_array('PARTDATA', path, snap, 'PartType0/SmoothingLength', noH=True, physicalUnits=True,
                                 numThreads=8)
    # Get particle masks for this galaxy
    dmmask = (dmgrp_ids == g) & (dmsubgrp_ids == sg)
    gasmask = (ggrp_ids == g) & (gsubgrp_ids == sg)

    # Get dark matter data
    poss_DM = dm_all_poss[dmmask, :]
    masses_DM = np.ones(poss_DM.shape[0])
    dm_smls = np.full_like(masses_DM, soft)

    # Get gas data
    gas_poss = gas_all_poss[gasmask, :]
    gas_ms = gas_masses[gasmask]
    gas_smls = gas_smooth_ls[gasmask]

    return poss_DM, masses_DM, dm_smls, gas_poss, gas_ms, gas_smls


def single_galaxy(g, sg, reg, snap, soft, t=0, p=0, num=0):

    # Define path
    path = '/cosma/home/dp004/dc-rope1/FLARES/FLARES-1/G-EAGLE_' + reg + '/data'

    # Get plot data
    poss_DM, masses_DM, smls_DM, poss_gas, masses_gas, smls_gas = get_galaxy_data(g, sg, path, snap, soft=soft)

    # Get the spheres centre
    centrex = np.sum(poss_DM[:, 0] * masses_DM, axis=0) / np.sum(masses_DM)
    centrey = np.sum(poss_DM[:, 1] * masses_DM, axis=0) / np.sum(masses_DM)
    centrez = np.sum(poss_DM[:, 2] * masses_DM, axis=0) / np.sum(masses_DM)
    centre = np.array([centrex, centrey, centrez])

    # Centre particles
    poss_gas -= centre
    poss_DM -= centre

    # Remove boundary particles
    rgas = np.linalg.norm(poss_gas, axis=1)
    rDM = np.linalg.norm(poss_DM, axis=1)
    okinds_gas = rgas < rDM.max()
    poss_gas = poss_gas[okinds_gas, :]
    masses_gas = masses_gas[okinds_gas]
    smls_gas = smls_gas[okinds_gas]
    print(rDM.max())
    print('There are', len(masses_gas), 'gas particles in the region')
    print('There are', len(masses_DM), 'DM particles in the region')

    # Define the box size
    lbox = (rDM.max() + 0.05 * rDM.max()) * 2

    # Define particles
    # qv_gas = QuickView(poss_gas, mass=masses_gas, hsml=smls_gas, plot=False, r=lbox * 3/4, t=t, p=p, roll=0,
    #                    xsize=500, ysize=500, x=0, y=0, z=0, extent=[-lbox / 2., lbox / 2., -lbox / 2., lbox / 2.])
    # qv_DM = QuickView(poss_DM, mass=masses_DM, hsml=smls_DM, plot=False, r=lbox * 3/4, t=t, p=p, roll=0,
    #                    xsize=500, ysize=500, x=0, y=0, z=0, extent=[-lbox / 2., lbox / 2., -lbox / 2., lbox / 2.])
    qv_gas = QuickView(poss_gas, mass=masses_gas, plot=False, r=lbox * 3/4, t=t, p=p, roll=0,
                       xsize=5000, ysize=5000, x=0, y=0, z=0, extent=[-lbox / 2., lbox / 2., -lbox / 2., lbox / 2.])
    qv_DM = QuickView(poss_DM, mass=masses_DM, plot=False, r=lbox * 3/4, t=t, p=p, roll=0,
                       xsize=5000, ysize=5000, x=0, y=0, z=0, extent=[-lbox / 2., lbox / 2., -lbox / 2., lbox / 2.])

    # Get colomaps
    cmap_gas = ml.cm.plasma
    cmap_dm = ml.cm.Greys_r

    # Get each particle type image
    imgs = {'gas': qv_gas.get_image(), 'dm': qv_DM.get_image()}
    extents = {'gas': qv_gas.get_extent(), 'dm': qv_DM.get_extent()}

    fig = plt.figure(figsize=(7, 7))
    ax = fig.add_subplot(111)

    ax.imshow(imgs['gas'], extent=extents['gas'], origin='lower', cmap='Greys_r')
    ax.tick_params(axis='both', left=False, top=False, right=False, bottom=False, labelleft=False,
                   labeltop=False, labelright=False, labelbottom=False)

    fig.savefig('plots/spheres/Gas/Gas_galaxy_reg' + reg + '_snap' + snap + '_angle%05d.png' % num,
                bbox_inches='tight')
    plt.close(fig)

    fig = plt.figure(figsize=(7, 7))
    ax = fig.add_subplot(111)

    ax.imshow(imgs['dm'], extent=extents['dm'], origin='lower', cmap='magma')
    ax.tick_params(axis='both', left=False, top=False, right=False, bottom=False, labelleft=False,
                   labeltop=False, labelright=False, labelbottom=False)

    fig.savefig('plots/spheres/DM/DM_galaxy_reg' + reg + '_snap' + snap + '_angle%05d.png' % num,
                bbox_inches='tight')

    print(imgs['dm'][np.where(imgs['dm'] != 0.0)].min())
    print(imgs['gas'][np.where(imgs['gas'] != 0.0)].min())

    # Convert images to rgb arrays
    rgb_gas = cmap_gas(get_normalised_image(imgs['gas'], vmin=imgs['gas'][np.where(imgs['gas'] != 0.0)].min()))
    rgb_DM = cmap_dm(get_normalised_image(imgs['dm'], vmin=imgs['dm'][np.where(imgs['dm'] != 0.0)].min()))

    blend = Blend.Blend(rgb_DM, rgb_gas)
    dmgas_output = blend.Overlay()
    rgb_output = dmgas_output

    fig = plt.figure(figsize=(7, 7))
    ax = fig.add_subplot(111)

    ax.imshow(rgb_output, extent=extents['gas'], origin='lower')
    ax.tick_params(axis='both', left=False, top=False, right=False, bottom=False, labelleft=False,
                   labeltop=False, labelright=False, labelbottom=False)

    fig.savefig('plots/spheres/Galaxies/galid' + str(g) + 'p' + str(sg) + '/all_parts_galaxy_reg' + reg
                + '_snap' + snap + '_galid:' + str(g) + 'p' + str(sg) + '_angle%05d.png'%num,
                bbox_inches='tight')
    plt.close(fig)


# Define softening lengths
soft = 0.001802390 / 0.677

# # Define region list
regions = []
for reg in range(0, 40):
    if reg < 10:
        regions.append('0' + str(reg))
    else:
        regions.append(str(reg))

# Define snapshot list
snaps = ['000_z015p000', '001_z014p000', '002_z013p000', '003_z012p000', '004_z011p000', '005_z010p000',
         '006_z009p000', '007_z008p000', '008_z007p000', '009_z006p000', '010_z005p000', '011_z004p770']

# Define angles
ps = np.linspace(0, 360, 360)

ind = int(sys.argv[1])
reg, snap, g, sg = regions[int(sys.argv[2])], snaps[int(sys.argv[3])], int(sys.argv[4]), int(sys.argv[5])
if not 'galid' + str(g) + 'p' + str(sg) in os.listdir('plots/spheres/Galaxies/'):
    os.mkdir('galid' + str(g) + 'p' + str(sg))
print('Phi=', ps[ind], 'Region:', reg, 'Snapshot:', snap, 'Galaxy:', str(g) + '.' + str(sg))
single_galaxy(g, sg, reg, snap, soft, t=0, p=ps[ind], num=ind)
