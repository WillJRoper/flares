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


def get_sphere_data(path, snap, part_type, soft):

    # Get positions masses and smoothing lengths
    poss = E.read_array('SNAP', path, snap, 'PartType' + str(part_type) + '/Coordinates',
                        noH=True, numThreads=8)
    if part_type != 1:
        masses = E.read_array('SNAP', path, snap, 'PartType' + str(part_type) + '/Mass',
                              noH=True, numThreads=8) * 10 ** 10
        smls = E.read_array('SNAP', path, snap, 'PartType' + str(part_type) + '/SmoothingLength',
                            noH=True, numThreads=8)
    else:
        masses = np.ones(poss.shape[0])
        smls = np.full_like(masses, soft)

    return poss, masses, smls


def single_sphere(reg, snap, part_type, soft, t=0, p=0, num=0):

    # Define path
    path = '/cosma/home/dp004/dc-rope1/FLARES/FLARES-1/G-EAGLE_' + reg + '/data'

    # Get plot data
    poss_gas, masses_gas, smls_gas = get_sphere_data(path, snap, part_type=0, soft=None)
    poss_DM, masses_DM, smls_DM = get_sphere_data(path, snap, part_type=1, soft=soft)
    poss_stars, masses_stars, smls_stars = get_sphere_data(path, snap, part_type=4, soft=None)

    # Get the spheres centre
    centre, radius, mindist = spherical_region(path, snap)

    # Centre particles
    poss_gas -= centre
    poss_DM -= centre
    poss_stars -= centre

    # Remove boundary particles
    rgas = np.linalg.norm(poss_gas, axis=1)
    rDM = np.linalg.norm(poss_DM, axis=1)
    rstars = np.linalg.norm(poss_stars, axis=1)
    okinds_gas = rgas < 14 / 0.677
    poss_gas = poss_gas[okinds_gas, :]
    masses_gas = masses_gas[okinds_gas]
    smls_gas = smls_gas[okinds_gas]
    okinds_DM = rDM < 14 / 0.677
    poss_DM = poss_DM[okinds_DM, :]
    masses_DM = masses_DM[okinds_DM]
    smls_DM = smls_DM[okinds_DM]
    okinds_stars = rstars < 14 / 0.677
    poss_stars = poss_stars[okinds_stars, :]
    masses_stars = masses_stars[okinds_stars]
    smls_stars = smls_stars[okinds_stars]

    print('There are', len(masses_gas), 'gas particles in the region')
    print('There are', len(masses_DM), 'DM particles in the region')
    print('There are', len(masses_stars), 'star particles in the region')

    # fig = plt.figure(1, figsize=(7, 7))

    # Define the box size
    lbox = (15 / 0.677) * 2

    # Define particles
    qv_gas = QuickView(poss_gas, mass=masses_gas, hsml=smls_gas, plot=False, r=lbox * 3/4, t=t, p=p, roll=0,
                       xsize=1000, ysize=1000, x=0, y=0, z=0, extent=[-lbox / 2., lbox / 2., -lbox / 2., lbox / 2.])
    qv_DM = QuickView(poss_DM, mass=masses_DM, hsml=smls_DM, plot=False, r=lbox * 3/4, t=t, p=p, roll=0,
                       xsize=1000, ysize=1000, x=0, y=0, z=0, extent=[-lbox / 2., lbox / 2., -lbox / 2., lbox / 2.])
    qv_stars = QuickView(poss_stars, mass=masses_stars, hsml=smls_stars, plot=False, r=lbox * 3/4, t=t, p=p, roll=0,
                       xsize=1000, ysize=1000, x=0, y=0, z=0, extent=[-lbox / 2., lbox / 2., -lbox / 2., lbox / 2.])

    # Get colomaps
    cmap_gas = ml.cm.magma
    cmap_dm = ml.cm.plasma
    cmap_stars = ml.cm.Greys_r

    # Get each particle type image
    imgs = {'gas': qv_gas.get_image(), 'dm': qv_DM.get_image(), 'stars': qv_stars.get_image()}
    extents = {'gas': qv_gas.get_extent(), 'dm': qv_DM.get_extent(), 'stars': qv_stars.get_extent()}

    # Convert images to rgb arrays
    rgb_gas = cmap_gas(get_normalised_image(np.log10(imgs['gas']),
                                            vmin=np.log10(imgs['gas'][np.where(imgs['gas'] != 0.0)].min())))
    rgb_DM = cmap_dm(get_normalised_image(np.log10(imgs['dm']),
                                          vmin=np.log10(imgs['dm'][np.where(imgs['dm'] != 0.0)].min())))
    rgb_stars = cmap_stars(get_normalised_image(np.log10(imgs['stars']),
                                                vmin=np.log10(imgs['stars'][np.where(imgs['stars'] != 0.0)].min())))

    fig = plt.figure(figsize=(7, 7))
    ax = fig.add_subplot(111)
    print(extents['gas'], extents['dm'], extents['stars'])
    ax.imshow(rgb_gas, extent=extents['gas'], origin='lower')
    ax.tick_params(axis='both', left=False, top=False, right=False, bottom=False, labelleft=False,
                   labeltop=False, labelright=False, labelbottom=False)

    fig.savefig('plots/spheres/Gas/Gas_single_sphere_reg' + reg + '_snap' + snap + '_angle%05d.png' % num,
                bbox_inches='tight')
    plt.close(fig)

    fig = plt.figure(figsize=(7, 7))
    ax = fig.add_subplot(111)

    ax.imshow(rgb_DM, extent=extents['dm'], origin='lower')
    ax.tick_params(axis='both', left=False, top=False, right=False, bottom=False, labelleft=False,
                   labeltop=False, labelright=False, labelbottom=False)

    fig.savefig('plots/spheres/DM/DM_single_sphere_reg' + reg + '_snap' + snap + '_angle%05d.png' % num,
                bbox_inches='tight')
    plt.close(fig)

    fig = plt.figure(figsize=(7, 7))
    ax = fig.add_subplot(111)

    ax.imshow(rgb_stars, extent=extents['stars'], origin='lower')
    ax.tick_params(axis='both', left=False, top=False, right=False, bottom=False, labelleft=False,
                   labeltop=False, labelright=False, labelbottom=False)

    fig.savefig('plots/spheres/Stars/Stars_single_sphere_reg' + reg + '_snap' + snap + '_angle%05d.png' % num,
                bbox_inches='tight')
    plt.close(fig)

    blend1 = Blend.Blend(rgb_DM, rgb_gas)
    dmgas_output = blend1.Overlay()
    blend2 = Blend.Blend(dmgas_output, rgb_stars)
    rgb_output = blend2.Overlay()

    fig = plt.figure(figsize=(7, 7))
    ax = fig.add_subplot(111)

    ax.imshow(rgb_output, extent=extents['gas'], origin='lower')
    ax.tick_params(axis='both', left=False, top=False, right=False, bottom=False, labelleft=False,
                   labeltop=False, labelright=False, labelbottom=False)

    fig.savefig('plots/spheres/All/all_parts_single_sphere_reg' + reg + '_snap' + snap + '_angle%05d.png'%num,
                bbox_inches='tight')
    plt.close(fig)


# Define softening lengths
csoft = 0.001802390 / 0.677

# # Define region list
# regions = []
# for reg in range(0, 40):
#     if reg < 10:
#         regions.append('0' + str(reg))
#     else:
#         regions.append(str(reg))
#
# # Define snapshots
# snaps = ['000_z015p000', '001_z014p000', '002_z013p000', '003_z012p000', '004_z011p000', '005_z010p000',
#          '006_z009p000', '007_z008p000', '008_z007p000', '009_z006p000', '010_z005p000', '011_z004p770']
#
# # Define a list of regions and snapshots
# reg_snaps = []
# for snap in snaps:
#
#     for reg in regions:
#
#         reg_snaps.append((reg, snap))

ind = int(sys.argv[1])
# print(reg_snaps[ind])
# reg, snap = reg_snaps[ind]
reg, snap = '34', '011_z004p770'
ps = np.linspace(0, 360, 360)
single_sphere(reg, snap, part_type=0, soft=csoft, p=ps[ind], num=ind)
