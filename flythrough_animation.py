#!/cosma/home/dp004/dc-rope1/.conda/envs/flares-env/bin/python
import matplotlib
matplotlib.use('Agg')
import numpy as np
import sphviewer as sph
from sphviewer.tools import cmaps, Blend, camera_tools
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
    poss = E.read_array('PARTDATA', path, snap, 'PartType' + str(part_type) + '/Coordinates',
                        noH=True, numThreads=8)
    if part_type != 1:
        masses = E.read_array('PARTDATA', path, snap, 'PartType' + str(part_type) + '/Mass',
                              noH=True, numThreads=8) * 10 ** 10
        smls = E.read_array('PARTDATA', path, snap, 'PartType' + str(part_type) + '/SmoothingLength',
                            noH=True, numThreads=8)
    else:
        masses = np.ones(poss.shape[0])
        smls = np.full_like(masses, soft)

    return poss, masses, smls


def single_sphere(reg, snap, soft):

    # Define path
    path = '/cosma/home/dp004/dc-rope1/FLARES/FLARES-1/G-EAGLE_' + reg + '/data'

    # Get plot data
    poss_gas, masses_gas, smls_gas = get_sphere_data(path, snap, part_type=0, soft=None)
    poss_DM, masses_DM, smls_DM = get_sphere_data(path, snap, part_type=1, soft=soft)
    
    # Get centres of groups
    grp_cops = E.read_array('SUBFIND', path, snap, 'FOF/GroupCentreOfPotential',
                            noH=True, numThreads=8)
    grp_ms = E.read_array('SUBFIND', path, snap, 'FOF/GroupMass',
                            noH=True, numThreads=8)

    # Get the spheres centre
    centre, radius, mindist = spherical_region(path, snap)

    # Centre particles
    poss_gas -= centre
    poss_DM -= centre

    # Remove boundary particles
    rgas = np.linalg.norm(poss_gas, axis=1)
    rDM = np.linalg.norm(poss_DM, axis=1)
    okinds_gas = rgas < 14 / 0.677
    poss_gas = poss_gas[okinds_gas, :]
    masses_gas = masses_gas[okinds_gas]
    smls_gas = smls_gas[okinds_gas]
    okinds_DM = rDM < 14 / 0.677
    poss_DM = poss_DM[okinds_DM, :]
    masses_DM = masses_DM[okinds_DM]
    smls_DM = smls_DM[okinds_DM]

    print('There are', len(masses_gas), 'gas particles in the region')
    print('There are', len(masses_DM), 'DM particles in the region')

    # Set up particle objects
    P_DM = sph.Particles(poss_DM, mass=masses_DM, hsml=smls_DM)
    P_gas = sph.Particles(poss_gas, mass=masses_gas, hsml=smls_gas)

    # Initialise the scene
    S_DM = sph.Scene(P_DM)
    S_gas = sph.Scene(P_gas)

    # Define targets
    targets = [[0, 0, 0]]
    targets.append(grp_cops[np.argmax(grp_ms)] - centre)

    # Define the box size
    lbox = (15 / 0.677) * 2

    # Define anchors
    anchors = {}
    anchors['sim_times'] = [0.0, 1.0, 'pass', 3.0, 'same', 'same', 'same']
    anchors['id_frames'] = [0, 180, 750, 840, 930, 1500, 1680]
    anchors['id_targets'] = [0, 'pass', 1, 'same', 'same', 'pass', 0]
    anchors['r'] = [lbox * 3/4, 'pass', lbox / 10, 'same', 'same', 'pass', lbox * 3/4]
    anchors['t'] = [0, 'pass', 'pass', 180, 'pass', 'pass', 0]
    anchors['p'] = [0, 'pass', 350, 'pass', 0, 'pass', 359]
    anchors['zoom'] = [1., 'pass', 8, 'same', 'same', 'pass', 0]
    anchors['extent'] = [10, 'pass', 60, 'same', 'same', 'pass', 10]

    # Define the camera trajectory
    data = camera_tools.get_camera_trajectory(targets, anchors)

    num = 0
    for i in data:
        print(num)
        i['xsize'] = 500
        i['ysize'] = 500
        i['roll'] = 0
        S = sph.Scene(P_DM)
        S.update_camera(**i)
        R = sph.Render(S)
        R.set_logscale()
        img = R.get_image()

        try:
            vmin = img[np.where(img != 0)].min()
            vmax = img.max()
        except ValueError:
            vmin = 0
            vmax = 1

        plt.imsave('plots/spheres/All/all_parts_ani_reg' + reg + '_snap' + snap + '_angle%05d.png'%num, img,
                   vmin=vmin, vmax=vmax, cmap='magma')
        plt.close()
        num += 1

    # # Define particles
    # qv_gas = QuickView(poss_gas, mass=masses_gas, hsml=smls_gas, plot=False, r=lbox * 3/4, t=t, p=p, roll=0,
    #                    xsize=5000, ysize=5000, x=0, y=0, z=0, extent=[-lbox / 2., lbox / 2., -lbox / 2., lbox / 2.])
    # qv_DM = QuickView(poss_DM, mass=masses_DM, hsml=smls_DM, plot=False, r=lbox * 3/4, t=t, p=p, roll=0,
    #                    xsize=5000, ysize=5000, x=0, y=0, z=0, extent=[-lbox / 2., lbox / 2., -lbox / 2., lbox / 2.])
    #
    # # Get colomaps
    # cmap_gas = ml.cm.plasma
    # cmap_dm = ml.cm.Greys_r
    #
    # # Get each particle type image
    # imgs = {'gas': qv_gas.get_image(), 'dm': qv_DM.get_image()}
    # extents = {'gas': qv_gas.get_extent(), 'dm': qv_DM.get_extent()}
    #
    # # Convert images to rgb arrays
    # rgb_gas = cmap_gas(get_normalised_image(np.log10(imgs['gas']),
    #                                         vmin=np.log10(imgs['gas'][np.where(imgs['gas'] != 0.0)].min()+7)))
    # rgb_DM = cmap_dm(get_normalised_image(np.log10(imgs['dm']),
    #                                       vmin=np.log10(imgs['dm'][np.where(imgs['dm'] != 0.0)].min()-0.5)))
    #
    # blend = Blend.Blend(rgb_DM, rgb_gas)
    # rgb_output = blend.Overlay()
    #
    # fig = plt.figure(figsize=(7, 7))
    # ax = fig.add_subplot(111)
    #
    # ax.imshow(rgb_output, extent=extents['gas'], origin='lower')
    # ax.tick_params(axis='both', left=False, top=False, right=False, bottom=False, labelleft=False,
    #                labeltop=False, labelright=False, labelbottom=False)
    #
    # fig.savefig('plots/spheres/All/all_parts_single_sphere_reg' + reg + '_snap' + snap + '_angle%05d.png'%num,
    #             bbox_inches='tight')
    # plt.close(fig)


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

# print(reg_snaps[ind])
# reg, snap = reg_snaps[ind]
reg, snap = '00', '010_z005p000'
single_sphere(reg, snap, soft=csoft)
