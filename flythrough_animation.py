#!/cosma/home/dp004/dc-rope1/.conda/envs/flares-env/bin/python
import matplotlib as ml
ml.use('Agg')
import numpy as np
import sphviewer as sph
from sphviewer.tools import cmaps, Blend, camera_tools
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.spatial import ConvexHull
import eagle_IO as E
import sys
from guppy import hpy; h=hpy()
import gc
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

    print('Defined convex hull')

    cen = [np.median(dm_cood[:, 0]), np.median(dm_cood[:, 1]), np.median(dm_cood[:,2])]
    pedge = dm_cood[hull.vertices]  #edge particles
    y_obs = np.zeros(len(pedge))
    p0 = np.append(cen, 15 / 0.677)

    print('Defined convex hull')

    popt, pcov = curve_fit(_sphere, pedge, y_obs, p0, method='lm', sigma=np.ones(len(pedge)) * 0.001)
    dist = np.sqrt(np.sum((pedge-popt[:3])**2, axis=1))
    centre, radius, mindist = popt[:3], popt[3], np.min(dist)

    print('computed fit')

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


def getimage(path, snap, soft, num, centre, data, part_type):

    # Get plot data
    if part_type == 0:
        poss_gas, masses_gas, smls_gas = get_sphere_data(path, snap, part_type=0, soft=None)
    elif part_type == 1:
        poss_gas, masses_gas, smls_gas = get_sphere_data(path, snap, part_type=1, soft=soft)
    else:
        return -1

    # Centre particles
    poss_gas -= centre

    # Remove boundary particles
    rgas = np.linalg.norm(poss_gas, axis=1)
    okinds_gas = rgas < 14 / 0.677
    poss_gas = poss_gas[okinds_gas, :]
    masses_gas = masses_gas[okinds_gas]
    smls_gas = smls_gas[okinds_gas]

    if part_type == 0:
        print('There are', len(masses_gas), 'gas particles in the region')
    elif part_type == 1:
        print('There are', len(masses_gas), 'dark matter particles in the region')

    # Set up particle objects
    P_gas = sph.Particles(poss_gas, mass=masses_gas, hsml=smls_gas)

    # Initialise the scene
    S_gas = sph.Scene(P_gas)

    i = data[num]
    i['xsize'] = 5000
    i['ysize'] = 5000
    i['roll'] = 0
    S_gas.update_camera(**i)
    R_gas = sph.Render(S_gas)
    R_gas.set_logscale()
    img_gas = R_gas.get_image()

    if part_type == 0:
        np.save('animationdata/gas_animationdata_reg' + reg + '_snap' + snap + '_angle%05d.npy'%num, img_gas)
    if part_type == 1:
        np.save('animationdata/dm_animationdata_reg' + reg + '_snap' + snap + '_angle%05d.npy'%num, img_gas)

    # fig = plt.figure()
    # ax = fig.add_subplot(111)
    #
    # ax.hist(get_normalised_image(img_gas).ravel(), bins=256, range=(0.0, 1.0), fc='k', ec='k')
    #
    # if part_type == 0:
    #     fig.savefig('plots/spheres/All/histDM_animation_reg' + reg + '_snap' + snap + '_angle%05d.png'%num,
    #                 bbox_inches='tight')
    # if part_type == 1:
    #     fig.savefig('plots/spheres/All/histgas_animation_reg' + reg + '_snap' + snap + '_angle%05d.png'%num,
    #                 bbox_inches='tight')
    # plt.close(fig)

    vmax_gas = img_gas.max()
    vmin_gas = vmax_gas * 0.5

    # Get colormaps
    if part_type == 0:
        cmap_gas = cmaps.twilight()
    elif part_type == 1:
        cmap_gas = ml.cm.Greys_r

    # Convert images to rgb arrays
    rgb_gas = cmap_gas(get_normalised_image(img_gas, vmin=vmin_gas))

    return rgb_gas, R_gas.get_extent()


def single_sphere(reg, snap, soft, num, runall=True):

    if not runall:
        if 'gas_animationdata_reg' + reg + '_snap' + snap + '_angle%05d.npy'%num in os.listdir('animationdata/') and \
                'dm_animationdata_reg' + reg + '_snap' + snap + '_angle%05d.npy'%num in os.listdir('animationdata/'):
            return

    # Define path
    path = '/cosma/home/dp004/dc-rope1/FLARES/FLARES-1/G-EAGLE_' + reg + '/data'

    # Get centres of groups
    grp_cops = E.read_array('SUBFIND', path, snap, 'FOF/GroupCentreOfPotential',
                            noH=True, numThreads=8)
    grp_ms = E.read_array('SUBFIND', path, snap, 'FOF/GroupMass',
                          noH=True, numThreads=8)

    # Get the spheres centre
    centre, radius, mindist = spherical_region(path, snap)

    # Define targets
    sinds = np.argsort(grp_ms)
    grp_cops = grp_cops[sinds]
    targets = [[0, 0, 0]]
    targets.append(grp_cops[0, :] - centre)
    targets.append(grp_cops[1, :] - centre)

    del grp_cops, grp_ms
    gc.collect()

    # Define the box size
    lbox = (15 / 0.677) * 2

    # Define anchors
    anchors = {}
    anchors['sim_times'] = [0.0, 'same', 'same', 'same', 'same', 'same', 'same', 'same']
    anchors['id_frames'] = [0, 180, 750, 840, 930, 1500, 1680, 2000]
    anchors['id_targets'] = [0, 'pass', 2, 'pass', 'pass', 'pass', 'pass', 0]
    anchors['r'] = [lbox * 3 / 4, 'pass', lbox / 100, 'same', 'same', 'pass', 'pass', lbox * 3 / 4]
    anchors['t'] = [0, 'pass', 'pass', -180, 'pass', -270, 'pass', -360]
    anchors['p'] = [0, 'pass', 'pass', 'pass', 'pass', 'same', 'pass', 360 * 3]
    anchors['zoom'] = [1., 'same', 'same', 'same', 'same', 'same', 'same', 'same']
    anchors['extent'] = [10, 'same', 'same', 'same', 'same', 'same', 'same', 'same']

    # Define the camera trajectory
    data = camera_tools.get_camera_trajectory(targets, anchors)

    # Get images
    rgb_DM, extent = getimage(path, snap, soft, num, centre, data, part_type=1)
    rgb_gas, _ = getimage(path, snap, soft, num, centre, data, part_type=0)

    blend = Blend.Blend(rgb_DM, rgb_gas)
    rgb_output = blend.Overlay()

    fig = plt.figure(figsize=(4, 4))
    ax = fig.add_subplot(111)

    ax.imshow(rgb_output, extent=extent, origin='lower')
    ax.tick_params(axis='both', left=False, top=False, right=False, bottom=False, labelleft=False,
                   labeltop=False, labelright=False, labelbottom=False)

    fig.savefig('plots/spheres/All/all_parts_animation_reg' + reg + '_snap' + snap + '_angle%05d.png'%num,
                bbox_inches='tight')
    plt.close(fig)


# Define softening lengths
csoft = 0.001802390 / 0.677

reg, snap = '00', '010_z005p000'
if __name__ == '__main__':
    single_sphere(reg, snap, soft=csoft, num=int(sys.argv[1]))
