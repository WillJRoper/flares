#!/cosma/home/dp004/dc-rope1/.conda/envs/flares-env/bin/python
import matplotlib
matplotlib.use('Agg')
import numpy as np
import matplotlib.gridspec as gridspec
from sphviewer.tools import cmaps, Blend, QuickView
import matplotlib as ml
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.spatial import ConvexHull
import eagle_IO.eagle_IO as E
import sphviewer
import sys
import h5py
import pickle


def _sphere(coords, a, b, c, r):

    # Equation of a sphere
    x, y, z = coords[:, 0], coords[:, 1], coords[:, 2]

    return (x - a) ** 2 + (y - b) ** 2 + (z - c) ** 2 - r ** 2


def spherical_region(sim, snap):
    """
    Inspired from David Turner's suggestion
    """

    gas_cood = E.read_array('PARTDATA', sim, snap, '/PartType1/Coordinates',
                           noH=True, physicalUnits=False, numThreads=4)  # gas particle coordinates

    hull = ConvexHull(gas_cood)

    cen = [np.median(gas_cood[:, 0]), np.median(gas_cood[:, 1]), np.median(gas_cood[:,2])]
    pedge = gas_cood[hull.vertices]  #edge particles
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


def set_up_single_sphere(reg, snap, soft, centre, part_type):

    # Define path
    path = '/cosma/home/dp004/dc-rope1/FLARES/FLARES-1/G-EAGLE_' + reg + '/data'

    # Get plot data
    poss, masses, smls = get_sphere_data(path, snap, part_type, soft=soft)

    # Centre particles
    poss -= centre

    print("Centered positions")

    # Calculate radii
    rs = np.linalg.norm(poss, axis=1)

    # Remove boundary particles
    okinds = rs < 14 / 0.677
    poss = poss[okinds, :]
    masses = masses[okinds]
    smls = smls[okinds]

    print('There are', len(masses), part_type, 'particles in region', reg)

    # Set up pysphviewer objects
    part = sphviewer.Particles(poss, mass=masses, hsml=smls)
    scene = sphviewer.Scene(part)

    return scene

def get_single_sphere_imgs(scene, p, t):

    # Define the box size
    lbox = (15 / 0.677) * 2

    scene.update_camera(r=lbox * 3/4, t=t, p=p, roll=0, xsize=5000, ysize=5000, x=0, y=0, z=0,
                        extent=[-lbox / 2., lbox / 2., -lbox / 2., lbox / 2.])

    # Define particles
    render = sphviewer.Render(scene)

    # Get image
    img = render.get_image()
    extent = render.get_extent()

    return img, extent


def blend(rgb_DM, rgb_gas):

    blend1 = Blend.Blend(rgb_DM, rgb_gas)
    gasgas_output = blend1.Overlay()
    rgb_output = gasgas_output

    return rgb_output


def spheregrid(snap, num):

    # Define region list
    regions = []
    for reg in range(0, 40):
        if reg < 10:
            regions.append('0' + str(reg))
        else:
            regions.append(str(reg))

    ovdens = np.loadtxt("region_overdensity.txt", dtype=float)
    sinds = np.argsort(ovdens)[::-1]
    regions = np.array(regions)
    regions = regions[sinds]

    fig = plt.figure(figsize=(8 * 3.95, 5 * 4))
    gs = gridspec.GridSpec(nrows=5, ncols=8)
    gs.update(wspace=0.0, hspace=0.0)

    pltgrid = []
    for i in range(5):
        for j in range(8):
            pltgrid.append((i, j))

    for reg, (i, j) in zip(regions, pltgrid):

        print(reg, i, j)

        # Load the current snapshot data
        hdf = h5py.File(f'spheresdata/spheregird_img_data_{reg}_{snap}_gas.hdf5', 'r')

        img, extent = hdf[str(num)]['img'][...], hdf[str(num)].attrs['extent']

        hdf.close()

        ax = fig.add_subplot(gs[i, j])

        # Get colormaps
        cmap = cmaps.desert()

        scaled_img = np.log10(img)

        print(scaled_img.max())

        # Convert images to rgb arrays
        rgb = cmap(get_normalised_image(scaled_img, vmin=10, vmax=14))

        ax.imshow(rgb, extent=extent, origin='lower')
        ax.tick_params(axis='both', left=False, top=False, right=False, bottom=False, labelleft=False,
                       labeltop=False, labelright=False, labelbottom=False)

        # ax.spines['top'].set_visible(False)
        # ax.spines['right'].set_visible(False)
        # ax.spines['left'].set_visible(False)
        # ax.spines['bottom'].set_visible(False)

    fig.savefig('plots/spheres/Gas/gas_grid_sphere_snap' + snap + '_angle%05d.png'%num,
                bbox_inches='tight', facecolor='k')
    plt.close(fig)


# Define softening lengths
csoft = 0.001802390 / 0.677

# Define region list
regions = []
for reg in range(0, 40):
    if reg < 10:
        regions.append('0' + str(reg))
    else:
        regions.append(str(reg))

# Get colormaps
cmap_gas = cmaps.twilight()

ind = int(sys.argv[1])

snap = '011_z004p770'
nframes = 180

# =============================================================================

# reg = regions[ind]
#
# # Define path
# path = '/cosma/home/dp004/dc-rope1/FLARES/FLARES-1/G-EAGLE_' + reg + '/data'
#
# # Get the spheres centre
# centre, radius, mindist = spherical_region(path, snap)
#
#
# # Define rotations
# ps = np.linspace(0, 360, nframes)
#
# # Set up particle objects
# gas_scene = set_up_single_sphere(reg, snap, csoft, centre, part_type=0)
#
# # Load the current snapshot data
# hdf = h5py.File(f'spheresdata/spheregird_img_data_{reg}_{snap}_gas.hdf5', 'w')
#
# for num, p in enumerate(ps):
#
#     rgb_gas, extent = get_single_sphere_imgs(gas_scene, p, t=0)
#
#     frame = hdf.create_group(str(num))
#     frame.create_dataset('img', shape=rgb_gas.shape, dtype=float, data=rgb_gas,
#                             compression='gzip')
#     frame.attrs['extent'] = extent
#
# hdf.close()

# =============================================================================

spheregrid(snap, ind)
#
# for num in range(nframes):
#     spheregrid(snap, nframes, num)
