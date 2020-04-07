#!/cosma/home/dp004/dc-rope1/.conda/envs/flares-env/bin/python
import matplotlib
matplotlib.use('Agg')
import numpy as np
import sphviewer as sph
from sphviewer.tools import cmaps
import matplotlib.pyplot as plt
import eagle_IO as E
import sys


def get_sphere_data(path, snap, part_type, soft):

    # Get positions masses and smoothing lengths
    poss = E.read_array('SNAP', path, snap, 'PartType' + str(part_type) + '/Coordinates',
                        noH=True, numThreads=8)
    masses = E.read_array('SNAP', path, snap, 'PartType' + str(part_type) + '/Mass',
                          noH=True, numThreads=8)
    if part_type != 1:
        smls = E.read_array('SNAP', path, snap, 'PartType' + str(part_type) + '/SmoothingLength',
                            noH=True, numThreads=8)
    else:
        smls = np.full_like(masses, soft)

    return poss, masses, smls

def single_sphere(reg, snap, part_type, soft, cent):

    # Define path
    path = '/cosma/home/dp004/dc-rope1/FLARES/FLARES-1/G-EAGLE_' + reg + '/data'

    # Get plot data
    poss, masses, smls = get_sphere_data(path, snap, part_type, soft)

    # Centre particles
    poss -= cent

    # Remove boundary particles
    r = np.linalg.norm(poss, axis=1)
    okinds = r < 14 / 0.7
    poss = poss[okinds, :]
    masses = masses[okinds]
    smls = smls[okinds]

    fig = plt.figure(1, figsize=(7, 7))

    Particles = sph.Particles(poss, masses, smls)

    lbox = (15/0.7) * 2
    Camera = sph.Camera(r='infinity', t=0, p=0, roll=0, xsize=5000, ysize=5000, x=0, y=0, z=0,
                        extent=[-lbox / 2., lbox / 2., -lbox / 2., lbox / 2.])
    Scene = sph.Scene(Particles, Camera)
    Render = sph.Render(Scene)
    extent = Render.get_extent()
    img = Render.get_image()

    plt.imshow(np.log10(img), cmap=cmaps.twilight(), extent=extent, origin='lower')
    plt.axis('off')

    fig.savefig('plots/spheres/single_sphere_reg' + reg + '_snap' + snap + '_PartType' + str(part_type) + '.png',
                bbox_inches='tight', dpi=300)


# Load centres
cents = np.loadtxt('Region_cents.txt')
print(cents)
# Define softening lengths
csoft = 0.001802390 / 0.677

# Define region list
regions = []
for reg in range(0, 40):
    if reg < 10:
        regions.append('0' + str(reg))
    else:
        regions.append(str(reg))

# Define snapshots
snaps = ['000_z015p000', '001_z014p000', '002_z013p000', '003_z012p000', '004_z011p000', '005_z010p000',
         '006_z009p000', '007_z008p000', '008_z007p000', '009_z006p000', '010_z005p000', '011_z004p770']
./
# Define a list of regions and snapshots
reg_snaps = []
regcents = []
for reg in regions:

    c = cents[reg]

    for snap in snaps:

        reg_snaps.append((reg, snap))
        regcents.append(c)

ind = int(sys.argv[1])
print(reg_snaps[ind], regcents[ind])
reg, snap = reg_snaps[ind]
single_sphere(reg, snap, part_type=1, soft=csoft, cent=regcents[ind])
