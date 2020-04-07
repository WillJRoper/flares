import matplotlib
matplotlib.use('Agg')
import numpy as np
import sphviewer as sph
from sphviewer.tools import cmaps
import matplotlib.pyplot as plt

tag = '009_z006p000'
sim = 1

q = 'PartType1/Coordinates'

coords = np.load(f'data/Central_{q.replace("/","_")}_{sim:02d}_{tag}.npy','r')

print(coords.shape)

fig = plt.figure(1, figsize=(7,7))

mass = np.ones(len(coords))
Particles = sph.Particles(coords, mass)

lbox = 2.
Camera = sph.Camera(r='infinity', t=0, p=0, roll=0, xsize=5000, ysize=5000, x=0, y=0, z=0, extent=[-lbox/2.,lbox/2.,-lbox/2.,lbox/2.])
Scene = sph.Scene(Particles, Camera)
Render = sph.Render(Scene)
extent = Render.get_extent()
img = Render.get_image()

plt.imshow(np.log10(img), cmap=cmaps.twilight(), origin='lower')
plt.axis('off')

fig.savefig('figs/single.png')
