import matplotlib
matplotlib.use('Agg')
import eagle_IO.eagle_IO as E # -*- coding: utf-8 -*-
import numpy as np
from sphviewer.tools import QuickView, cmaps
import matplotlib.pyplot as plt

Sim = '/cosma/home/dp004/dc-wilk2/FLARES/FLARES-1/G-EAGLE_06/data'

tag = '010_z005p000'

data = 'PartType4/Coordinates'

coords = E.read_array('SNAP', Sim, tag, data, noH=True)

print(coords.shape)






med2 = np.median(coords[:,2])


coords_rec = coords - np.median(coords[:,:], axis=0)
r = np.linalg.norm(coords_rec, axis=1)
s = r<14/0.7


# qv = QuickView(coords_rec[s], r='infinity', plot=False,logscale=True)
#


# fig = plt.figure(1, figsize=(10,10),dpi=400)
# img = qv.get_image()
# print(np.min(img), np.max(img))
#
# # plt.imshow(img, extent=qv.get_extent(), origin='lower', cmap=cmaps.twilight())
# plt.imshow(img, extent=qv.get_extent(), origin='lower', cmap='bone')
#
# fig.savefig('stars_sphere.png')


import sphviewer as sph

mass = np.ones(len(coords_rec[s]))

Particles = sph.Particles(coords_rec[s], mass)


lbox = (15/0.7)*2

Camera = sph.Camera(r='infinity', t=0, p=0, roll=0, xsize=5000, ysize=5000, x=0, y=0, z=0, extent=[-lbox/2.,lbox/2.,-lbox/2.,lbox/2.])

Scene = sph.Scene(Particles, Camera)

Render = sph.Render(Scene)

extent = Render.get_extent()


img = Render.get_image()
img = np.log10(img)

print(np.min(img), np.max(img))

fig = plt.figure(1,figsize=(10,10))
plt.axis('off')
plt.imshow(img, extent=[extent[0]+Scene.Camera.get_params()['x'],extent[1]+Scene.Camera.get_params()['x'],extent[2]+Scene.Camera.get_params()['y'],extent[3]+Scene.Camera.get_params()['y']], cmap=cmaps.sunlight(), origin='lower')
plt.savefig('stars_sphere.png')
