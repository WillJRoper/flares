import eagle_IO.eagle_IO as E # -*- coding: utf-8 -*-
import numpy as np
import sphviewer as sph


# tag = '009_z006p000'
data = 'PartType1/Coordinates'



tag = '006_z005p971'



print(tag)

coords = E.read_array('SNAP', f'/cosma7/data//Eagle/ScienceRuns/Planck1/L0100N1504/PE/REFERENCE/data', tag, data, noH=True)

print(coords.shape)

for i in range(3):
    print(np.min(coords[:,i]),np.median(coords[:,i]),np.max(coords[:,i]))


s = coords[:,2]<2*(14/0.7)

mass = np.ones(len(coords[s]))

print('here')

Particles = sph.Particles(coords[s], mass)

print('here')
lbox = 100
print('here')

Camera = sph.Camera(r='infinity', t=0, p=0, roll=0, xsize=5000, ysize=5000, x=0, y=0, z=0, extent=[-lbox/2.,lbox/2.,-lbox/2.,lbox/2.])
Scene = sph.Scene(Particles, Camera)
Render = sph.Render(Scene)
extent = Render.get_extent()
img = Render.get_image()

print('here')

np.save(f'data/DM_EAGLE.npy', img)
