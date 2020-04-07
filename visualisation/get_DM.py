import eagle_IO.eagle_IO as E # -*- coding: utf-8 -*-
import numpy as np
import sphviewer as sph


# tag = '009_z006p000'
data = 'PartType1/Coordinates'


tags = []
for i in range(11):
    z = 15 - i
    tags.append(f'{i:03d}_z{z:03d}p000')

print(tags)

tags = ['009_z006p000']



for sim in np.arange(10,41,1):

    print(sim)

    for tag in tags:

        print(tag)

        coords = E.read_array('SNAP', f'/cosma/home/dp004/dc-wilk2/FLARES/FLARES-1/G-EAGLE_{sim:02d}/data', tag, data, noH=True)

        print(coords.shape)

        med2 = np.median(coords[:,2])

        coords_rec = coords - np.median(coords[:,:], axis=0)
        r = np.linalg.norm(coords_rec, axis=1)
        s = r<14/0.7

        mass = np.ones(len(coords_rec[s]))

        Particles = sph.Particles(coords_rec[s], mass)

        lbox = (15/0.7)*2

        Camera = sph.Camera(r='infinity', t=0, p=0, roll=0, xsize=5000, ysize=5000, x=0, y=0, z=0, extent=[-lbox/2.,lbox/2.,-lbox/2.,lbox/2.])
        Scene = sph.Scene(Particles, Camera)
        Render = sph.Render(Scene)
        extent = Render.get_extent()
        img = Render.get_image()

        np.save(f'data/DM_G-EAGLE_{sim:02d}_{tag}.npy', img)
