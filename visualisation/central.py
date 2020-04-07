import eagle_IO.eagle_IO as E # -*- coding: utf-8 -*-
import numpy as np




# -- extract central 2 Mpc cube


tag = '010_z005p000'

sim = 8

quantities = ['PartType0/Coordinates','PartType1/Coordinates','PartType4/Coordinates']
# quantities = ['PartType4/Coordinates']



for q in quantities:

    coords = E.read_array('SNAP', f'/cosma/home/dp004/dc-wilk2/FLARES/FLARES-1/G-EAGLE_{sim:02d}/data', tag, q, noH=True)
    print(coords.shape)
    coords_rec = coords - np.median(coords[:,:], axis=0)
    size = 20

    for i in range(3):
        print(np.min(coords_rec[:,i]), np.median(coords_rec[:,i]), np.max(coords_rec[:,i]))

    s = (np.fabs(coords_rec[:,0])<size/2)&(np.fabs(coords_rec[:,1])<size/2)&(np.fabs(coords_rec[:,2])<size/2)

    print(len(coords_rec[s]))

    np.save(f'data/Central_{q.replace("/","_")}_{sim:02d}_{tag}.npy', coords_rec[s])
