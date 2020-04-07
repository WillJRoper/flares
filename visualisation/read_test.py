

import eagle_IO.eagle_IO as E # -*- coding: utf-8 -*-
import numpy as np


Sim = '/cosma/home/dp004/dc-wilk2/FLARES/FLARES-1/G-EAGLE_06/data'

tag = '000_z015p000'

data = 'PartType0/Coordinates'

coords = E.read_array('SNAP', Sim, tag, data)

print(coords.shape)


for i in range(3):
    print(np.min(coords[:,i]), np.max(coords[:,i]))


from sphviewer.tools import QuickView, cmaps

qv = QuickView(coords, r='infinity', plot=False)

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

fig = plt.figure(1, figsize=(7,7))

plt.imshow(qv.get_image(), extent=qv.get_extent(), origin='lower', cmap=cmaps.twilight(), vmin=-0.5)

fig.savefig('gas.png')
