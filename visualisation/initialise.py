

import eagle_IO.eagle_IO as E # -*- coding: utf-8 -*-
import numpy as np
from sphviewer.tools import QuickView, cmaps
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

Sim = '/cosma/home/dp004/dc-wilk2/FLARES/FLARES-1/G-EAGLE_06/data'

tag = '010_z005p000'

data = 'PartType0/Coordinates'

coords = E.read_array('SNAP', Sim, tag, data, noH=True)

print(coords.shape)

for i in range(3):
    print(np.min(coords[:,i]), np.max(coords[:,i]))
