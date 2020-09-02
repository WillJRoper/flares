#!/cosma/home/dp004/dc-rope1/.conda/envs/flares-env/bin/python
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import astropy.units as u
from matplotlib.colors import LogNorm
import matplotlib.gridspec as gridspec
from scipy.stats import binned_statistic
import eagle_IO.eagle_IO as E
import seaborn as sns
import pickle
import itertools
matplotlib.use('Agg')

sns.set_style('whitegrid')

regions = []
for reg in range(0, 40):

    if reg < 10:
        regions.append('000' + str(reg))
    else:
        regions.append('00' + str(reg))

snaps = ['003_z012p000', '004_z011p000', '005_z010p000',
         '006_z009p000', '007_z008p000', '008_z007p000',
         '009_z006p000', '010_z005p000', '011_z004p770']

for reg in regions:

    for snap in snaps:

        print(reg, snap)

        path = '/cosma7/data/dp004/dc-love2/data/G-EAGLE/geagle_' + reg + '/data/'
        # path = '/cosma/home/dp004/dc-rope1/FLARES/FLARES-1/FLARES_00_medFBlim/data/'
        try:
            part_sum = 0
            for part_type in [0, 1, 4, 5]:
                pids = E.read_array('SNAP', path, snap, 'PartType' + str(part_type) + '/ParticleIDs', numThreads=8)
                print(reg, snap, pids.size, "particles of type", part_type)
                part_sum += pids.size
            print(reg, snap, part_sum, "particles in total:", int(part_sum**(1/3)))
        except OSError:
            continue