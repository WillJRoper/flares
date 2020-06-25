#!/cosma/home/dp004/dc-rope1/.conda/envs/flares-env/bin/python
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.colors import LogNorm
import eagle_IO.eagle_IO as E
import seaborn as sns
import pickle
import itertools
matplotlib.use('Agg')

sns.set_style('whitegrid')

regions = []
for reg in range(0, 40):

    if reg < 10:
        regions.append('0' + str(reg))
    else:
        regions.append(str(reg))

snaps = ['000_z015p000', '001_z014p000', '002_z013p000', '003_z012p000', '004_z011p000', '005_z010p000',
         '006_z009p000', '007_z008p000', '008_z007p000', '009_z006p000', '010_z005p000', '011_z004p770']

star_ms = []
for reg in regions:
    for snap in snaps:
        print(reg, snap)

        path = '/cosma/home/dp004/dc-rope1/FLARES/FLARES-1/G-EAGLE_' + reg + '/data'

        star_ms.extend(E.read_array('SNAP', path, snap, 'PartType4/Mass', numThreads=8))

star_ms = np.array(star_ms)
H, bins = np.histogram(star_ms / 0.6777, np.logspace(5, 12, 200))

bin_cents = bins[1:] - ((bins[1] - bins[0]) / 2)

fig = plt.figure()
ax = fig.add_subplot(111)

ax.bar(bin_cents, H)
ax.set_ylabel('$N$')
ax.set_xlabel('$M_{*}/M_{\odot}$')

fig.savefig('plots/star_particle_mass_hist.png')
