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
        regions.append('000' + str(reg))
    else:
        regions.append('00' + str(reg))

snap = '010_z005p000'

star_ms = []
for reg in regions:

    print(reg)

    path = '/cosma7/data/dp004/dc-love2/data/G-EAGLE/geagle_' + reg + '/data/'

    star_ms.extend(E.read_array('SNAP', path, snap, 'PartType4/Mass'))

star_ms = np.array(star_ms)
H, bins = np.histogram(star_ms / 0.6777, np.logspace(5, 10, 200))

bin_cents = bins[1:] - ((bins[1] - bins[0]) / 2)

fig = plt.figure()
ax = fig.add_subplot(111)

ax.bar(bin_cents, H)
ax.set_ylabel('$N$')
ax.set_xlabel('$M_{*}/M_{\odot}$')

fig.savefig('plots/star_particle_mass_hist.png')
