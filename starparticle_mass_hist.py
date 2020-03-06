#!/cosma/home/dp004/dc-rope1/.conda/envs/flares-env/bin/python
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.colors import LogNorm
import eagle_IO as E
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

star_ms_dict = {}
for reg in regions:

    print(reg)

    path = '/cosma7/data/dp004/dc-love2/data/G-EAGLE/geagle_' + reg + '/data/'

    star_ms_dict = E.read_array('PARTDATA', path, snap, 'PartType4/Mass', noH=True)

star_ms = np.concatenate(list(star_ms_dict))

H, bins = np.histogram(star_ms, np.logspace(6, 7.5, 100))

bin_cents = bins[1:] - ((bins[1] - bins[0]) / 2)

fig = plt.figure()
ax = fig.add_subplot(111)

ax.plot(bin_cents, H)
ax.set_ylabel('$N$')
ax.set_xlabel('$M_{*}/M_{\odot}$')

fig.savefig('plots/star_particle_mass_hist.png')
