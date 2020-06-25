#!/cosma/home/dp004/dc-rope1/.conda/envs/flares-env/bin/python
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.colors import LogNorm
import matplotlib.gridspec as gridspec
import eagle_IO.eagle_IO as E
import seaborn as sns
import pickle
import itertools
matplotlib.use('Agg')

sns.set_style('whitegrid')

regions = []
for reg in range(0, 2):

    if reg < 10:
        regions.append('0' + str(reg))
    else:
        regions.append(str(reg))

snaps = ['000_z015p000', '001_z014p000', '002_z013p000', '003_z012p000', '004_z011p000', '005_z010p000',
         '006_z009p000', '007_z008p000', '008_z007p000', '009_z006p000', '010_z005p000', '011_z004p770']

star_ms = []
group_star_ms = []
for reg in regions:
    for snap in snaps:
        print(reg, snap)

        path = '/cosma/home/dp004/dc-rope1/FLARES/FLARES-1/G-EAGLE_' + reg + '/data'

        star_ms.extend(E.read_array('SNAP', path, snap, 'PartType4/Mass', numThreads=8) * 10**10)
        group_star_ms.extend(E.read_array('PARTDATA', path, snap, 'PartType4/Mass', numThreads=8) * 10 ** 10)

star_ms = np.array(star_ms)
group_star_ms = np.array(group_star_ms)
print(group_star_ms.size / star_ms.size)
H, bins = np.histogram(star_ms / 0.6777, np.logspace(5.5, 10, 100))
H2, bins2 = np.histogram(group_star_ms / 0.6777, np.logspace(5.5, 10, 100))

bin_cents = bins[1:] - ((bins[1] - bins[0]) / 2)

fig = plt.figure()
gs = gridspec.GridSpec(nrows=2, ncols=1)
gs.update(wspace=0.0, hspace=0.0)
ax1 = fig.add_subplot(gs[0])
ax2 = fig.add_subplot(gs[1])

ax1.plot(bin_cents, H, label='SNAP')
ax1.plot(bin_cents, H2, label='PARTDATA', linestyle='--')
ax2.plot(bin_cents, H-H2, label='PARTDATA', linestyle='--')
ax1.set_ylabel('$N$')
ax2.set_ylabel('$N_{\mathrm{SNAP}-N_\mathrm{PARTDATA}}$')
ax2.set_xlabel('$M_{*}/M_{\odot}$')

ax1.set_yscale("log")
ax1.set_xscale("log")
ax2.set_xscale("log")

handles, labels = ax1.get_legend_handles_labels()
ax1.legend(handles, labels, loc='best')

fig.savefig('plots/star_particle_mass_hist.png')
