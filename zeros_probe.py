#!/cosma/home/dp004/dc-rope1/.conda/envs/flares-env/bin/python
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import astropy.units as u
import astropy.constants as const
from matplotlib.colors import LogNorm
import eagle_IO as E
import seaborn as sns
from flares import flares
matplotlib.use('Agg')

sns.set_style('whitegrid')

regions = []
for reg in range(0, 40):

    if reg < 10:
        regions.append('000' + str(reg))
    else:
        regions.append('00' + str(reg))

snap = '010_z005p000'

submass_dict = {}
starmass_dict = {}
for reg in regions:

    print(reg)

    path = '/cosma7/data/dp004/dc-love2/data/G-EAGLE/geagle_' + reg + '/data/'

    submass_dict[reg] = E.read_array('SUBFIND', path, snap, 'Subhalo/Mass', noH=True, numThreads=8)
    starmass_dict[reg] = E.read_array('SUBFIND', path, snap,
                                      'Subhalo/ApertureMeasurements/Mass/030kpc', noH=True, numThreads=8)

submass = np.concatenate(list(submass_dict.values())) * 10**10
starmass = np.concatenate(list(starmass_dict.values())) * 10**10

fig = plt.figure(figsize=(17, 4))
ax1 = fig.add_subplot(141)
ax2 = fig.add_subplot(142)
ax3 = fig.add_subplot(143)

starmass_dict = {}
cbars = {}
labels = ['$M_{\mathrm{gas}}/M_\odot + 1$', '$M_{\mathrm{DM}}/M_\odot + 1$', '$M_{*}/M_\odot + 1$']
for drop, parts, ax, title in zip([0, 1, 4], [(1, 4), (0, 4), (0, 1)], [ax1, ax2, ax3],
                                  ['$M_{\mathrm{gas}}=0$', '$M_{\mathrm{DM}}=0$', '$M_{*}=0$', 'BH']):

    x_plt = starmass[:, parts[0]][np.where(starmass[:, drop] == 0)] + 1
    y_plt = starmass[:, parts[1]][np.where(starmass[:, drop] == 0)] + 1

    cbars[drop] = ax.hexbin(x_plt, y_plt, gridsize=100, mincnt=1, xscale='log', norm=LogNorm(),
                            yscale='log', linewidths=0.2, cmap='viridis', zorder=1)

    ax.set_xscale('log')
    ax.set_yscale('log')

    ax.set_title(title)

    # ax.set_ylim(10**6, 10**13)

    ax.set_xlabel(labels[parts[0]])
    ax.set_ylabel(labels[parts[0]])

cax = fig.colorbar(cbars[0], ax=ax3)
cax.ax.set_ylabel(r'$N$')

fig.savefig('plots/zeromassprobe_' + snap + '.png', bbox_inches='tight')

plt.close(fig)
