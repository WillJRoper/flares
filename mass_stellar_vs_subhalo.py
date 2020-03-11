#!/cosma/home/dp004/dc-rope1/.conda/envs/flares-env/bin/python
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.colors import LogNorm
import eagle_IO as E
import seaborn as sns
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
for reg in regions:

    print(reg)

    path = '/cosma7/data/dp004/dc-love2/data/G-EAGLE/geagle_' + reg + '/data/'

    submass_dict[reg] = E.read_array('SUBFIND', path, snap, 'Subhalo/Mass', noH=True, numThreads=8)

submass = np.concatenate(list(submass_dict.values())) * 10**10

fig = plt.figure(figsize=(16, 4))
ax1 = fig.add_subplot(141)
ax2 = fig.add_subplot(142)
ax3 = fig.add_subplot(143)
ax4 = fig.add_subplot(144)

starmass_dict = {}
cbars = {}
for part, ax, title in zip([0, 1, 4, 5], [ax1, ax2, ax3, ax4], ['Gas', 'DM', 'Stars', 'BH']):

    for reg in regions:

        print(reg)

        path = '/cosma7/data/dp004/dc-love2/data/G-EAGLE/geagle_' + reg + '/data/'

        starmass_dict[reg] = E.read_array('SUBFIND', path, snap,
                                          'Subhalo/ApertureMeasurements/Mass/010kpc', noH=True, numThreads=8)[:, part]

    starmass = np.concatenate(list(starmass_dict.values())) * 10**10

    submass_plt = submass[starmass > 0]
    starmass_plt = starmass[starmass > 0]
    starmass_plt = starmass_plt[submass_plt > 0]
    submass_plt = submass_plt[submass_plt > 0]

    ax.plot(np.linspace(submass_plt.min(), submass_plt.max(), 100),
            np.linspace(submass_plt.min(), submass_plt.max(), 100),
            linestyle='--', zorder=0)

    cbars[part] = ax.hexbin(submass_plt, starmass_plt, gridsize=100, mincnt=1, xscale='log', norm=LogNorm(1, 10**4.5),
                     yscale='log', linewidths=0.2, cmap='viridis', zorder=1)

    ax.set_xscale('log')
    ax.set_yscale('log')

    ax.set_title(title)

    ax.set_ylim(10**6, 10**13)

ax1.set_xlabel('$M_{\mathrm{tot}}/M_\odot$')
ax2.set_xlabel('$M_{\mathrm{tot}}/M_\odot$')
ax3.set_xlabel('$M_{\mathrm{tot}}/M_\odot$')
ax4.set_xlabel('$M_{\mathrm{tot}}/M_\odot$')
ax1.set_ylabel('$M/M_\odot$')

cax = fig.colorbar(cbars[0], ax=ax4)
cax.ax.set_ylabel(r'$N$')

fig.savefig('plots/starvssubhalo_mass_' + snap + '.png', bbox_inches='tight')

plt.close(fig)

