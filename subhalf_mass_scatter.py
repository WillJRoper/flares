#!/cosma/home/dp004/dc-rope1/.conda/envs/flares-env/bin/python
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import astropy.units as u
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
# x_tags = ['Subhalo/StarFormationRate', 'Subhalo/Stars/Mass', 'Subhalo/Stars/Metallicity', 'Stars/SmoothedMetallicity',
#           'Subhalo/VmaxRadius', 'Subhalo/SF/Mass', 'Subhalo/Mass']
x_tags = ['Subhalo/StarFormationRate', 'Subhalo/Stars/Mass', 'Subhalo/SF/Mass', 'Subhalo/Mass', 'Subhalo/NSF/Mass']
xlabels = ['$\mathrm{SRF}/[M_{\odot}/\mathrm{yr}]$ ', '$M_{*}/M_\odot$', '$M_{\mathrm{SF}}/M_\odot$',
           '$M_{\mathrm{sub}}/M_\odot$', '$M_{\mathrm{NSF}}/M_\odot$']
logs = [True, True, True, True, True, True, True]

half_mass_rads_dict = {}
for reg in regions:

    print(reg)

    path = '/cosma7/data/dp004/dc-love2/data/G-EAGLE/geagle_' + reg + '/data/'

    half_mass_rads_dict[reg] = E.read_array('SUBFIND', path, snap, 'Subhalo/HalfMassRad', noH=True)[:, 4]

half_mass_rads = np.concatenate(list(half_mass_rads_dict.values()))

save_dict = {}

for xtag, tolog, xlab in zip(x_tags, logs, xlabels):

    xaxis_dict = {}
    for reg in regions:
        print(reg)
        path = '/cosma7/data/dp004/dc-love2/data/G-EAGLE/geagle_' + reg + '/data/'

        try:
            xaxis_dict[reg] = E.read_array('SUBFIND', path, snap, xtag, noH=True)
        except KeyError:
            xaxis_dict[reg] = E.read_array('SUBFIND', path, snap, xtag, noH=False)

    xs = np.concatenate(list(xaxis_dict.values()))
    if xtag[-4:] == 'Mass':
        xs *= 10**10
    save_dict[xtag] = xs
    fig = plt.figure()
    ax = fig.add_subplot(111)

    xs = xs[half_mass_rads > 0]
    half_mass_rads_plt = half_mass_rads[half_mass_rads > 0]
    half_mass_rads_plt = half_mass_rads_plt[xs > 0]
    xs = xs[xs > 0]

    if tolog:
        cbar = ax.hexbin(xs, half_mass_rads_plt, gridsize=100, mincnt=1, xscale='log', norm=LogNorm(),
                         yscale='log', linewidths=0.2, cmap='viridis')
    else:
        cbar = ax.hexbin(xs, half_mass_rads_plt, gridsize=100, mincnt=1, norm=LogNorm(), linewidths=0.2, cmap='viridis')

    ax.set_xlabel(xlab)
    ax.set_ylabel('$R_{1/2}/$ckpc')

    cax = fig.colorbar(cbar, ax=ax)
    cax.ax.set_ylabel(r'$N$')

    fig.savefig('plots/' + 'HalfMassRadiusCorrleations_' + xtag.replace('/', '-') + '_' + snap + '.png',
                bbox_inches='tight')

    plt.close(fig)

save_dict['Subhalo/HalfMassRad'] = half_mass_rads

with open('halfmassrad_data.pck', 'wb') as pfile1:
    pickle.dump(save_dict, pfile1)


