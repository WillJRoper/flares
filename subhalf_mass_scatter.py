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
x_tags = ['Subhalo/StarFormationRate', 'Subhalo/Stars/Mass', 'Subhalo/Stars/Metallicity', 'Stars/SmoothedMetallicity',
          'Subhalo/VmaxRadius', 'Subhalo/SF/Mass', 'Subhalo/Mass']
logs = [False, True, False, False, True, True, True]

half_mass_rads_dict = {}
for reg in regions:

    print(reg)

    path = '/cosma7/data/dp004/dc-love2/data/G-EAGLE/geagle_' + reg + '/data/'

    half_mass_rads_dict[reg] = E.read_array('SUBFIND', path, snap, 'Subhalo/HalfMassRad', noH=True)[:, 4]

half_mass_rads = np.concatenate(list(half_mass_rads_dict.values()))

save_dict = {}

for xtag, tolog in zip(x_tags, logs):

    xaxis_dict = {}
    for reg in regions:
        print(reg)
        path = '/cosma7/data/dp004/dc-love2/data/G-EAGLE/geagle_' + reg + '/data/'

        try:
            xaxis_dict[reg] = E.read_array('SUBFIND', path, snap, xtag, noH=True)
        except KeyError:
            xaxis_dict[reg] = E.read_array('SUBFIND', path, snap, xtag, noH=False)

    xs = np.concatenate(list(xaxis_dict.values()))
    save_dict[xtag] = xs
    fig = plt.figure()
    ax = fig.add_subplot(111)

    xs = xs[half_mass_rads > 0]
    half_mass_rads_plt = half_mass_rads[half_mass_rads > 0]
    half_mass_rads_plt = half_mass_rads_plt[xs > 0]
    xs = xs[xs > 0]

    if tolog:
        ax.hexbin(xs, half_mass_rads_plt, gridsize=50, mincnt=1, xscale='log', norm=LogNorm(),
                  yscale='log', linewidths=0.2, cmap='viridis')
    else:
        ax.hexbin(xs, half_mass_rads_plt, gridsize=50, mincnt=1, norm=LogNorm(), linewidths=0.2, cmap='viridis')

    ax.set_xlabel(xtag)
    ax.set_ylabel('$R_{1/2}/$ckpc')

    fig.savefig('plots/' + 'HalfMassRadiusCorrleations_' + xtag.replace('/', '-') + '_' + snap + '.png',
                bbox_inches='tight')

    plt.close(fig)

save_dict['Subhalo/HalfMassRad'] = half_mass_rads

with open('halfmassrad_data.pck', 'wb') as pfile1:
    pickle.dump(save_dict, pfile1)


