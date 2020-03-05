#!/cosma/home/dp004/dc-rope1/.conda/envs/flares-env/bin/python
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.colors import LogNorm
import eagle_IO as E
import seaborn as sns
import itertools
matplotlib.use('Agg')

sns.set_style('whitegrid')

regions = []
for reg in range(0, 40):

    if reg < 10:
        regions.append('000' + str(reg))
    else:
        regions.append('00' + str(reg))

path = '/cosma7/data/dp004/dc-love2/data/G-EAGLE/geagle_0032/data/'
snap = '010_z005p000'

# Load the groups and arrays names
tags = np.genfromtxt('HDF5_tags.txt', dtype=str, delimiter='\n')
groups = ['FOF', 'FOF_PARTICLES', 'SNIP_FOF', 'SNIP_FOF_PARTICLES', 'SUBFIND', 'SUBFIND_GROUP', 'SUBFIND_IDS',
          'SNIP_SUBFIND', 'SNIP_SUBFIND_GROUP', 'SNIP_SUBFIND_IDS', 'SNIP', 'SNAP', 'PARTDATA', 'SNIP_PARTDATA']
combos = itertools.combinations(tags, 2)

for tagx, tagy in combos:

    existx = False
    existy = False

    for group in groups:

        try:
            xs = E.read_array(group, path, snap, tagx, noH=True)
            existx = True
            print('x', group)
        except KeyError:
            continue
        except ValueError:
            continue

    for group in groups:

        try:
            ys = E.read_array(group, path, snap, tagy, noH=True)
            existy = True
            print('y', group)
        except KeyError:
            continue
        except ValueError:
            continue

    if existy and existx:

        print(tagx, tagy)

        if xs.shape != ys.shape:
            continue
        elif len(xs.shape) != 1 or len(xs.shape) != 1:
            continue

        fig = plt.figure()
        ax = fig.add_subplot(111)

        if xs.max() / xs.min() > 100 and ys.max() / ys.min() > 100 and xs.min() > 0 and ys.min() > 0:
            ax.hexbin(xs, ys, gridsize=50, mincnt=1, xscale='log', norm=LogNorm(),
                      yscale='log', linewidths=0.2, cmap='viridis')
        elif xs.max() / xs.min() > 100 and ys.max() / ys.min() <= 100 and xs.min() > 0:
            ax.hexbin(xs, ys, gridsize=50, mincnt=1, xscale='log', norm=LogNorm(),
                      linewidths=0.2, cmap='viridis')
        elif xs.max() / xs.min() <= 100 and ys.max() / ys.min() > 100 and ys.min() > 0:
            ax.hexbin(xs, ys, gridsize=50, mincnt=1, yscale='log', norm=LogNorm(),
                      linewidths=0.2, cmap='viridis')
        else:
            ax.hexbin(xs, ys, gridsize=50, mincnt=1, norm=LogNorm(),
                      linewidths=0.2, cmap='viridis')

        ax.set_xlabel(tagx)
        ax.set_ylabel(tagy)

        fig.savefig('plots/' + tagy.replace('/', '_') + 'vs' + tagx.replace('/', '_') + '_' + snap + '.png',
                    bbox_inches='tight')

        plt.close(fig)

    else:
        continue
