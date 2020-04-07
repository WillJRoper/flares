import matplotlib
matplotlib.use('Agg')
import numpy as np
from sphviewer.tools import cmaps
import matplotlib.pyplot as plt

tag = '009_z006p000'


sims = range(40)

# sims = [0]

x = np.linspace(-1,1,5000)
y = np.linspace(-1,1,5000)
xv, yv = np.meshgrid(x, y)
depth = np.sqrt(1.064**2 - (xv**2 + yv**2))


for sim in sims:

    try:

        print(sim)

        img = np.load(f'data/DM_G-EAGLE_{sim:02d}_{tag}.npy')
        img /= depth

        fig = plt.figure(1,figsize=(10,10), dpi=200)
        plt.axis('off')
        plt.imshow(np.log10(img), cmap=cmaps.twilight(), origin='lower', vmin = 3.5, vmax = 6.5)
        plt.savefig(f'figs/DM/G-EAGLE_{sim:02d}_{tag}_mean.png')
        fig.clf()

    except:

        print(sim, 'failed')
