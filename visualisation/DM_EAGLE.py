import matplotlib
matplotlib.use('Agg')
import numpy as np
from sphviewer.tools import cmaps
import matplotlib.pyplot as plt



img = np.load(f'data/DM_EAGLE.npy')

fig = plt.figure(1,figsize=(10,10), dpi=500)
plt.axis('off')
plt.imshow(np.log10(img), cmap=cmaps.twilight(), origin='lower', vmin = 3.5, vmax = 6.5)

plt.savefig(f'figs/DM_EAGLE.png')
fig.clf()
