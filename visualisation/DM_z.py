import matplotlib
matplotlib.use('Agg')
import numpy as np
from sphviewer.tools import cmaps
import matplotlib.pyplot as plt

tags = []
for i in range(11):
    z = 15 - i
    tags.append(f'{i:03d}_z{z:03d}p000')

print(tags)


sim = 5

fig, axes = plt.subplots(1, 11, figsize=(11,1), dpi = 500)

fig.subplots_adjust(left=0.0, bottom=0.0, top=1.0, right=1.0, wspace=0.0, hspace=0.0)

for ax, tag in zip(axes.flatten(), tags):
    img = np.load(f'data/DM_G-EAGLE_{sim:02d}_{tag}.npy')
    ax.imshow(np.log10(img), cmap=cmaps.twilight(), origin='lower', vmin = 2, vmax = 7.0)
    # ax.text(0.5,0.5,rf'$\delta={deltas[sim]:.4f}$', fontsize='20', color='1.0', alpha=0.5, transform=ax.transAxes,horizontalalignment='center',verticalalignment='center')


for ax in axes.flatten():
    ax.axis('off')

fig.savefig('figs/DM_z.png')
