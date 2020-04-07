import matplotlib
matplotlib.use('Agg')
import numpy as np
from sphviewer.tools import cmaps
import matplotlib.pyplot as plt


labels = False

tag = '009_z006p000'

print(tag)

deltas = np.array([0.969639,0.918132,0.851838,0.849271,0.845644,0.842128,0.841291,0.83945,0.838891,0.832753,0.830465,0.829349,0.827842,0.824159,0.821425,0.820476,0.616236,0.616012,0.430745,0.430689,0.266515,0.266571,0.121315,0.121147,-0.007368,-0.007424,-0.121207,-0.121319,-0.222044,-0.222156,-0.311441,-0.311329,-0.066017,-0.066185,-0.00748,-0.007424,0.055076,0.054909,-0.47874,-0.433818])


x = np.linspace(-1,1,5000)
y = np.linspace(-1,1,5000)
xv, yv = np.meshgrid(x, y)
depth = np.sqrt(1.064**2 - (xv**2 + yv**2))

fig, axes = plt.subplots(5, 8, figsize=(16,10), dpi = 200)

fig.subplots_adjust(left=0.0, bottom=0.0, top=1.0, right=1.0, wspace=0.0, hspace=0.0)


sims = range(40)

sorted = np.argsort(deltas)
print(sorted)

for ax, sim in zip(axes.flatten(), sorted):

    print(sim)
    print(deltas[sim])

    try:
        img = np.load(f'data/DM_G-EAGLE_{sim:02d}_{tag}.npy')
        ax.imshow(np.log10(img/depth), cmap=cmaps.twilight(), origin='lower', vmin = 3.5, vmax = 6.5)
        if labels: ax.text(0.5,0.5,rf'$\delta={deltas[sim]:.4f}$', fontsize='20', color='1.0', alpha=0.5, transform=ax.transAxes,horizontalalignment='center',verticalalignment='center')
    except:
        print('failed')
        if labels: ax.text(0.5,0.5,rf'$\delta={deltas[sim]:.4f}$', fontsize='20', color='k', alpha=0.5, transform=ax.transAxes,horizontalalignment='center',verticalalignment='center')

    ax.axis('off')

fig.savefig('figs/DM_multiple_mean.png')
