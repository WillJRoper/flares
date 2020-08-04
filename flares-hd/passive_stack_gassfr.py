#!/cosma/home/dp004/dc-rope1/.conda/envs/flares-env/bin/python
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import matplotlib
import astropy.units as u
import astropy.constants as cons
from astropy.cosmology import Planck13 as cosmo
from matplotlib.colors import LogNorm
import eagle_IO.eagle_IO as E
import seaborn as sns
import h5py
matplotlib.use('Agg')


lim = 75 / 1000
soft = 0.001802390 / 0.6777 / 4
scale = 10 / 1000

# Define resolution
res = int(np.floor(2 * lim / soft))

regions = []
for reg in range(0, 40):

    if reg < 10:
        regions.append('0' + str(reg))
    else:
        regions.append(str(reg))

snaps = ['000_z015p000', '001_z014p000', '002_z013p000', '003_z012p000', '004_z011p000', '005_z010p000',
         '006_z009p000', '007_z008p000', '008_z007p000', '009_z006p000', '010_z005p000', '011_z004p770']

# Define galaxy thresholds
ssfr_thresh = 0.1

count = 0

star_img = np.zeros((res, res))
gas_img = np.zeros((res, res))

for reg in regions:

    for snap in snaps:

        print(reg, snap)

        z_str = snap.split('z')[1].split('p')
        z = float(z_str[0] + '.' + z_str[1])

        path = '/cosma/home/dp004/dc-rope1/FLARES/FLARES-1/G-EAGLE_' + reg + '/data'

        try:
            app_mass = E.read_array('SUBFIND', path, snap, 'Subhalo/ApertureMeasurements/Mass/030kpc',
                                    numThreads=8) * 10**10 / 0.6777
            sfrs = E.read_array('SUBFIND', path, snap, 'Subhalo/ApertureMeasurements/SFR/030kpc',
                                numThreads=8) * 10**10 / 0.6777
            gal_cops = E.read_array('SUBFIND', path, snap, 'Subhalo/CentreOfPotential',
                                  numThreads=8) / 0.6777
        except ValueError:
            continue
        except OSError:
            continue

        ssfrs = sfrs / app_mass[:, 4]

        okinds = np.logical_and(app_mass[:, 4] > 1e9, np.logical_and(ssfrs <= ssfr_thresh, app_mass[:, 1] > 0))
        app_mass = app_mass[okinds]
        sfrs = sfrs[okinds]
        cops = gal_cops[okinds]

        print("There are", len(cops), "passive galaxies in", reg, snap)

        if len(cops) > 0:
            try:

                star_poss = E.read_array('PARTDATA', path, snap, 'PartType4/Coordinates', numThreads=8) / 0.6777
                gas_poss = E.read_array('PARTDATA', path, snap, 'PartType0/Coordinates', numThreads=8) / 0.6777
                stellar_masses = E.read_array('PARTDATA', path, snap, 'PartType4/Mass', numThreads=8) * 10 ** 10 / 0.6777
                gas_masses = E.read_array('PARTDATA', path, snap, 'PartType0/Mass', numThreads=8) * 10 ** 10 / 0.6777

            except ValueError:
                continue
            except OSError:
                continue

        for cop in cops:

            # Get only stars within the aperture
            star_okinds = np.logical_and(np.abs(star_poss[:, 0] - cop[0]) < lim,
                                         np.logical_and(np.abs(star_poss[:, 1] - cop[1]) < lim,
                                                        np.abs(star_poss[:, 2] - cop[2]) < lim))
            gas_okinds = np.logical_and(np.abs(gas_poss[:, 0] - cop[0]) < lim,
                                        np.logical_and(np.abs(gas_poss[:, 1] - cop[1]) < lim,
                                                       np.abs(gas_poss[:, 2] - cop[2]) < lim))
            this_star_poss = star_poss[star_okinds, :] - cop
            this_gas_poss = gas_poss[gas_okinds, :] - cop
            this_star_ms = stellar_masses[star_okinds]
            this_gas_ms = gas_masses[gas_okinds]

            # Histogram positions into images
            Hstar, _, _ = np.histogram2d(this_star_poss[:, 0], this_star_poss[:, 1], bins=res,
                                         range=((-lim, lim), (-lim, lim)), weights=this_star_ms)
            Hgas, _, _ = np.histogram2d(this_gas_poss[:, 0], this_gas_poss[:, 1], bins=res,
                                        range=((-lim, lim), (-lim, lim)), weights=this_gas_ms)

            star_img += Hstar
            gas_img += Hgas
            count += 1

print(count, "Passive Galaxies")

fig = plt.figure()
ax1 = fig.add_subplot(131)
ax2 = fig.add_subplot(132)
ax3 = fig.add_subplot(133)

ax1.imshow(np.zeros_like(star_img), cmap='magma', extent=(-lim, lim, -lim, lim))
ax2.imshow(np.zeros_like(star_img), cmap='magma', extent=(-lim, lim, -lim, lim))
ax3.imshow(np.zeros_like(star_img), cmap='magma', extent=(-lim, lim, -lim, lim))

im1 = ax1.imshow(np.log10(star_img), cmap='magma', extent=(-lim, lim, -lim, lim))
im2 = ax2.imshow(np.log10(gas_img), cmap='magma', extent=(-lim, lim, -lim, lim))
ax3.imshow(np.log10(gas_img), cmap='magma', extent=(-lim, lim, -lim, lim), alpha=0.8)
ax3.imshow(np.log10(star_img), cmap='Greys_r', extent=(-lim, lim, -lim, lim), alpha=0.5)

app1 = plt.Circle((0., 0.), 0.03, facecolor='none', edgecolor='r', linestyle='-')
app2 = plt.Circle((0., 0.), 0.03, facecolor='none', edgecolor='r', linestyle='-')
app3 = plt.Circle((0., 0.), 0.03, facecolor='none', edgecolor='r', linestyle='-')

ax1.add_artist(app1)
ax2.add_artist(app2)
ax3.add_artist(app3)

ax1.set_title("Stellar")
ax2.set_title("Gas")
ax3.set_title("Gas + Stellar")

# Remove ticks
ax1.tick_params(axis='both', left=False, top=False, right=False, bottom=False, labelleft=False,
                labeltop=False, labelright=False, labelbottom=False)
ax2.tick_params(axis='both', left=False, top=False, right=False, bottom=False, labelleft=False,
                labeltop=False, labelright=False, labelbottom=False)
ax3.tick_params(axis='both', left=False, top=False, right=False, bottom=False, labelleft=False,
                labeltop=False, labelright=False, labelbottom=False)

# Draw scale line
right_side = lim - (lim * 0.1)
vert = lim - (lim * 0.175)
lab_vert = vert + (lim * 0.1) * 5 / 8
lab_horz = right_side - scale / 2
ax1.plot([right_side - scale, right_side], [vert, vert], color='w', linewidth=0.5)
ax2.plot([right_side - scale, right_side], [vert, vert], color='w', linewidth=0.5)
ax3.plot([right_side - scale, right_side], [vert, vert], color='w', linewidth=0.5)

# Label scale
ax1.text(lab_horz, lab_vert, str(int(scale*1e3)) + ' ckpc', horizontalalignment='center',
         fontsize=4, color='w')
ax2.text(lab_horz, lab_vert, str(int(scale*1e3)) + ' ckpc', horizontalalignment='center',
         fontsize=4, color='w')
ax3.text(lab_horz, lab_vert, str(int(scale*1e3)) + ' ckpc', horizontalalignment='center',
         fontsize=4, color='w')

# Add colorbars
cax1 = inset_axes(ax1, width="50%", height="3%", loc='lower left')
cax2 = inset_axes(ax2, width="50%", height="3%", loc='lower left')
cbar1 = fig.colorbar(im1, cax=cax1, orientation="horizontal")
cbar2 = fig.colorbar(im2, cax=cax2, orientation="horizontal")

# Label colorbars
cbar1.ax.set_xlabel(r'$\log_{10}(M_{\star}/M_{\odot})$', fontsize=3, color='w', labelpad=1.0)
cbar1.ax.xaxis.set_label_position('top')
cbar1.outline.set_edgecolor('w')
cbar1.outline.set_linewidth(0.05)
cbar1.ax.tick_params(axis='x', length=1, width=0.2, pad=0.01, labelsize=2, color='w', labelcolor='w')
cbar2.ax.set_xlabel(r'$\log_{10}(M_{\mathrm{gas}}/M_{\odot})$', fontsize=3, color='w',
                    labelpad=1.0)
cbar2.ax.xaxis.set_label_position('top')
cbar2.outline.set_edgecolor('w')
cbar2.outline.set_linewidth(0.05)
cbar2.ax.tick_params(axis='x', length=1, width=0.2, pad=0.01, labelsize=2, color='w', labelcolor='w')

fig.savefig("plots/passive_stack_gassfr.png", bbox_inches='tight', dpi=300)
