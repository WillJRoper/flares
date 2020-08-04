#!/cosma/home/dp004/dc-rope1/.conda/envs/flares-env/bin/python
"""
Creates the plot of metallicity against birth density, with
the background coloured by f_th.
"""

import matplotlib.pyplot as plt
import numpy as np
import pickle
import eagle_IO.eagle_IO as E
import matplotlib.gridspec as gridspec
from unyt import mh, cm, Gyr, g, Msun, Mpc
from matplotlib.colors import LogNorm


def get_data(load=False):

    regions = []
    for reg in range(0, 40):
        if reg < 10:
            regions.append('0' + str(reg))
        else:
            regions.append(str(reg))

    # Define snapshots
    snaps = ['003_z012p000', '004_z011p000', '005_z010p000',
             '006_z009p000', '007_z008p000', '008_z007p000',
             '009_z006p000', '010_z005p000', '011_z004p770']

    if load:

        with open('metvsbd.pck', 'rb') as pfile1:
            save_dict = pickle.load(pfile1)
        gas_met_dict = save_dict['met']
        gas_bd_dict = save_dict['bd']

    else:

        gas_met_dict = {}
        gas_bd_dict = {}

        for snap in snaps:

            gas_met_dict[snap] = {}
            gas_bd_dict[snap] = {}

        for reg in regions:

            for snap in snaps:

                path = '/cosma/home/dp004/dc-rope1/FLARES/FLARES-1/G-EAGLE_' + reg + '/data'

                # Get halo IDs and halo data
                try:
                    gal_den = E.read_array('PARTDATA', path, snap, 'PartType0/Density', noH=True,
                                            physicalUnits=True, numThreads=8)
                    gal_met = E.read_array('PARTDATA', path, snap, 'PartType0/Metallicity', noH=True,
                                           physicalUnits=True, numThreads=8)
                except ValueError:
                    continue
                except OSError:
                    continue
                except KeyError:
                    continue

                gas_bd_dict[snap][reg] = gal_den
                gas_met_dict[snap][reg] = gal_met

        with open('metvsbd.pck', 'wb') as pfile1:
            pickle.dump({'bd': gas_bd_dict, 'met': gas_met_dict}, pfile1)

    return gas_bd_dict, gas_met_dict


snaps = ['003_z012p000', '004_z011p000', '005_z010p000',
         '006_z009p000', '007_z008p000', '008_z007p000',
         '009_z006p000', '010_z005p000', '011_z004p770']

gas_bd_dict, gas_met_dict = get_data(load=False)

# plt.style.use("mnras.mplstyle")

# EAGLE parameters
parameters = {"f_th,min": 0.3,
              "f_th,max": 3,
              "n_Z": 1.0,
              "n_n": 1.0,
              "Z_pivot": 0.1 * 0.012,
              "n_pivot": 0.67}

star_formation_parameters = {"threshold_Z0": 0.002,
                             "threshold_n0": 0.1,
                             "slope": -0.64}


number_of_bins = 128

# Constants; these could be put in the parameter file but are rarely changed.
birth_density_bins = np.logspace(-3, 6.8, number_of_bins)
metal_mass_fraction_bins = np.logspace(-5.9, 0, number_of_bins)

# Now need to make background grid of f_th.
birth_density_grid, metal_mass_fraction_grid = np.meshgrid(
    0.5 * (birth_density_bins[1:] + birth_density_bins[:-1]),
    0.5 * (metal_mass_fraction_bins[1:] + metal_mass_fraction_bins[:-1]))

f_th_grid = parameters["f_th,min"] + (parameters["f_th,max"] - parameters["f_th,min"]) / (
    1.0
    + (metal_mass_fraction_grid / parameters["Z_pivot"]) ** parameters["n_Z"]
    * (birth_density_grid / parameters["n_pivot"]) ** (-parameters["n_n"])
)

axlims_x = []
axlims_y = []

# Set up plot
fig = plt.figure(figsize=(18, 10))
gs = gridspec.GridSpec(3, 6)
gs.update(wspace=0.0, hspace=0.0)
ax1 = fig.add_subplot(gs[0, 0])
ax2 = fig.add_subplot(gs[0, 1])
ax3 = fig.add_subplot(gs[0, 2])
ax4 = fig.add_subplot(gs[1, 0])
ax5 = fig.add_subplot(gs[1, 1])
ax6 = fig.add_subplot(gs[1, 2])
ax7 = fig.add_subplot(gs[2, 0])
ax8 = fig.add_subplot(gs[2, 1])
ax9 = fig.add_subplot(gs[2, 2])

for ax, snap, (i, j) in zip([ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8, ax9], snaps,
                            [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2), (2, 0), (2, 1), (2, 2)]):

    print("Plotting snapshot:", snap)

    z_str = snap.split('z')[1].split('p')
    z = float(z_str[0] + '.' + z_str[1])

    ax.loglog()

    metal_mass_fractions = np.concatenate(list(gas_met_dict[snap].values()))
    gas_bd = (np.concatenate(list(gas_bd_dict[snap].values()))
                  * 10**10 * Msun / Mpc ** 3 / mh).to(1 / cm ** 3).value

    H, _, _ = np.histogram2d(gas_bd, metal_mass_fractions,
                             bins=[birth_density_bins, metal_mass_fraction_bins])

    ax.contour(birth_density_grid, metal_mass_fraction_grid, H.T, levels=6, cmap="magma")

    # Add line showing SF law
    sf_threshold_density = star_formation_parameters["threshold_n0"] * \
                           (metal_mass_fraction_bins
                            / star_formation_parameters["threshold_Z0"]) ** (star_formation_parameters["slope"])
    ax.plot(sf_threshold_density, metal_mass_fraction_bins, linestyle="dashed", label="SF threshold")

    ax.text(0.1, 0.9, f'$z={z}$', bbox=dict(boxstyle="round,pad=0.3", fc='w', ec="k", lw=1, alpha=0.8),
            transform=ax.transAxes, horizontalalignment='left', fontsize=8)

    axlims_x.extend(ax.get_xlim())
    axlims_y.extend(ax.get_ylim())

    # Label axes
    if i == 2:
        ax.set_xlabel("Gas Density [$n_H$ cm$^{-3}$]")
    if j == 0:
        ax.set_ylabel("Gas Metal Mass Fraction $Z$")

for ax in [ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8, ax9]:
    ax.set_xlim(10**-3, 10**6.8)
    ax.set_ylim(10**-5.9, 10**0)
    for spine in ax.spines.values():
        spine.set_edgecolor('k')

# Remove axis labels
ax1.tick_params(axis='x', top=False, bottom=False, labeltop=False, labelbottom=False)
ax2.tick_params(axis='both', left=False, top=False, right=False, bottom=False, labelleft=False, labeltop=False,
                labelright=False, labelbottom=False)
ax3.tick_params(axis='both', left=False, top=False, right=False, bottom=False, labelleft=False, labeltop=False,
                labelright=False, labelbottom=False)
ax4.tick_params(axis='x', top=False, bottom=False, labeltop=False, labelbottom=False)
ax5.tick_params(axis='both', left=False, top=False, right=False, bottom=False, labelleft=False, labeltop=False,
                labelright=False, labelbottom=False)
ax6.tick_params(axis='both', left=False, top=False, right=False, bottom=False, labelleft=False, labeltop=False,
                labelright=False, labelbottom=False)
ax8.tick_params(axis='y', left=False, right=False, labelleft=False, labelright=False)
ax9.tick_params(axis='y', left=False, right=False, labelleft=False, labelright=False)

legend = ax7.legend(markerfirst=True, loc="lower left", fontsize=8)
plt.setp(legend.get_texts())

try:
    fontsize = legend.get_texts()[0].get_fontsize()
except:
    fontsize = 6

ax9.text(0.975, 0.025, "\n".join([f"${k.replace('_', '_{') + '}'}$: ${v:.4g}$" for k, v in parameters.items()]),
         color="k", transform=ax9.transAxes, ha="right", va="bottom", fontsize=fontsize)

ax3.text(0.975, 0.975, "Contour lines \n linearly spaced", color="k", transform=ax3.transAxes, ha="right", va="top",
         fontsize=fontsize)

fig.savefig('plots/gas_density_metallicity_redshift.png', bbox_inches='tight')
