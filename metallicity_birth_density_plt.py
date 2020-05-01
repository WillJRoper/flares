#!/cosma/home/dp004/dc-rope1/.conda/envs/flares-env/bin/python
"""
Creates the plot of metallicity against birth density, with
the background coloured by f_E.
"""

import matplotlib.pyplot as plt
import numpy as np
import pickle
import eagle_IO.eagle_IO as E
import matplotlib.gridspec as gridspec
from unyt import mh, cm, Gyr, g, Msun, Mpc
from matplotlib.colors import LogNorm


def get_part_ids(sim, snapshot, part_type, all_parts=False):
    # Get the particle IDs
    if all_parts:
        part_ids = E.read_array('SNAP', sim, snapshot, 'PartType' + str(part_type) + '/ParticleIDs', numThreads=8)
    else:
        part_ids = E.read_array('PARTDATA', sim, snapshot, 'PartType' + str(part_type) + '/ParticleIDs',
                                numThreads=8)

    # Extract the halo IDs (group names/keys) contained within this snapshot
    group_part_ids = E.read_array('PARTDATA', sim, snapshot, 'PartType' + str(part_type) + '/ParticleIDs',
                                  numThreads=8)
    grp_ids = E.read_array('PARTDATA', sim, snapshot, 'PartType' + str(part_type) + '/GroupNumber',
                           numThreads=8)
    subgrp_ids = E.read_array('PARTDATA', sim, snapshot, 'PartType' + str(part_type) + '/SubGroupNumber',
                              numThreads=8)

    # Remove particles not associated to a subgroup
    okinds = subgrp_ids != 1073741824
    group_part_ids = group_part_ids[okinds]
    grp_ids = grp_ids[okinds]
    subgrp_ids = subgrp_ids[okinds]

    # Convert IDs to float(groupNumber.SubGroupNumber) format, i.e. group 1 subgroup 11 = 1.00011
    halo_ids = np.zeros(grp_ids.size, dtype=float)
    for (ind, g), sg in zip(enumerate(grp_ids), subgrp_ids):
        halo_ids[ind] = float(str(int(g)) + '.%05d' % int(sg))

    # Sort particle IDs
    part_ids = np.sort(part_ids)  # this is optional
    unsort_part_ids = part_ids[:]
    sinds = np.argsort(part_ids)
    part_ids = part_ids[sinds]

    # Get the index of particles in the snapshot array from the in group array
    sorted_index = np.searchsorted(part_ids, group_part_ids)
    yindex = np.take(sinds, sorted_index, mode="clip")
    mask = unsort_part_ids[yindex] != group_part_ids
    result = np.ma.array(yindex, mask=mask)

    # Apply mask to the id arrays
    part_groups = halo_ids[np.logical_not(result.mask)]
    parts_in_groups = result.data[np.logical_not(result.mask)]

    # Produce a dictionary containing the index of particles in each halo
    halo_part_inds = {}
    for ind, grp in zip(parts_in_groups, part_groups):
        halo_part_inds.setdefault(grp, set()).update({ind})

    return halo_part_inds


def get_data(masslim=1e8, load=False):

    # Define snapshots
    snaps = ['004_z008p075', '008_z005p037', '010_z003p984',
             '013_z002p478', '017_z001p487', '018_z001p259',
             '019_z001p004', '020_z000p865', '024_z000p366']
    prog_snaps = ['003_z008p988', '007_z005p487', '009_z004p485',
                  '012_z003p017', '016_z001p737', '017_z001p487',
                  '018_z001p259', '019_z001p004', '023_z000p503']

    if load:

        with open('metvsbd.pck', 'rb') as pfile1:
            save_dict = pickle.load(pfile1)
        stellar_met_dict = save_dict['met']
        stellar_bd_dict = save_dict['bd']

    else:

        stellar_met_dict = {}
        stellar_bd_dict = {}

        for snap, prog_snap in zip(snaps, prog_snaps):

            path = '/cosma7/data//Eagle/ScienceRuns/Planck1/L0100N1504/PE/REFERENCE/data'

            # Get particle IDs
            halo_part_inds = get_part_ids(path, snap, 4, all_parts=False)

            # Get halo IDs and halo data
            try:
                subgrp_ids = E.read_array('SUBFIND', path, snap, 'Subhalo/SubGroupNumber', numThreads=8)
                grp_ids = E.read_array('SUBFIND', path, snap, 'Subhalo/GroupNumber', numThreads=8)
                gal_ms = E.read_array('SUBFIND', path, snap, 'Subhalo/ApertureMeasurements/Mass/030kpc',
                                      noH=False, physicalUnits=False, numThreads=8)[:, 4] * 10**10
                gal_bd = E.read_array('PARTDATA', path, snap, 'PartType4/BirthDensity', noH=True,
                                        physicalUnits=True, numThreads=8)
                gal_met = E.read_array('PARTDATA', path, snap, 'PartType4/Metallicity', noH=True,
                                       physicalUnits=True, numThreads=8)
            except ValueError:
                continue
            except OSError:
                continue
            except KeyError:
                continue

            z_str = snap.split('z')[1].split('p')
            z = float(z_str[0] + '.' + z_str[1])

            # Remove particles not associated to a subgroup
            okinds = np.logical_and(subgrp_ids != 1073741824, gal_ms > masslim)
            grp_ids = grp_ids[okinds]
            subgrp_ids = subgrp_ids[okinds]
            halo_ids = np.zeros(grp_ids.size, dtype=float)
            for (ind, g), sg in zip(enumerate(grp_ids), subgrp_ids):
                halo_ids[ind] = float(str(int(g)) + '.%05d'%int(sg))

            stellar_bd = []
            stellar_met = []
            for halo in halo_ids:

                # Add stars from these galaxies
                stellar_bd.extend(gal_bd[list(halo_part_inds[halo])])
                stellar_met.extend(gal_met[list(halo_part_inds[halo])])

            stellar_bd_dict[snap] = stellar_bd
            stellar_met_dict[snap] = stellar_met

        with open('metvsbd.pck', 'wb') as pfile1:
            pickle.dump({'bd': stellar_bd_dict, 'met': stellar_met_dict}, pfile1)

    return stellar_bd_dict, stellar_met_dict


snaps = ['004_z008p075', '008_z005p037', '010_z003p984',
         '013_z002p478', '017_z001p487', '018_z001p259',
         '019_z001p004', '020_z000p865', '024_z000p366']

stellar_bd_dict, stellar_met_dict = get_data(masslim=10**9.5, load=True)

stellar_met = np.concatenate(list(stellar_met_dict.values()))
stellar_bd = np.concatenate(list(stellar_bd_dict.values())) * 10**10

# plt.style.use("mnras.mplstyle")

# EAGLE parameters
parameters = {"f_E,min": 0.3,
              "f_E,max": 3,
              "n_Z": 0.87,
              "n_n": 0.87,
              "Z_pivot": 0.1 * 0.012,
              "n_pivot": 0.67}

star_formation_parameters = {"threshold_Z0": 0.002,
                             "threshold_n0": 0.1,
                             "slope": -0.64}


number_of_bins = 128

# Constants; these could be put in the parameter file but are rarely changed.
birth_density_bins = np.logspace(-3, 6.8, number_of_bins)
metal_mass_fraction_bins = np.logspace(-5.9, 0, number_of_bins)

# Now need to make background grid of f_E.
birth_density_grid, metal_mass_fraction_grid = np.meshgrid(
    0.5 * (birth_density_bins[1:] + birth_density_bins[:-1]),
    0.5 * (metal_mass_fraction_bins[1:] + metal_mass_fraction_bins[:-1]))

f_E_grid = parameters["f_E,min"] + (parameters["f_E,max"] - parameters["f_E,min"]) / (
    1.0
    + (metal_mass_fraction_grid / parameters["Z_pivot"]) ** parameters["n_Z"]
    * (birth_density_grid / parameters["n_pivot"]) ** (-parameters["n_n"])
)

# fig, ax = plt.subplots()
#
# ax.loglog()
#
# mappable = ax.pcolormesh(birth_density_bins, metal_mass_fraction_bins, f_E_grid, norm=LogNorm(1e-1, 1e1))
#
# fig.colorbar(mappable, label="Feedback energy fraction $f_{th}$", pad=0)
#
# metal_mass_fractions = stellar_met
# stellar_bd = stellar_bd
#
# H, _, _ = np.histogram2d((stellar_bd * Msun / Mpc**3 / mh).to(1 / cm ** 3).value, metal_mass_fractions,
#                          bins=[birth_density_bins, metal_mass_fraction_bins])
#
# ax.contour(birth_density_grid, metal_mass_fraction_grid, H.T, levels=6, cmap="magma")
#
# # Add line showing SF law
# sf_threshold_density = star_formation_parameters["threshold_n0"] * \
#                        (metal_mass_fraction_bins
#                         / star_formation_parameters["threshold_Z0"]) ** (star_formation_parameters["slope"])
# ax.plot(sf_threshold_density, metal_mass_fraction_bins, linestyle="dashed", label="SF threshold")
#
# legend = ax.legend(markerfirst=True, loc="lower left")
# plt.setp(legend.get_texts())
#
# ax.set_xlabel("Stellar Birth Density [$n_H$ cm$^{-3}$]")
# ax.set_ylabel("Smoothed Metal Mass Fraction $Z$")
#
# try:
#     fontsize=legend.get_texts()[0].get_fontsize()
# except:
#     fontsize=6
#
#
# ax.text(
#     0.975,
#     0.025,
#     "\n".join(
#         [f"${k.replace('_', '_{') + '}'}$: ${v:.4g}$" for k, v in parameters.items()]
#     ),
#     color="white",
#     transform=ax.transAxes,
#     ha="right",
#     va="bottom",
#     fontsize=fontsize,
# )
#
# ax.text(
#     0.975,
#     0.975,
#     "Contour lines linearly spaced",
#     color="white",
#     transform=ax.transAxes,
#     ha="right",
#     va="top",
#     fontsize=fontsize,
# )
#
# fig.savefig(f"plots/birth_density_metallicity.png")
#
# plt.close()

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

    mappable = ax.pcolormesh(birth_density_bins, metal_mass_fraction_bins, f_E_grid, norm=LogNorm(1e-1, 1e1))

    if i == 0 and j == 0:
        # Add colorbars
        cax1 = ax.inset_axes([0.1, 0.1, 0.3, 0.03])
        cbar1 = fig.colorbar(mappable, cax=cax1, orientation="horizontal")
        cbar1.ax.set_xlabel(r'$f_{th}$', labelpad=1.5, fontsize=9, color='w')
        cbar1.ax.xaxis.set_label_position('top')
        cbar1.ax.tick_params(axis='x', labelsize=8, color='w')

    metal_mass_fractions = np.array(stellar_met_dict[snap])
    stellar_bd = np.array(stellar_bd_dict[snap]) * 10**10

    H, _, _ = np.histogram2d((stellar_bd * Msun / Mpc ** 3 / mh).to(1 / cm ** 3).value, metal_mass_fractions,
                             bins=[birth_density_bins, metal_mass_fraction_bins])

    ax.contour(birth_density_grid, metal_mass_fraction_grid, H.T, levels=6, cmap="magma")

    # Add line showing SF law
    sf_threshold_density = star_formation_parameters["threshold_n0"] * \
                           (metal_mass_fraction_bins
                            / star_formation_parameters["threshold_Z0"]) ** (star_formation_parameters["slope"])
    ax.plot(sf_threshold_density, metal_mass_fraction_bins, linestyle="dashed", label="SF threshold")

    ax.text(0.1, 0.95, f'$z={z}$', bbox=dict(boxstyle="round,pad=0.3", fc='w', ec="k", lw=1, alpha=0.8),
            transform=ax.transAxes, horizontalalignment='right', fontsize=8)

    axlims_x.extend(ax.get_xlim())
    axlims_y.extend(ax.get_ylim())

    # Label axes
    if i == 2:
        ax.set_xlabel("Stellar Birth Density [$n_H$ cm$^{-3}$]")
    if j == 0:
        ax.set_ylabel("Smoothed Metal Mass Fraction $Z$")

for ax in [ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8, ax9]:
    ax.set_xlim(np.min(axlims_x), np.max(axlims_x))
    ax.set_ylim(np.min(axlims_y), np.max(axlims_y))
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
         color="white", transform=ax9.transAxes, ha="right", va="bottom", fontsize=fontsize)

ax3.text(0.975, 0.975, "Contour lines linearly spaced", color="white", transform=ax3.transAxes, ha="right", va="top",
         fontsize=fontsize)

fig.savefig('plots/birthdensity_metallicity_redshift.png', bbox_inches='tight')
