#!/cosma/home/dp004/dc-rope1/.conda/envs/flares-env/bin/python
"""
Creates the plot of metallicity against birth density, with
the background coloured by f_E.
"""

import matplotlib.pyplot as plt
import numpy as np
import pickle
import eagle_IO.eagle_IO as E

from unyt import mh, cm, Gyr, g, Msun
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


stellar_met_dict, stellar_bd_dict = get_data(masslim=10**9.5, load=True)

stellar_met = np.concatenate(list(stellar_met_dict.values()))
stellar_bd = np.concatenate(list(stellar_bd_dict.values()))
print(stellar_met, stellar_bd)
# plt.style.use("mnras.mplstyle")

# EAGLE parameters
parameters = {"f_E,min": 0.3,
              "f_E,max": 3,
              "n_Z": 1.0,
              "n_n": 1.0,
              "Z_pivot": 0.1 * 0.012,
              "n_pivot": 0.67}

star_formation_parameters = {"threshold_Z0": 0.002,
                             "threshold_n0": 0.1,
                             "slope": -0.64}


number_of_bins = 128

# Constants; these could be put in the parameter file but are rarely changed.
birth_density_bins = np.logspace(-3, 5, number_of_bins)
metal_mass_fraction_bins = np.logspace(-6, 0, number_of_bins)

# Now need to make background grid of f_E.
birth_density_grid, metal_mass_fraction_grid = np.meshgrid(
    0.5 * (birth_density_bins[1:] + birth_density_bins[:-1]),
    0.5 * (metal_mass_fraction_bins[1:] + metal_mass_fraction_bins[:-1]))

f_E_grid = parameters["f_E,min"] + (parameters["f_E,max"] - parameters["f_E,min"]) / (
    1.0
    + (metal_mass_fraction_grid / parameters["Z_pivot"]) ** parameters["n_Z"]
    * (birth_density_grid / parameters["n_pivot"]) ** (-parameters["n_n"])
)

# Begin plotting

fig, ax = plt.subplots()

ax.loglog()

mappable = ax.pcolormesh(birth_density_bins, metal_mass_fraction_bins, f_E_grid, norm=LogNorm(1e-1, 1e1))

fig.colorbar(mappable, label="Feedback energy fraction $f_E$", pad=0)

metal_mass_fractions = stellar_met

H, _, _ = np.histogram2d((stellar_bd * Msun / cm**3 / mh).to(1 / cm ** 3).value, metal_mass_fractions,
                         bins=[birth_density_bins, metal_mass_fraction_bins])

ax.contour(birth_density_grid, metal_mass_fraction_grid, H.T, levels=6, cmap="viridis")

# Add line showing SF law
sf_threshold_density = star_formation_parameters["threshold_n0"] * \
                       (metal_mass_fraction_bins
                        / star_formation_parameters["threshold_Z0"]) ** (star_formation_parameters["slope"])
ax.plot(sf_threshold_density, metal_mass_fraction_bins, linestyle="dashed", label="SF threshold")

legend = ax.legend(markerfirst=True, loc="lower left")
plt.setp(legend.get_texts())

ax.set_xlabel("Stellar Birth Density [$n_H$ cm$^{-3}$]")
ax.set_ylabel("Smoothed Metal Mass Fraction $Z$ []")

try:
    fontsize=legend.get_texts()[0].get_fontsize()
except:
    fontsize=6


ax.text(
    0.975,
    0.025,
    "\n".join(
        [f"${k.replace('_', '_{') + '}'}$: ${v:.4g}$" for k, v in parameters.items()]
    ),
    color="white",
    transform=ax.transAxes,
    ha="right",
    va="bottom",
    fontsize=fontsize,
)

ax.text(
    0.975,
    0.975,
    "Contour lines linearly spaced",
    color="white",
    transform=ax.transAxes,
    ha="right",
    va="top",
    fontsize=fontsize,
)

fig.savefig(f"plots/birth_density_metallicity.png")