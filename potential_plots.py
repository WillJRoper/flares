#!/cosma/home/dp004/dc-rope1/.conda/envs/flares-env/bin/python
import numba as nb
import astropy.units as u
import astropy.constants as const
import matplotlib as ml
ml.use('Agg')
import numpy as np
from scipy.stats import binned_statistic
import matplotlib.pyplot as plt
import eagle_IO.eagle_IO as E
import seaborn as sns
from matplotlib import cm


sns.set_style('whitegrid')


def plot_median_stat(xs, ys, ax, color, norm, bins=None, ls='-', func="median"):

    if bins == None:
        bin = np.logspace(np.log10(xs.min()), np.log10(xs.max()), 25)
    else:
        bin = bins

    # Compute binned statistics
    y_stat, binedges, bin_ind = binned_statistic(xs, ys, statistic=func, bins=bin)

    # Compute bincentres
    bin_wid = binedges[1] - binedges[0]
    bin_cents = binedges[1:] - bin_wid / 2

    okinds = np.logical_and(~np.isnan(bin_cents), ~np.isnan(y_stat))

    im = ax.plot(bin_cents[okinds], y_stat[okinds], linestyle=ls, color=color)

    return im


def plot_spread_stat(xs, ys, ax, bins=None):

    if bins == None:
        bin = np.logspace(np.log10(xs.min()), np.log10(xs.max()), 25)
    else:
        bin = bins

    # Compute binned statistics
    y_stat_16, binedges, bin_ind = binned_statistic(xs, ys, statistic=lambda y: np.percentile(y, 16), bins=bin)
    y_stat_84, binedges, bin_ind = binned_statistic(xs, ys, statistic=lambda y: np.percentile(y, 84), bins=bin)

    # Compute bincentres
    bin_wid = binedges[1] - binedges[0]
    bin_cents = binedges[1:] - bin_wid / 2

    okinds = np.logical_and(~np.isnan(bin_cents), np.logical_and(~np.isnan(y_stat_16), ~np.isnan(y_stat_84)))

    ax.fill_between(bin_cents[okinds], y_stat_16[okinds], y_stat_84[okinds], alpha=0.4)


def get_part_inds(halo_ids, part_ids, group_part_ids, sorted):
    """ A function to find the indexes and halo IDs associated to particles/a particle producing an array for each

    :param halo_ids:
    :param part_ids:
    :param group_part_ids:
    :return:
    """

    # Sort particle IDs if required and store an unsorted version in an array
    if sorted:
        part_ids = np.sort(part_ids)
    unsort_part_ids = np.copy(part_ids)

    # Get the indices that would sort the array (if the array is sorted this is just a range form 0-Npart)
    if sorted:
        sinds = np.arange(part_ids.size)
    else:
        sinds = np.argsort(part_ids)
        part_ids = part_ids[sinds]

    # Get the index of particles in the snapshot array from the in particles in a group array
    sorted_index = np.searchsorted(part_ids, group_part_ids)  # find the indices that would sort the array
    yindex = np.take(sinds, sorted_index, mode="raise")  # take the indices at the indices found above
    mask = unsort_part_ids[yindex] != group_part_ids  # define the mask based on these particles
    result = np.ma.array(yindex, mask=mask)  # create a mask array

    # Apply the mask to the id arrays to get halo ids and the particle indices
    part_groups = halo_ids[np.logical_not(result.mask)]  # halo ids
    parts_in_groups = result.data[np.logical_not(result.mask)]  # particle indices

    return parts_in_groups, part_groups


@nb.njit(nogil=True, parallel=True)
def calc_3drad(poss):

    # Get galaxy particle indices
    rs = np.sqrt(poss[:, 0] ** 2 + poss[:, 1] ** 2 + poss[:, 2] ** 2)

    return rs

def calc_pot(totM, mg, r, soft, G):

    return G * totM * mg / (r + soft)


def get_main(path, snap, G):

    # Get the redshift
    z_str = snap.split('z')[1].split('p')
    z = float(z_str[0] + '.' + z_str[1])

    # Load all necessary arrays
    subfind_grp_ids = E.read_array('SUBFIND', path, snap, 'Subhalo/GroupNumber', numThreads=8)
    subfind_subgrp_ids = E.read_array('SUBFIND', path, snap, 'Subhalo/SubGroupNumber', numThreads=8)
    gal_cops = E.read_array('SUBFIND', path, snap, 'Subhalo/CentreOfPotential', noH=True,
                            physicalUnits=True, numThreads=8) * 1e3
    all_gal_ns = E.read_array('SUBFIND', path, snap, 'Subhalo/SubLengthType', numThreads=8)
    all_gal_totms = E.read_array('SUBFIND', path, snap, 'Subhalo/MassType', noH=True,
                            physicalUnits=True, numThreads=8) * 10**10
    all_gal_ms = E.read_array('SUBFIND', path, snap, 'Subhalo/ApertureMeasurements/Mass/030kpc', noH=True,
                            physicalUnits=True, numThreads=8) * 10 ** 10

    max_ind = np.argmax(all_gal_totms[:, 1])
    dm_mass = all_gal_totms[max_ind, 1] / all_gal_ns[max_ind, 1]

    # Remove particles not in a subgroup
    okinds = np.logical_and(subfind_subgrp_ids != 1073741824,
                            np.logical_and((all_gal_ns[:, 1]) > 0,
                                           np.logical_and(all_gal_ms[:, 4] >= 10**8.5,
                                                          np.logical_and(all_gal_ns[:, 0] > 0,
                                                                         all_gal_ns[:, 5] > 0)
                                                          )
                                           )
                            )
    subfind_grp_ids = subfind_grp_ids[okinds]
    subfind_subgrp_ids = subfind_subgrp_ids[okinds]
    gal_cops = gal_cops[okinds]
    all_gal_ms = all_gal_ms[okinds]

    # Convert IDs to float(groupNumber.SubGroupNumber) format, i.e. group 1 subgroup 11 = 1.00011
    halo_ids = np.zeros(subfind_grp_ids.size, dtype=float)
    for (ind, g), sg in zip(enumerate(subfind_grp_ids), subfind_subgrp_ids):
        halo_ids[ind] = float(str(int(g)) + '.%05d' % int(sg))

    star_halo_ids = np.copy(halo_ids)

    # Extract galaxies to test
    test_gals = star_halo_ids
    test_cops = gal_cops
    test_masses = all_gal_ms[:, 4]

    # Set up dictionaries to store results
    part_ms = {}
    gal_ms = {}
    means = {}
    all_poss = {}

    # Store the stellar mass of the galaxy and cop
    for id, cop, m in zip(test_gals, test_cops, test_masses):
        gal_ms[id] = m
        means[id] = cop

    for part_type in [0, 1, 4, 5]:

        print(part_type)

        # Get gas particle information
        poss = E.read_array('PARTDATA', path, snap, 'PartType' + str(part_type) + '/Coordinates', noH=True,
                            physicalUnits=True, numThreads=8) * 1e3
        if part_type != 1:
            masses = E.read_array('PARTDATA', path, snap, 'PartType' + str(part_type) + '/Mass', noH=True,
                                  physicalUnits=True, numThreads=8) * 10 ** 10
        else:
            masses = np.full(poss.shape[0], dm_mass)

        grp_ids = E.read_array('PARTDATA', path, snap, 'PartType' + str(part_type) + '/GroupNumber', noH=True,
                               physicalUnits=True, verbose=False, numThreads=8)

        subgrp_ids = E.read_array('PARTDATA', path, snap, 'PartType' + str(part_type) + '/SubGroupNumber', noH=True,
                                  physicalUnits=True, verbose=False, numThreads=8)

        part_ids = E.read_array('PARTDATA', path, snap, 'PartType' + str(part_type) + '/ParticleIDs', noH=True,
                                physicalUnits=True, verbose=False, numThreads=8)

        # A copy of this array is needed for the extraction method
        group_part_ids = np.copy(part_ids)

        print("There are", len(subgrp_ids), "particles")

        # Convert IDs to float(groupNumber.SubGroupNumber) format, i.e. group 1 subgroup 11 = 1.00011
        part_halo_ids = np.zeros(grp_ids.size, dtype=float)
        for (ind, g), sg in zip(enumerate(grp_ids), subgrp_ids):
            part_halo_ids[ind] = float(str(int(g)) + '.%05d' % int(sg))

        okinds = np.isin(part_halo_ids, test_gals)
        part_halo_ids = part_halo_ids[okinds]
        part_ids = part_ids[okinds]
        group_part_ids = group_part_ids[okinds]
        poss = poss[okinds]
        masses = masses[okinds]

        print("There are", len(part_halo_ids), "particles")

        print("Got halo IDs")

        parts_in_groups, part_groups = get_part_inds(part_halo_ids, part_ids, group_part_ids, False)

        # Produce a dictionary containing the index of particles in each halo
        halo_part_inds = {}
        for ind, grp in zip(parts_in_groups, part_groups):
            halo_part_inds.setdefault(grp, set()).update({ind})

        # Now the dictionary is fully populated convert values from sets to arrays for indexing
        for key, val in halo_part_inds.items():
            halo_part_inds[key] = np.array(list(val))

        # Store the stellar mass of the galaxy and cop
        for id in test_gals:
            mask = halo_part_inds[id]
            all_poss.setdefault(id, []).extend(poss[mask, :])
            part_ms.setdefault(id, []).extend(masses[mask])

    print('Got particle IDs')

    # ======================== Set up images ========================

    # Define comoving softening length in pkpc
    csoft = 0.001802390 / 0.6777 / (1 + z) * 1e3

    rs_dict = {}
    pot_dict = {}
    total_mass = {}

    # Calculate bins
    mass_bins = np.logspace(10**8.5, 10**11.5, 6)
    bin_inds = np.digitize(test_masses, mass_bins)

    print(bin_inds)

    for id, bin in zip(test_gals, bin_inds):

        # Get the luminosities
        gal_part_poss = all_poss[id] - means[id]
        masses = np.array(part_ms[id])
        total_mass[id] = gal_ms[id]
        gal_rs = calc_3drad(gal_part_poss)

        sinds = np.argsort(gal_rs)
        masses = masses[sinds]
        gal_rs = gal_rs[sinds]

        cumal_mass = np.cumsum(masses)

        # Calculate potential
        pot = calc_pot(cumal_mass, masses, gal_rs, csoft, G)

        rs_dict.setdefault(bin, []).extend(gal_rs)
        pot_dict.setdefault(bin, []).extend(pot)

    fig = plt.figure()
    ax = fig.add_subplot(111)

    ax.loglog()

    # Set up colormap
    # normalize item number values to colormap
    norm = ml.colors.LogNorm(vmin=10**8, vmax=10**11.5)

    jet = plt.get_cmap('plasma')
    scalarMap = ml.cm.ScalarMappable(norm=norm, cmap=jet)

    mass_bin_wid = mass_bins[1] - mass_bins[0]
    mass_bin_cents = mass_bins[1:] - mass_bin_wid

    for bin, m in zip(bin_inds, mass_bin_cents):

        sinds = np.argsort(rs_dict[bin])
        c = scalarMap.to_rgba(m)
        print(m, c, bin)

        plot_median_stat(np.array(rs_dict[bin])[sinds], np.array(pot_dict[bin])[sinds], ax, norm=norm, color=c)
        # plot_spread_stat(np.array(rs_dict[gal])[sinds], np.array(pot_dict[gal])[sinds], ax)

    ax.axvline(csoft, linestyle="--", color='k')

    ax.set_xlabel("$R /$ [pkpc]")
    ax.set_ylabel("$|U(<R)| / [\mathrm{M}_{\odot} \ \mathrm{pkpc}^2 \ \mathrm{s}^{-2}]$")

    # Make an axis for the colorbar on the right side
    cbar = fig.colorbar(scalarMap)

    cbar.set_label("$\log_{10}(M_{\star}/M_{\odot})$")

    fig.savefig("plots/radial_potential_" + reg + "_" + snap + ".png", bbox_inches="tight")


G = (const.G.to(u.kpc ** 3 * u.M_sun ** -1 * u.s ** -2)).value

reg = "00"

path = '/cosma/home/dp004/dc-rope1/FLARES/FLARES-1/G-EAGLE_' + reg + '/data'

snap = '010_z005p000'

get_main(path, snap, G)
