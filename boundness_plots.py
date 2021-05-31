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
from scipy.spatial import cKDTree


sns.set_style('whitegrid')


def plot_median_stat(xs, ys, ax, color, norm, bins=None, ls='-', func="median", alpha=0.2):

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

    im = ax.plot(bin_cents[okinds], y_stat[okinds], linestyle=ls, color=color, alpha=alpha)

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


def upper_tri_masking(A):
    m = A.shape[0]
    r = np.arange(m)
    mask = r[:, None] <= r
    return A[mask]


def kinetic(vels, masses):

    # Compute kinetic energy of the halo
    KE = 0.5 * masses * (vels[:, 0]**2 + vels[:, 1]**2 + vels[:, 2]**2)

    return KE


def grav(pos, soft, masses, G, conv, tree):

    GE = np.zeros(pos.shape[0])

    ks = tree.query_ball_point(pos, r=0.03, return_length=True, workers=16)

    dist, ind_lst = tree.query(pos, k=ks, workers=16)

    # Get separations
    for (i, ds), inds, m in zip(enumerate(dist), ind_lst, masses):

        # okinds = inds < masses.size
        # ds = np.array(ds)[okinds]
        # inds = np.array(inds)[okinds]

        # Get masses
        sep_masses = m * masses[inds]

        # Compute the sum of the gravitational energy of each particle from
        # GE = G*Sum_i(m_i*Sum_{j<i}(m_j/sqrt(r_{ij}**2+s**2)))
        GE[i] = G * np.sum(sep_masses / np.sqrt(ds**2 + soft ** 2)) * conv

    return GE


def halo_energy_calc_exact(pos, vels, part_ms, G, soft, conv, tree):

    # Compute kinetic energy of the halo
    KE = kinetic(vels, part_ms)

    GE = grav(pos, soft, part_ms, G, conv, tree)

    # Compute halo's energy
    halo_energy = KE - GE

    return halo_energy, KE, GE


def get_main(snap, G, conv):

    rs_dict = {}
    E_dict = {}
    GEs_dict = {}
    KEs_dict = {}
    ratio_dict = {}
    total_mass = {}

    idkey = 0

    # Calculate bins
    mass_bins = np.logspace(8, 11.5, 15)

    regions = []
    for reg in range(0, 40):
        if reg < 10:
            regions.append('0' + str(reg))
        else:
            regions.append(str(reg))

    # Get the redshift
    z_str = snap.split('z')[1].split('p')
    z = float(z_str[0] + '.' + z_str[1])

    # Define comoving softening length in pkpc
    csoft = 0.001802390 / 0.6777 / (1 + z) * 1e3

    for reg in regions:

        path = '/cosma/home/dp004/dc-rope1/FLARES/FLARES-1/G-EAGLE_' + reg + '/data'

        # Load all necessary arrays
        try:
            subfind_grp_ids = E.read_array('SUBFIND', path, snap, 'Subhalo/GroupNumber', numThreads=8)
            subfind_subgrp_ids = E.read_array('SUBFIND', path, snap, 'Subhalo/SubGroupNumber', numThreads=8)
            gal_cops = E.read_array('SUBFIND', path, snap, 'Subhalo/CentreOfPotential', noH=True,
                                    physicalUnits=True, numThreads=8)
            all_gal_ns = E.read_array('SUBFIND', path, snap, 'Subhalo/SubLengthType', numThreads=8)
            all_gal_totms = E.read_array('SUBFIND', path, snap, 'Subhalo/MassType', noH=True,
                                    physicalUnits=True, numThreads=8) * 10**10
            all_gal_ms = E.read_array('SUBFIND', path, snap, 'Subhalo/ApertureMeasurements/Mass/030kpc', noH=True,
                                    physicalUnits=True, numThreads=8) * 10 ** 10
        except ValueError:
            continue
        except OSError:
            continue
        except KeyError:
            continue

        max_ind = np.argmax(all_gal_totms[:, 1])
        dm_mass = all_gal_totms[max_ind, 1] / all_gal_ns[max_ind, 1]

        # Remove particles not in a subgroup
        okinds = np.logical_and(subfind_subgrp_ids != 1073741824,
                                np.logical_and((all_gal_ns[:, 1]) > 0,
                                               np.logical_and(all_gal_ms[:, 4] >= 10**8,
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
        gal_ms = {}
        means = {}
        all_poss = []
        all_vels = []
        part_ms = []
        part_ids = []

        # Store the stellar mass of the galaxy and cop
        for id, cop, m in zip(test_gals, test_cops, test_masses):
            gal_ms[id] = m
            means[id] = cop

        for part_type in [0, 1, 4, 5]:

            print(part_type)

            # Get gas particle information
            try:
                poss = E.read_array('PARTDATA', path, snap, 'PartType' + str(part_type) + '/Coordinates', noH=True,
                                    physicalUnits=True, numThreads=8)
                vels = E.read_array('PARTDATA', path, snap, 'PartType' + str(part_type) + '/Velocity', noH=True,
                                    physicalUnits=True, numThreads=8)
                if part_type != 1:
                    masses = E.read_array('PARTDATA', path, snap, 'PartType' + str(part_type) + '/Mass', noH=True,
                                          physicalUnits=True, numThreads=8) * 10 ** 10
                else:
                    masses = np.full(poss.shape[0], dm_mass)

                grp_ids = E.read_array('PARTDATA', path, snap, 'PartType' + str(part_type) + '/GroupNumber', noH=True,
                                       physicalUnits=True, verbose=False, numThreads=8)

                subgrp_ids = E.read_array('PARTDATA', path, snap, 'PartType' + str(part_type) + '/SubGroupNumber', noH=True,
                                          physicalUnits=True, verbose=False, numThreads=8)
            except ValueError:
                continue
            except OSError:
                continue
            except KeyError:
                continue

            # Convert IDs to float(groupNumber.SubGroupNumber) format, i.e. group 1 subgroup 11 = 1.00011
            part_halo_ids = np.zeros(grp_ids.size, dtype=float)
            for (ind, g), sg in zip(enumerate(grp_ids), subgrp_ids):
                part_halo_ids[ind] = float(str(int(g)) + '.%05d' % int(sg))

            okinds = np.isin(part_halo_ids, test_gals)
            part_halo_ids = part_halo_ids[okinds]
            poss = poss[okinds, :]
            vels = vels[okinds, :]
            masses = masses[okinds]

            print("There are", len(masses), "particles")

            all_poss.extend(poss)
            all_vels.extend(vels)
            part_ms.extend(masses)
            part_ids.extend(part_halo_ids)

        print('Got particle IDs')

        print(test_gals.size)

        all_poss = np.array(all_poss)
        all_vels = np.array(all_vels)
        part_ms = np.array(part_ms)
        part_ids = np.array(part_ids)

        print(all_poss.shape, all_vels.shape, part_ms.shape)

        # Get cops for each particle
        cops = np.zeros((len(part_ms), 3))
        for ind, hid in enumerate(part_ids):
            cops[ind, :] = means[hid]

        # Build tree
        if all_poss.shape[0] > 0:
            tree = cKDTree(all_poss)
        else:
            continue

        # Get the luminosities
        gal_part_poss = all_poss - cops
        gal_rs = calc_3drad(gal_part_poss)

        # # Limit to 30pkpc aperture
        # okinds = gal_rs <= 30
        # gal_rs = gal_rs[okinds]
        # gal_part_poss = gal_part_poss[okinds]
        # gal_part_vel = gal_part_vel[okinds]
        # part_ms = part_ms[okinds]

        # Calculate potential
        Ens, GEs, KEs = halo_energy_calc_exact(all_poss, all_vels,
                                               part_ms, G, csoft,
                                               conv, tree)

        idkey = 0

        for r, En, GE, KE, hid in zip(gal_rs, Ens, GEs, KEs, part_ids):

            rs_dict.setdefault(idkey, []).append(r)
            E_dict.setdefault(idkey, []).append(En)
            GEs_dict.setdefault(idkey, []).append(GE)
            KEs_dict.setdefault(idkey, []).append(KE)
            ratio_dict.setdefault(idkey, []).append(GE / KE)
            total_mass[idkey] = gal_ms[hid]

            idkey += 1

    # Set up colormap
    # normalize item number values to colormap
    norm = ml.colors.TwoSlopeNorm(vmin=8, vcenter=9, vmax=11.5)

    jet = plt.get_cmap('Spectral')
    scalarMap = ml.cm.ScalarMappable(norm=norm, cmap=jet)

    fig = plt.figure()
    ax = fig.add_subplot(111)

    ax.loglog()

    total_masses = list(total_mass.values())
    sinds = np.argsort(total_masses)
    ids = np.array(list(rs_dict.keys()))[sinds]

    for id in ids:

        sinds = np.argsort(rs_dict[id])
        c = scalarMap.to_rgba(np.log10(total_mass[id]))

        plot_median_stat(np.array(rs_dict[id])[sinds] * 10**3,
                         np.array(E_dict[id])[sinds], ax, norm=norm,
                         color=c, alpha=0.1)

    ax.axvline(csoft, linestyle="--", color='k')

    ax.set_xlabel("$R /$ [pkpc]")
    ax.set_ylabel("$U / [\mathrm{M}_{\odot} \ \mathrm{km}^2 \ \mathrm{s}^{-2}]$")

    # Make an axis for the colorbar on the right side
    cbar = fig.colorbar(scalarMap)

    cbar.set_label("$M_{\star}/M_{\odot}$")

    fig.savefig("plots/radial_energy_" + snap + ".png", bbox_inches="tight")

    fig = plt.figure()
    ax = fig.add_subplot(111)

    ax.loglog()

    total_masses = list(total_mass.values())
    sinds = np.argsort(total_masses)
    ids = np.array(list(rs_dict.keys()))[sinds]

    for id in ids:

        sinds = np.argsort(rs_dict[id])
        c = scalarMap.to_rgba(np.log10(total_mass[id]))

        plot_median_stat(np.array(rs_dict[id])[sinds] * 10**3,
                         np.array(ratio_dict[id])[sinds], ax, norm=norm,
                         color=c, alpha=0.1)

    ax.axvline(csoft, linestyle="--", color='k')

    ax.set_xlabel("$R /$ [pkpc]")
    ax.set_ylabel("$\mathrm{GE}/\mathrm{KE}$")

    # Make an axis for the colorbar on the right side
    cbar = fig.colorbar(scalarMap)

    cbar.set_label("$M_{\star}/M_{\odot}$")

    fig.savefig("plots/radial_energyratio_" + snap + ".png", bbox_inches="tight")

    bin_inds = np.digitize(list(total_mass.values()), mass_bins) - 1

    binned_rs_dict = {}
    binned_ratio_dict = {}

    for id, bin in zip(total_mass.keys(), bin_inds):

        binned_rs_dict.setdefault(bin, []).extend(rs_dict[id])
        binned_ratio_dict.setdefault(bin, []).extend(ratio_dict[id])

    fig = plt.figure()
    ax = fig.add_subplot(111)

    ax.loglog()

    mass_bin_wid = mass_bins[1] - mass_bins[0]
    mass_bin_cents = mass_bins[1:] - mass_bin_wid

    for bin, m in enumerate(mass_bin_cents):

        try:
            sinds = np.argsort(binned_rs_dict[bin])
        except KeyError:
            continue
        c = scalarMap.to_rgba(np.log10(m))
        print(np.log10(m), c, bin)

        plot_median_stat(np.array(binned_rs_dict[bin])[sinds] * 10**3,
                         np.array(binned_ratio_dict[bin])[sinds], ax, norm=norm,
                         color=c, alpha=1)

    ax.axvline(csoft, linestyle="--", color='k')

    ax.set_xlabel("$R /$ [pkpc]")
    ax.set_ylabel("$\mathrm{GE}/\mathrm{KE}$")

    # Make an axis for the colorbar on the right side
    cbar = fig.colorbar(scalarMap)

    cbar.set_label("$M_{\star}/M_{\odot}$")

    fig.savefig("plots/radial_ratio_" + snap + "_binned.png",
                bbox_inches="tight")

    # ========================== Cumalative ==========================

    fig = plt.figure()
    ax = fig.add_subplot(111)

    ax.loglog()

    total_masses = list(total_mass.values())
    sinds = np.argsort(total_masses)
    ids = np.array(list(rs_dict.keys()))[sinds]

    for id in ids:
        sinds = np.argsort(rs_dict[id])
        c = scalarMap.to_rgba(np.log10(total_mass[id]))

        plot_median_stat(np.array(rs_dict[id])[sinds] * 10 ** 3,
                         np.cumsum(-np.array(GEs_dict[id])[sinds]) + np.cumsum(np.array(KEs_dict[id])[sinds]), ax, norm=norm,
                         color=c, alpha=0.1)

    ax.axvline(csoft, linestyle="--", color='k')

    ax.set_xlabel("$R /$ [pkpc]")
    ax.set_ylabel(
        "$U / [\mathrm{M}_{\odot} \ \mathrm{km}^2 \ \mathrm{s}^{-2}]$")

    # Make an axis for the colorbar on the right side
    cbar = fig.colorbar(scalarMap)

    cbar.set_label("$M_{\star}/M_{\odot}$")

    fig.savefig("plots/radial_energy_" + snap + "_cumalative.png", bbox_inches="tight")

    fig = plt.figure()
    ax = fig.add_subplot(111)

    ax.loglog()

    total_masses = list(total_mass.values())
    sinds = np.argsort(total_masses)
    ids = np.array(list(rs_dict.keys()))[sinds]

    for id in ids:
        sinds = np.argsort(rs_dict[id])
        c = scalarMap.to_rgba(np.log10(total_mass[id]))

        plot_median_stat(np.array(rs_dict[id])[sinds] * 10 ** 3,
                         np.cumsum(np.array(GEs_dict[id])[sinds]) / np.cumsum(np.array(KEs_dict[id])[sinds]), ax, norm=norm,
                         color=c, alpha=0.1)

    ax.axvline(csoft, linestyle="--", color='k')

    ax.set_xlabel("$R /$ [pkpc]")
    ax.set_ylabel("$\mathrm{GE}/\mathrm{KE}$")

    # Make an axis for the colorbar on the right side
    cbar = fig.colorbar(scalarMap)

    cbar.set_label("$M_{\star}/M_{\odot}$")

    fig.savefig("plots/radial_energyratio_" + snap + "_cumalative.png",
                bbox_inches="tight")

    bin_inds = np.digitize(list(total_mass.values()), mass_bins) - 1

    binned_rs_dict = {}
    binned_ratio_dict = {}

    for id, bin in zip(total_mass.keys(), bin_inds):
        binned_rs_dict.setdefault(bin, []).extend(rs_dict[id])
        sinds = np.argsort(rs_dict[id])
        binned_ratio_dict.setdefault(bin, []).extend(np.cumsum(np.array(GEs_dict[id])[sinds]) / np.cumsum(np.array(KEs_dict[id])[sinds]))

    fig = plt.figure()
    ax = fig.add_subplot(111)

    ax.loglog()

    mass_bin_wid = mass_bins[1] - mass_bins[0]
    mass_bin_cents = mass_bins[1:] - mass_bin_wid

    for bin, m in enumerate(mass_bin_cents):

        try:
            sinds = np.argsort(binned_rs_dict[bin])
        except KeyError:
            continue
        c = scalarMap.to_rgba(np.log10(m))
        print(np.log10(m), c, bin)

        plot_median_stat(np.array(binned_rs_dict[bin])[sinds] * 10 ** 3,
                         np.array(binned_ratio_dict[bin])[sinds], ax,
                         norm=norm,
                         color=c, alpha=1)

    ax.axvline(csoft, linestyle="--", color='k')

    ax.set_xlabel("$R /$ [pkpc]")
    ax.set_ylabel("$\mathrm{GE}/\mathrm{KE}$")

    # Make an axis for the colorbar on the right side
    cbar = fig.colorbar(scalarMap)

    cbar.set_label("$M_{\star}/M_{\odot}$")

    fig.savefig("plots/radial_ratio_" + snap + "_binned_cumalative.png",
                bbox_inches="tight")


snaps = ['000_z015p000', '001_z014p000', '002_z013p000',
         '003_z012p000', '004_z011p000', '005_z010p000',
         '006_z009p000', '007_z008p000', '008_z007p000',
         '009_z006p000', '010_z005p000', '011_z004p770']

G = (const.G.to(u.Mpc ** 3 * u.M_sun ** -1 * u.s ** -2)).value
conv = (u.M_sun * u.Mpc**2 * u.s ** -2).to(u.M_sun * u.km**2 * u.s ** -2)

for snap in snaps:
    get_main(snap, G, conv)
