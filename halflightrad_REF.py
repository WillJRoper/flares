#!/cosma/home/dp004/dc-rope1/.conda/envs/flares-env/bin/python
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib
import astropy.units as u
from matplotlib.colors import LogNorm
import pickle
import os
import numba as nb
from utilities import calc_ages, get_Z_LOS
from scipy.stats import binned_statistic
from astropy.cosmology import Planck13 as cosmo
import seaborn as sns
os.environ['FLARE'] = '/cosma7/data/dp004/dc-wilk2/flare'
import FLARE.filters
import gc
from SynthObs.SED import models
import eagle_IO.eagle_IO as E
matplotlib.use('Agg')


sns.set_style('whitegrid')

# Define SED model
model = models.define_model('BPASSv2.2.1.binary/ModSalpeter_300',
                            path_to_SPS_grid = FLARE.FLARE_dir + '/data/SPS/nebular/3.0/') # DEFINE SED GRID -
model.dust_ISM = ('simple', {'slope': -1.0})
model.dust_BC = ('simple', {'slope': -1.0})

# Define the filters: FAKE.FAKE are just top-hat filters using for extracting rest-frame quantities.
filters = ['FAKE.TH.'+f for f in ['FUV', 'NUV', 'V']]
F = FLARE.filters.add_filters(filters, new_lam = model.lam)

# --- create new L grid for each filter. In units of erg/s/Hz
model.create_Lnu_grid(F)


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
    unsort_part_ids = np.copy(part_ids)
    sinds = np.argsort(part_ids)
    part_ids = part_ids[sinds]

    # Get the index of particles in the snapshot array from the in group array
    sorted_index = np.searchsorted(part_ids, group_part_ids)
    yindex = np.take(sinds, sorted_index, mode="raise")
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


def get_main(path, snap):

    # Get the redshift
    z_str = snap.split('z')[1].split('p')
    z = float(z_str[0] + '.' + z_str[1])

    # Load all necessary arrays
    all_poss = E.read_array('PARTDATA', path, snap, 'PartType4/Coordinates', noH=True,
                            physicalUnits=True, numThreads=8)
    gal_sml = E.read_array('PARTDATA', path, snap, 'PartType4/SmoothingLength', noH=True,
                           physicalUnits=True, numThreads=8)
    grp_ids = E.read_array('PARTDATA', path, snap, 'PartType4/GroupNumber', numThreads=8)
    subgrp_ids = E.read_array('PARTDATA', path, snap, 'PartType4/SubGroupNumber', numThreads=8)
    gal_ids = E.read_array('SUBFIND', path, snap, 'Subhalo/SubGroupNumber', numThreads=8)
    gal_gids = E.read_array('SUBFIND', path, snap, 'Subhalo/GroupNumber', numThreads=8)
    gal_cops = E.read_array('SUBFIND', path, snap, 'Subhalo/CentreOfPotential', noH=True,
                            physicalUnits=True, numThreads=8)

    # Load data for luminosities
    a_born = E.read_array('PARTDATA', path, snap, 'PartType4/StellarFormationTime', noH=True,
                          physicalUnits=True, numThreads=8)
    metallicities = E.read_array('PARTDATA', path, snap, 'PartType4/SmoothedMetallicity', noH=True,
                                 physicalUnits=True, numThreads=8)
    masses = E.read_array('PARTDATA', path, snap, 'PartType4/Mass', noH=True, physicalUnits=True,
                          numThreads=8) * 10 ** 10

    # Remove particles not in a subgroup
    nosub_mask = subgrp_ids != 1073741824
    all_poss = all_poss[nosub_mask, :]
    gal_sml = gal_sml[nosub_mask]
    grp_ids = grp_ids[nosub_mask]
    subgrp_ids = subgrp_ids[nosub_mask]
    a_born = a_born[nosub_mask]
    metallicities = metallicities[nosub_mask]
    masses = masses[nosub_mask]

    # Convert IDs to float(groupNumber.SubGroupNumber) format, i.e. group 1 subgroup 11 = 1.00011
    halo_ids = np.zeros(grp_ids.size, dtype=float)
    for (ind, g), sg in zip(enumerate(grp_ids), subgrp_ids):
        halo_ids[ind] = float(str(int(g)) + '.%05d' % int(sg))

    # Calculate ages
    ages = calc_ages(z, a_born)

    # Get particle indices
    halo_part_inds = get_subgroup_part_inds(path, snap, part_type=4, all_parts=False, sorted=False)

    # Get the position of each of these galaxies
    gal_ages = {}
    gal_mets = {}
    gal_ms = {}
    gal_smls = {}
    all_gal_poss = {}
    means = {}
    for id, cop in zip(halo_ids, gal_cops):
        mask = halo_part_inds[id]
        all_gal_poss[id] = all_poss[mask, :]
        gal_ages[id] = ages[mask]
        gal_mets[id] = metallicities[mask]
        gal_ms[id] = masses[mask]
        gal_smls[id] = gal_sml[mask]
        means[id] = cop

    print('There are', len(gal_ages.keys()), 'galaxies')

    del subgrp_ids, ages, all_poss, metallicities, masses, gal_sml, a_born, grp_ids

    gc.collect()

    # If there are no galaxies exit
    if len(gal_ages.keys()) == 0:
        return

    print('Got galaxy properties')

    # Get gas particle information
    gas_all_poss = E.read_array('PARTDATA', path, snap, 'PartType0/Coordinates', noH=True, physicalUnits=True,
                                numThreads=8)
    gsubgrp_ids = E.read_array('PARTDATA', path, snap, 'PartType0/SubGroupNumber', numThreads=8)
    gas_metallicities = E.read_array('PARTDATA', path, snap, 'PartType0/SmoothedMetallicity', noH=True,
                                     physicalUnits=True, numThreads=8)
    gas_smooth_ls = E.read_array('PARTDATA', path, snap, 'PartType0/SmoothingLength', noH=True, physicalUnits=True,
                                 numThreads=8)
    gas_masses = E.read_array('PARTDATA', path, snap, 'PartType0/Mass', noH=True, physicalUnits=True,
                              numThreads=8) * 10 ** 10

    # Remove particles not in a subgroup
    nosub_mask = gsubgrp_ids != 1073741824
    gas_all_poss = gas_all_poss[nosub_mask, :]
    gas_metallicities = gas_metallicities[nosub_mask]
    gas_smooth_ls = gas_smooth_ls[nosub_mask]
    gas_masses = gas_masses[nosub_mask]

    # Get particle indices
    halo_part_inds = get_subgroup_part_inds(path, snap, part_type=0, all_parts=False, sorted=False)

    # Get the position of each of these galaxies
    gas_mets = {}
    gas_ms = {}
    gas_smls = {}
    all_gas_poss = {}
    for g, sg in zip(gal_gids, gal_ids):
        id = float(str(int(g)) + '.%05d' % int(sg))
        mask = halo_part_inds[id]
        all_gas_poss[id] = gas_all_poss[mask, :]
        gas_mets[id] = gas_metallicities[mask]
        gas_ms[id] = gas_masses[mask]
        gas_smls[id] = gas_smooth_ls[mask]

    print('Got particle IDs')

    del gsubgrp_ids, gas_all_poss, gas_metallicities, gas_smooth_ls, gas_masses

    gc.collect()

    print('Got gas properties')

    save_dict = {'gal_ages': gal_ages, 'gal_mets': gal_mets, 'gal_ms': gal_ms, 'gas_mets': gas_mets,
                 'gas_ms': gas_ms, 'gal_smls': gal_smls, 'gas_smls': gas_smls, 'all_gas_poss': all_gas_poss,
                 'all_gal_poss': all_gal_poss, 'means': means}

    return save_dict


def plot_meidan_stat(xs, ys, ax, bins=None):

    if bins == None:
        bin = np.logspace(np.log10(xs.min()), np.log10(xs.max()), 20)
    else:
        bin = bins

    # Compute binned statistics
    y_stat, binedges, bin_ind = binned_statistic(xs, ys, statistic='median', bins=bin)

    # Compute bincentres
    bin_wid = binedges[1] - binedges[0]
    bin_cents = binedges[1:] - bin_wid / 2

    okinds = np.logical_and(~np.isnan(bin_cents), ~np.isnan(y_stat))

    ax.plot(bin_cents[okinds], y_stat[okinds], color='r', linestyle='-')


def get_lumins(gal_poss, gal_ms, gal_ages, gal_mets, gas_mets, gas_poss, gas_ms, gas_sml,
               lkernel, kbins, conv, model, F, i, j, f, dust):

    if f == 'mass':
        lumins = gal_ms

    elif dust:

        # Define dimensions array
        if i == 0 and j == 1:
            k = 2
        elif i == 0 and j == 2:
            k = 1
        else:
            k = 0
        dimens = np.array([i, j, k])

        gal_met_surfden = get_Z_LOS(gal_poss, gas_poss, gas_ms, gas_mets, gas_sml, dimens, lkernel, kbins, conv)
        tauVs_ISM = (10 ** 5.2) * gal_met_surfden
        tauVs_BC = 2.0 * (gal_mets / 0.01)
        lumins = models.generate_Lnu_array(model, gal_ms, gal_ages, gal_mets, tauVs_ISM, tauVs_BC, F,
                                           f=f, fesc=0.0)
    else:
        lumins = models.generate_Lnu_array(model, gal_ms, gal_ages, gal_mets, np.zeros_like(gal_mets),
                                           np.zeros_like(gal_mets), F, f=f, fesc=0.0)

    return lumins


@nb.njit(nogil=True, parallel=True)
def calc_light_mass_rad(poss, ls, ms):

    # Get galaxy particle indices
    rs = np.sqrt(poss[:, 0]**2 + poss[:, 1]**2 + poss[:, 2]**2)

    # Sort the radii and masses
    sinds = np.argsort(rs)
    rs = rs[sinds]
    ls = ls[sinds]

    # Get the cumalative sum of masses
    l_profile = np.cumsum(ls)

    # Get the total mass and half the total mass
    tot_l = l_profile[-1]
    half_l = tot_l / 2

    # Get the half mass radius particle
    hmr_ind = np.argmin(np.abs(l_profile - half_l))
    hmr = rs[hmr_ind]

    return hmr, np.sum(ms)


def hl_main(snap, model, F, f, conv=1, i=0, j=1, dust=False):

    # Get the redshift
    z_str = snap.split('z')[1].split('p')
    z = float(z_str[0] + '.' + z_str[1])

    model.create_Lnu_grid(F)

    kinp = np.load('/cosma/home/dp004/dc-rope1/cosma7/FLARES/flares/los_extinction/kernel_sph-anarchy.npz',
                   allow_pickle=True)
    lkernel = kinp['kernel']
    header = kinp['header']
    kbins = header.item()['bins']

    path = '/cosma7/data//Eagle/ScienceRuns/Planck1/L0100N1504/PE/REFERENCE/data'

    save_dict = get_main(path, snap)

    gal_ages = save_dict['gal_ages']
    gal_mets = save_dict['gal_mets']
    gal_ms = save_dict['gal_ms']
    gas_mets = save_dict['gas_mets']
    gas_ms = save_dict['gas_ms']
    gas_smls = save_dict['gas_smls']
    all_gas_poss = save_dict['all_gas_poss']
    all_gal_poss = save_dict['all_gal_poss']
    means = save_dict['means']

    print('Extracted galaxy positions for', len(gal_ages.keys()), 'galaxies')

    # Create images for these galaxies
    hls = np.zeros(len(gal_ages))
    ms = np.zeros(len(gal_ages))
    for ind, id in enumerate(gal_ages.keys()):

        # print('Computing luminosities for', id, f)

        # Get the luminosities
        if len(gas_ms[id]) == 0:
            continue
        if np.sum(gal_ms[id]) < 1e8:
            continue
        gas_poss = all_gas_poss[id]
        gal_poss = all_gal_poss[id]
        # means[id] = np.mean(gal_poss, axis=0)
        gal_poss -= means[id]
        gas_poss -= means[id]
        ls = get_lumins(gal_poss, gal_ms[id], gal_ages[id], gal_mets[id], gas_mets[id],
                        gas_poss, gas_ms[id], gas_smls[id], lkernel, kbins, conv, model,
                        F, i, j, f, dust)

        # Compute half mass radii
        hls[ind], ms[ind] = calc_light_mass_rad(gal_poss, ls, gal_ms[id])
        # print(hls[ind])

    return hls[hls > 0.0], ms[hls > 0.0]


regions = []
reg_ints = list(range(0, 40))
# reg_ints.append(39)
for reg in reg_ints:
    if reg < 10:
        regions.append('0' + str(reg))
    else:
        regions.append(str(reg))

# fs = ['mass', 'FAKE.TH.V', 'FAKE.TH.NUV', 'FAKE.TH.FUV']
fs = ['mass']
conv = (u.solMass / u.Mpc ** 2).to(u.g / u.cm ** 2)
ii, jj = 0, 1
dust = False

snaps = ['004_z008p075', '008_z005p037', '010_z003p984',
         '013_z002p478', '017_z001p487', '018_z001p259',
         '019_z001p004', '020_z000p865', '024_z000p366']
axlims_x = []
axlims_y = []

for f in fs:
    half_mass_rads_dict = {}
    xaxis_dict = {}
    for snap in snaps:
        half_mass_rads_dict[snap] = {}
        xaxis_dict[snap] = {}

    for snap in snaps:

        print(snap)
        try:
            half_mass_rads_dict[snap], xaxis_dict[snap] = hl_main(snap, model, F, f,
                                                                  conv=conv, i=ii, j=jj, dust=dust)
        except FileNotFoundError:
            continue


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

        z_str = snap.split('z')[1].split('p')
        z = float(z_str[0] + '.' + z_str[1])

        # Convert inputs to physical kpc
        convert_pMpc = 1 / (1 + z)

        # Define comoving softening length in kpc
        csoft = 0.001802390 / 0.677 * convert_pMpc

        xs = np.concatenate(list(xaxis_dict[snap].values()))
        half_mass_rads_plt = np.concatenate(list(half_mass_rads_dict[snap].values()))

        xs_plt = xs[half_mass_rads_plt > 0]
        half_mass_rads_plt = half_mass_rads_plt[half_mass_rads_plt > 0]
        half_mass_rads_plt = half_mass_rads_plt[xs_plt > 1e8]
        xs_plt = xs_plt[xs_plt > 1e8]

        if len(xs_plt) > 0:
            cbar = ax.hexbin(xs_plt, half_mass_rads_plt / csoft, gridsize=100, mincnt=1, xscale='log', yscale='log',
                             norm=LogNorm(),
                             linewidths=0.2, cmap='viridis')
            plot_meidan_stat(xs_plt, half_mass_rads_plt / csoft, ax)

        ax.text(0.8, 0.9, f'$z={z}$', bbox=dict(boxstyle="round,pad=0.3", fc='w', ec="k", lw=1, alpha=0.8),
                transform=ax.transAxes, horizontalalignment='right', fontsize=8)

        axlims_x.extend(ax.get_xlim())
        axlims_y.extend(ax.get_ylim())

        # Label axes
        if i == 2:
            ax.set_xlabel(r'$M_{\mathrm{\star}}/M_\odot$')
        if j == 0:
            ax.set_ylabel('$R_{1/2,\mathrm{\star}}/\epsilon$')

    for ax in [ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8, ax9]:
        ax.set_xlim(np.min(axlims_x), np.max(axlims_x))
        ax.set_ylim(np.min(axlims_y), np.max(axlims_y))

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

    if f != 'mass':
        fig.savefig('plots/HalfLightRadius_all_snaps_' + f + '_coords' + str(ii) + '-' + str(jj) + 'REF.png',
                    bbox_inches='tight')
    else:
        fig.savefig('plots/HalfMassRadius_all_snaps_coords' + str(ii) + '-' + str(jj) + 'REF.png',
                    bbox_inches='tight')

    plt.close(fig)
