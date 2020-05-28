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
from astropy.cosmology import Planck13 as cosmo
import seaborn as sns
os.environ['FLARE'] = '/cosma7/data/dp004/dc-wilk2/flare'
import FLARE.filters
from scipy.stats import binned_statistic
from SynthObs.SED import models
matplotlib.use('Agg')


sns.set_style('whitegrid')

# Define SED model
model = models.define_model('BPASSv2.2.1.binary/ModSalpeter_300',
                            path_to_SPS_grid = FLARE.FLARE_dir + '/data/SPS/nebular/3.0/') # DEFINE SED GRID -
model.dust_ISM = ('simple', {'slope': -1.0})
model.dust_BC = ('simple', {'slope': -1.0})
filters = FLARE.filters.NIRCam
F = FLARE.filters.add_filters(filters, new_lam = model.lam)


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
               lkernel, kbins, conv, model, F, i, j, f):

    # Define dimensions array
    if i == 0 and j == 1:
        k = 2
    elif i == 0 and j == 2:
        k = 1
    else:
        k = 0
    dimens = np.array([i, j, k])

    gal_met_surfden = get_Z_LOS(gal_poss, gas_poss, gas_ms, gas_mets, gas_sml, dimens, lkernel, kbins, conv)

    # Calculate optical depth of ISM and birth cloud
    tauVs_ISM = (10 ** 5.2) * gal_met_surfden
    tauVs_BC = 2.0 * (gal_mets / 0.01)

    # Extract the flux in nanoJansky
    L = (models.generate_Fnu_array(model, gal_ms, gal_ages, gal_mets, tauVs_ISM,
                                   tauVs_BC, F, 'JWST.NIRCAM.' + f) * u.nJy).decompose()

    return L


@nb.njit(nogil=True, parallel=True)
def calc_light_mass_rad(poss, ls, i, j):

    # Get galaxy particle indices
    rs = np.sqrt(poss[:, i]**2 + poss[:, j]**2)

    # Sort the radii and masses
    sinds = np.argsort(rs)
    rs = rs[sinds]
    ls = ls[sinds]
    ls = ls[rs < 30/1e3]
    rs = rs[rs < 30/1e3]

    if len(ls) < 20:
        return 0.0

    # Get the cumalative sum of masses
    l_profile = np.cumsum(ls)

    # Get the total mass and half the total mass
    tot_l = l_profile[-1]
    half_l = tot_l / 2

    # Get the half mass radius particle
    hmr_ind = np.argmin(np.abs(l_profile - half_l))
    hmr = rs[hmr_ind]

    return hmr


def hl_main(snap, reg, model, F, f, npart_lim=0, conv=1, i=0, j=1):

    # Get the redshift
    z_str = snap.split('z')[1].split('p')
    z = float(z_str[0] + '.' + z_str[1])

    model.create_Fnu_grid(F, z, cosmo)

    kinp = np.load('/cosma/home/dp004/dc-rope1/cosma7/FLARES/flares/los_extinction/kernel_sph-anarchy.npz',
                   allow_pickle=True)
    lkernel = kinp['kernel']
    header = kinp['header']
    kbins = header.item()['bins']

    if npart_lim > 0:
        with open('UVimg_data/stellardata_reg' + reg + '_snap'
                  + snap + '_npartgreaterthan' + str(npart_lim) + '.pck', 'rb') as pfile1:
            save_dict = pickle.load(pfile1)
    else:
        with open('UVimg_data/stellardata_reg' + reg + '_snap' + snap + '.pck', 'rb') as pfile1:
            save_dict = pickle.load(pfile1)

    gal_ages = save_dict['gal_ages']
    gal_mets = save_dict['gal_mets']
    gal_ms = save_dict['gal_ms']
    gas_mets = save_dict['gas_mets']
    gas_ms = save_dict['gas_ms']
    gas_smls = save_dict['gas_smls']
    all_gas_poss = save_dict['all_gas_poss']
    all_gal_poss = save_dict['all_gal_poss']
    means = save_dict['means']

    print('Extracted galaxy positions')

    # Create images for these galaxies
    hls = np.zeros(len(gal_ages))
    ms = np.zeros(len(gal_ages))
    for ind, id in enumerate(gal_ages.keys()):

        if len(gal_ages[id]) == 0:
            continue

        # Get the luminosities
        try:
            # print(all_gal_poss[id])
            ls = get_lumins(all_gal_poss[id] - means[id], gal_ms[id], gal_ages[id], gal_mets[id], gas_mets[id],
                            all_gas_poss[id] - means[id], gas_ms[id], gas_smls[id], lkernel, kbins, conv, model,
                            F, i, j, f)

            # Compute half mass radii
            hls[ind] = calc_light_mass_rad(all_gal_poss[id] - means[id], ls, i, j)
            ms[ind] = np.sum(gal_ms[id])

        except KeyError:
            print("Galaxy", id, "did not appear in the dictionary")
            continue

    return hls[ms > 0], ms[ms > 0]


regions = []
for reg in range(0, 40):

    if reg < 10:
        regions.append('0' + str(reg))
    else:
        regions.append(str(reg))
# regions = []
# reg_ints = list(range(39, -1, -1))
# print(reg_ints)
# reg_ints.append(39)
# for reg in reg_ints:
#     if reg < 10:
#         regions.append('0' + str(reg))
#     else:
#         regions.append(str(reg))

fs = ['F150W', 'F200W', 'F444W']
conv = (u.solMass / u.Mpc ** 2).to(u.g / u.cm ** 2)
ii, jj = 0, 1

snaps = ['003_z012p000', '004_z011p000', '005_z010p000',
         '006_z009p000', '007_z008p000', '008_z007p000',
         '009_z006p000', '010_z005p000', '011_z004p770']
axlims_x = []
axlims_y = []

# Define comoving softening length in Mpc
csoft = 0.001802390 / 0.677
for f in fs:
    half_mass_rads_dict = {}
    xaxis_dict = {}
    for snap in snaps:
        half_mass_rads_dict[snap] = {}
        xaxis_dict[snap] = {}

    for reg in regions:

        for snap in snaps:

            print(reg, snap)
            try:
                half_mass_rads_dict[snap][reg], xaxis_dict[snap][reg] = hl_main(snap, reg, model, F, f,
                                                                                conv=conv, i=ii, j=jj)
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

        xs = np.concatenate(list(xaxis_dict[snap].values()))
        half_mass_rads_plt = np.concatenate(list(half_mass_rads_dict[snap].values()))

        xs_plt = xs[half_mass_rads_plt > 0]
        half_mass_rads_plt = half_mass_rads_plt[half_mass_rads_plt > 0]
        half_mass_rads_plt = half_mass_rads_plt[xs_plt > 1e8]
        xs_plt = xs_plt[xs_plt > 1e8]

        if len(xs_plt) > 0:
            cbar = ax.hexbin(xs_plt, half_mass_rads_plt / (csoft / (1 + z)), gridsize=100, mincnt=1, xscale='log',
                             yscale='log', norm=LogNorm(), linewidths=0.2, cmap='viridis')
            plot_meidan_stat(xs_plt, half_mass_rads_plt / (csoft / (1 + z)), ax)

        ax.text(0.8, 0.1, f'$z={z}$', bbox=dict(boxstyle="round,pad=0.3", fc='w', ec="k", lw=1, alpha=0.8),
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

    fig.savefig('plots/HalfLightRadius_all_snaps_' + f + '_coords' + str(ii) + '-' + str(jj) + '.png',
                bbox_inches='tight')

    plt.close(fig)
