#!/cosma/home/dp004/dc-rope1/.conda/envs/flares-env/bin/python
import matplotlib as ml
ml.use('Agg')
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LogNorm
import eagle_IO as E
import h5py
import sys
import seaborn as sns


sns.set_style('whitegrid')


def get_change_in_radius(snap, prog_snap, savepath, gal_data, gals):

    # Open graph file
    hdf = h5py.File(savepath + 'SubMgraph_' + snap + '.hdf5', 'r')

    # Initialise arrays for results
    delta_hmrs = np.zeros(len(gals))
    delta_ms = np.zeros(len(gals))
    major_minor = np.zeros(len(gals))
    wet_dry = np.zeros(len(gals))

    print('There are', len(gals), 'Galaxies to test in snapshot', snap)

    # Loop over galaxies
    for ind, i in enumerate(gals):

        # Get this halo's stellar mass and half mass radius
        mass, hmr = gal_data[snap][i]['m'], gal_data[snap][i]['hmr']

        # Get progenitors
        try:
            progs = hdf[str(i)]['Prog_haloIDs'][...]
            progs = progs[hdf[str(i)]['prog_stellar_mass_contribution'][...] * 10**10 > 0]
        except KeyError:
            # Define change in properties
            delta_hmrs[ind] = 2**30
            delta_ms[ind] = 2**30
            major_minor[ind] = 2**30
            wet_dry[ind] = 2**30
            print(i, ind, "Galaxy has no dark matter")
            continue

        if len(progs) != 2:

            # Define change in properties
            delta_hmrs[ind] = 2**30
            delta_ms[ind] = 2**30
            major_minor[ind] = 2**30
            wet_dry[ind] = 2**30

        else:

            # Get progenitor properties
            try:
                prog_cont = hdf[str(i)]['prog_npart_contribution'][...]
                prog_masses = hdf[str(i)]['prog_stellar_mass_contribution'][...] * 10**10
                prog_hmrs = np.array([gal_data[prog_snap][p]['hmr'] for p in progs])
                prog_gas_mass = hdf[str(i)]['Prog_gas_mass'][...] * 10**10
                prog_stellar_mass = hdf[str(i)]['Prog_stellar_mass'][...] * 10**10
            except KeyError:
                # Define change in properties
                delta_hmrs[ind] = 2 ** 30
                delta_ms[ind] = 2 ** 30
                major_minor[ind] = 2 ** 30
                wet_dry[ind] = 2 ** 30
                print(i, ind, "Galaxy was missing a dataset")
                continue

            # Get main progenitor information
            main = np.argmax(prog_cont)
            main_stellar_mass = prog_stellar_mass[main]
            main_hmr = prog_hmrs[main]

            # Define change in properties
            major_minor[ind] = (np.sum(prog_stellar_mass) - main_stellar_mass) / main_stellar_mass
            wet_dry[ind] = np.sum(prog_gas_mass) / np.sum(prog_stellar_mass)
            delta_hmrs[ind] = hmr / main_hmr
            delta_ms[ind] = mass / np.sum(prog_masses)

    hdf.close()

    return delta_hmrs[delta_ms < 2**30], delta_ms[delta_ms < 2**30], \
           wet_dry[delta_ms < 2**30], major_minor[delta_ms < 2**30]


def main_change(masslim=1e8, hmrcut=False):

    regions = []
    for reg in range(0, 2):
        if reg < 10:
            regions.append('0' + str(reg))
        else:
            regions.append(str(reg))

    delta_hmr_dict = {}
    delta_ms_dict = {}
    wet_dry_dict= {}
    major_minor_dict = {}

    # Define snapshots
    snaps = ['001_z014p000', '002_z013p000', '003_z012p000', '004_z011p000', '005_z010p000',
             '006_z009p000', '007_z008p000', '008_z007p000', '009_z006p000', '010_z005p000', '011_z004p770']
    prog_snaps = ['000_z015p000', '001_z014p000', '002_z013p000', '003_z012p000', '004_z011p000', '005_z010p000',
                  '006_z009p000', '007_z008p000', '008_z007p000', '009_z006p000', '010_z005p000']

    for snap, prog_snap in zip(snaps, prog_snaps):

        delta_hmr_dict[snap] = {}
        delta_ms_dict[snap] = {}
        wet_dry_dict[snap] = {}
        major_minor_dict[snap] = {}

        for reg in regions:

            savepath = '/cosma/home/dp004/dc-rope1/FLARES/FLARES-1/MergerGraphs/GEAGLE_' + reg + '/'

            path = '/cosma/home/dp004/dc-rope1/FLARES/FLARES-1/G-EAGLE_' + reg + '/data'

            # Get halo IDs and halo data
            try:
                subgrp_ids = E.read_array('SUBFIND', path, snap, 'Subhalo/SubGroupNumber', numThreads=8)
                grp_ids = E.read_array('SUBFIND', path, snap, 'Subhalo/GroupNumber', numThreads=8)
                gal_hmrs = E.read_array('SUBFIND', path, snap, 'Subhalo/HalfMassRad', noH=True,
                                        physicalUnits=True, numThreads=8)[:, 4]
                gal_ms = E.read_array('SUBFIND', path, snap, 'Subhalo/ApertureMeasurements/Mass/030kpc',
                                      noH=False, physicalUnits=False, numThreads=8)[:, 4] * 10**10

                # Get halo IDs and halo data
                prog_subgrp_ids = E.read_array('SUBFIND', path, prog_snap, 'Subhalo/SubGroupNumber', numThreads=8)
                prog_grp_ids = E.read_array('SUBFIND', path, prog_snap, 'Subhalo/GroupNumber', numThreads=8)
                prog_gal_hmrs = E.read_array('SUBFIND', path, prog_snap, 'Subhalo/HalfMassRad', noH=True,
                                        physicalUnits=True, numThreads=8)[:, 4]
                prog_gal_ms = E.read_array('SUBFIND', path, prog_snap, 'Subhalo/ApertureMeasurements/Mass/030kpc',
                                           noH=False, physicalUnits=False, numThreads=8)[:, 4] * 10**10
            except ValueError:
                continue
            except OSError:
                continue

            z_str = snap.split('z')[1].split('p')
            z = float(z_str[0] + '.' + z_str[1])

            # Convert inputs to physical kpc
            convert_pMpc = 1 / (1 + z)

            # Define comoving softening length in kpc
            csoft = 0.001802390 / 0.677 * convert_pMpc

            # Remove particles not associated to a subgroup
            if hmrcut:
                okinds = np.logical_and(subgrp_ids != 1073741824, np.logical_and(gal_ms > 0, gal_hmrs / csoft < 1.2))
            else:
                okinds = np.logical_and(subgrp_ids != 1073741824, gal_ms > 0)
            gal_hmrs = gal_hmrs[okinds]
            gal_ms = gal_ms[okinds]
            grp_ids = grp_ids[okinds]
            subgrp_ids = subgrp_ids[okinds]
            halo_ids = np.zeros(grp_ids.size, dtype=float)
            for (ind, g), sg in zip(enumerate(grp_ids), subgrp_ids):
                halo_ids[ind] = float(str(int(g)) + '.%05d'%int(sg))

            # Remove particles not associated to a subgroup
            okinds = prog_subgrp_ids != 1073741824
            prog_gal_hmrs = prog_gal_hmrs[okinds]
            prog_gal_ms = prog_gal_ms[okinds]
            prog_grp_ids = prog_grp_ids[okinds]
            prog_subgrp_ids = prog_subgrp_ids[okinds]
            prog_ids = np.zeros(prog_grp_ids.size, dtype=float)
            for (ind, g), sg in zip(enumerate(prog_grp_ids), prog_subgrp_ids):
                prog_ids[ind] = float(str(int(g)) + '.%05d'%int(sg))

            # Initialise galaxy data
            gal_data = {snap: {}, prog_snap: {}}
            for m, hmr, i in zip(gal_ms, gal_hmrs, halo_ids):
                gal_data[snap][i] = {'m': m, 'hmr': hmr}
            for m, hmr, i in zip(prog_gal_ms, prog_gal_hmrs, prog_ids):
                gal_data[prog_snap][i] = {'m': m, 'hmr': hmr}

            # Get change in stellar mass and half mass radius
            try:
                results_tup = get_change_in_radius(snap, prog_snap, savepath, gal_data, halo_ids[gal_ms > masslim])
                delta_hmr_dict[snap][reg] = results_tup[0]
                delta_ms_dict[snap][reg] = results_tup[1]
                wet_dry_dict[snap][reg] = results_tup[2]
                major_minor_dict[snap][reg] = results_tup[3]
            except OSError:
                continue

    delta_hmr = []
    delta_mass = []
    major_minor = []
    wet_dry = []
    for snap in snaps:
        delta_hmr.append(np.concatenate(list(delta_hmr_dict[snap].values())))
        delta_mass.append(np.concatenate(list(delta_ms_dict[snap].values())))
        major_minor.append(np.concatenate(list(major_minor_dict[snap].values())))
        wet_dry.append(np.concatenate(list(wet_dry_dict[snap].values())))

    delta_hmr = np.concatenate(delta_hmr)
    delta_mass = np.concatenate(delta_mass)
    major_minor = np.concatenate(major_minor)
    wet_dry = np.concatenate(wet_dry)

    okinds = np.logical_and(major_minor > 0, wet_dry > 0)
    delta_hmr = delta_hmr[okinds]
    delta_mass = delta_mass[okinds]
    major_minor = major_minor[okinds]
    wet_dry = wet_dry[okinds]

    # Define limits
    majlowlim = 0.5
    minlowlim = 0.05
    wetlowlim = 10
    dryuplim = 1

    # Set up plot
    fig = plt.figure()
    ax = fig.add_subplot(111)

    # Plot results
    cbar = ax.hexbin(major_minor, wet_dry, gridsize=100, mincnt=1, xscale='log', yscale='log',
                     norm=LogNorm(), linewidths=0.2, cmap='viridis')

    # Label axes
    ax.set_xlabel(r'$M_{secondary}/M_{primary}$')
    ax.set_ylabel('$f_{\mathrm{gas,merger}}$')

    fig.colorbar(cbar, ax=ax)

    fig.savefig('plots/wetdrymajorminormerger.png', bbox_inches='tight')

    plt.close()

    # Set up plot
    fig = plt.figure()
    ax = fig.add_subplot(111)

    H, bin_edges = np.histogram(np.log10(major_minor)[np.where(np.log10(major_minor) < np.inf)],
                                bins=int(np.sqrt(len(major_minor))))

    bin_wid = bin_edges[1] - bin_edges[0]
    bin_cents = bin_edges[1:] - (bin_wid / 2)

    # Plot results
    ax.bar(bin_cents, np.log10(H), color='k', edgecolor='k', alpha=0.6, width=bin_wid)
    ax.axvline(np.log10(majlowlim))

    # Label axes
    ax.set_xlabel(r'$\log_{10}(M_{secondary}/M_{primary})$')
    ax.set_ylabel('$\log_{10}(N)$')

    # fig.colorbar(cbar, ax=ax)

    fig.savefig('plots/majorminormerger_hist.png', bbox_inches='tight')

    plt.close()

    # Set up plot
    fig = plt.figure()
    ax = fig.add_subplot(111)

    H, bin_edges = np.histogram(np.log10(wet_dry)[np.where(np.log10(wet_dry) < np.inf)], bins=100)

    bin_wid = bin_edges[1] - bin_edges[0]
    bin_cents = bin_edges[1:] - (bin_wid / 2)

    # Plot results
    ax.bar(bin_cents, np.log10(H), color='k', edgecolor='k', alpha=0.6, width=bin_wid)

    # Label axes
    ax.set_xlabel(r'$\log_{10}(f_{\mathrm{gas,merger}})$')
    ax.set_ylabel('$\log_{10}(N)$')

    fig.savefig('plots/wetdrymerger_hist.png', bbox_inches='tight')

    plt.close()

    # Get major/minor/accretion and wet/mix/dry divisions
    maj_wet_inds = np.logical_and(major_minor >= majlowlim, wet_dry >= wetlowlim)
    min_wet_inds = np.logical_and(np.logical_and(major_minor < majlowlim, major_minor >= minlowlim), wet_dry >= wetlowlim)
    acc_wet_inds = np.logical_and(major_minor < minlowlim, wet_dry >= wetlowlim)
    maj_dry_inds = np.logical_and(major_minor >= majlowlim, wet_dry <= dryuplim)
    min_dry_inds = np.logical_and(np.logical_and(major_minor < majlowlim, major_minor >= minlowlim), wet_dry <= dryuplim)
    acc_dry_inds = np.logical_and(major_minor < minlowlim, wet_dry <= dryuplim)
    maj_mix_inds = np.logical_and(major_minor >= majlowlim, np.logical_and(wet_dry < wetlowlim, wet_dry > dryuplim))
    min_mix_inds = np.logical_and(np.logical_and(major_minor < majlowlim, major_minor >= minlowlim), np.logical_and(wet_dry < wetlowlim, wet_dry > dryuplim))
    acc_mix_inds = np.logical_and(major_minor < minlowlim, np.logical_and(wet_dry < wetlowlim, wet_dry > dryuplim))

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

    axlims_x = []
    axlims_y = []
    axes = [ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8, ax9]
    gridrefs = [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2), (2, 0), (2, 1), (2, 2)]
    inds = [maj_wet_inds, min_wet_inds, acc_wet_inds,
            maj_mix_inds, min_mix_inds, acc_mix_inds,
            maj_dry_inds, min_dry_inds, acc_dry_inds]
    labels = ['Major-Wet', 'Minor-Wet', 'Accretion-Wet',
              'Major-Mix', 'Minor-Mix', 'Accretion-Mix',
              'Major-Dry', 'Minor-Dry', 'Accretion-Dry']

    for ax, (i, j), ind, lab in zip(axes, gridrefs, inds, labels):

        xs_plt = delta_mass[ind]
        delta_hmr_plt = delta_hmr[ind]

        if len(xs_plt) > 0:
            cbar = ax.hexbin(xs_plt, delta_hmr_plt, gridsize=100, mincnt=1, xscale='log', yscale='log',
                             norm=LogNorm(),
                             linewidths=0.2, cmap='viridis')

        ax.text(0.8, 0.9, lab, bbox=dict(boxstyle="round,pad=0.3", fc='w', ec="k", lw=1, alpha=0.8),
                transform=ax.transAxes, horizontalalignment='right', fontsize=8)

        axlims_x.extend(ax.get_xlim())
        axlims_y.extend(ax.get_ylim())

        # Label axes
        if i == 2:
            ax.set_xlabel(r'$M_{\star}/M_{\star, \mathrm{from progs}}$')
        if j == 0:
            ax.set_ylabel('$R_{1/2,\mathrm{\star}}/R_{1/2,\mathrm{\star},\mathrm{main prog}}$')

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

    fig.savefig('plots/change_in_halfmassradius_mergersplit.png', bbox_inches='tight')


regions = []
for reg in range(0, 40):
    if reg < 10:
        regions.append('0' + str(reg))
    else:
        regions.append(str(reg))

main_change(masslim=10**8)

