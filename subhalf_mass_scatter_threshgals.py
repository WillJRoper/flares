#!/cosma/home/dp004/dc-rope1/.conda/envs/flares-env/bin/python
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import astropy.units as u
from matplotlib.colors import LogNorm
import matplotlib.gridspec as gridspec
import eagle_IO as E
import seaborn as sns
import h5py
import pickle
import itertools
matplotlib.use('Agg')

sns.set_style('whitegrid')


regions = []
for reg in range(0, 1):

    if reg < 10:
        regions.append('000' + str(reg))
    else:
        regions.append('00' + str(reg))

gregions = []
for reg in range(0, 1):

    if reg < 10:
        gregions.append('0' + str(reg))
    else:
        gregions.append(str(reg))


snaps = ['003_z012p000', '004_z011p000', '005_z010p000',
         '006_z009p000', '007_z008p000', '008_z007p000',
         '009_z006p000', '010_z005p000', '011_z004p770']

gsnaps = list(reversed(['000_z015p000', '001_z014p000', '002_z013p000', '003_z012p000', '004_z011p000', '005_z010p000',
                        '006_z009p000', '007_z008p000', '008_z007p000', '009_z006p000', '010_z005p000', '011_z004p770']))
axlims_x = []
axlims_y = []

# Define thresholds for roots
mthresh = 10**9
rthresh = 1.1

half_mass_rads_dict = {}
xaxis_dict = {}
ms = {}
rs = {}
for snap in snaps:

    half_mass_rads_dict[snap] = {}
    xaxis_dict[snap] = {}
    ms[snap] = {}
    rs[snap] = {}

for reg in regions:

    for snap in snaps:

        print(reg, snap)

        path = '/cosma7/data/dp004/dc-love2/data/G-EAGLE/geagle_' + reg + '/data/'

        half_mass_rads_dict[snap][reg] = E.read_array('SUBFIND', path, snap, 'Subhalo/HalfMassRad', noH=True,
                                                      numThreads=8)[:, 4] * 1e3
        xaxis_dict[snap][reg] = E.read_array('SUBFIND', path, snap, 'Subhalo/ApertureMeasurements/Mass/030kpc',
                                             noH=True, numThreads=8)[:, 4] * 10**10
        subgroup_ids = E.read_array('SUBFIND', path, snap, 'Subhalo/SubGroupNumber', numThreads=8)
        group_ids = E.read_array('SUBFIND', path, snap, 'Subhalo/GroupNumber', numThreads=8)
        halo_ids = np.zeros_like(group_ids, dtype=float)
        for (ind, g), sg in zip(enumerate(group_ids), subgroup_ids):
            halo_ids[ind] = float(str(g) + '.' + str(sg))

        ms[snap][reg] = {}
        rs[snap][reg] = {}
        for simid, m, r in zip(halo_ids, xaxis_dict[snap][reg], half_mass_rads_dict[snap][reg]):
            ms[snap][reg][simid] = m
            rs[snap][reg][simid] = r

# Get halos which are in the distribution at the z=4.77
halos_in_pop = {}
for reg in regions:
    for grp in ms['011_z004p770'][reg].keys():
        # if ms['011_z004p770'][reg][grp] > 0:
        #     print(ms['011_z004p770'][reg][grp], thresh)
        if ms['011_z004p770'][reg][grp] >= mthresh and rs['011_z004p770'][reg][grp] <= rthresh:
            halos_in_pop.setdefault(reg, []).append(grp)
print(halos_in_pop)
# Get the halos from the graph that make up these halos
halos_included = {}
for reg, greg in zip(regions, gregions):
    halos_included[reg] = {}
    for grp in halos_in_pop[reg]:
        halos = [grp, ]
        print(grp, list(gsnaps))
        for snap in list(gsnaps):

            print(reg, grp, snap)

            # Add halos to dictionary
            halos_included[reg].setdefault(snap, set()).update(set(halos))

            hdf = h5py.File('/cosma/home/dp004/dc-rope1/FLARES/FLARES-1/MergerGraphs/GEAGLE_' + greg +
                            '/SubMgraph_' + snap + '_PartType1.hdf5', 'r')
            progs = np.concatenate([hdf[str(halo)]['Prog_haloIDs'][...] for halo in halos])
            halos = list(set(progs))
            hdf.close()

ms_plt = {}
rs_plt = {}
for snap in snaps:
    ms_plt[snap] = []
    rs_plt[snap] = []
    for reg in regions:
        for grp in halos_included[reg][snap]:
            # print(snap, reg, grp)
            ms_plt[snap].append(ms[snap][reg][grp])
            rs_plt[snap].append(rs[snap][reg][grp])

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

    thrsh_ms = np.array(ms_plt[snap])
    thrsh_rs = np.array(rs_plt[snap])
    
    thrsh_ms_plt = thrsh_ms[thrsh_rs > 0]
    thrsh_rs = thrsh_rs[thrsh_rs > 0]
    thrsh_rs = thrsh_rs[thrsh_ms_plt > 1e8]
    thrsh_ms_plt = thrsh_ms_plt[thrsh_ms_plt > 1e8]
    
    xs_plt = xs[half_mass_rads_plt > 0]
    half_mass_rads_plt = half_mass_rads_plt[half_mass_rads_plt > 0]
    half_mass_rads_plt = half_mass_rads_plt[xs_plt > 1e8]
    xs_plt = xs_plt[xs_plt > 1e8]
    
    cbar = ax.hexbin(xs_plt, half_mass_rads_plt, gridsize=100, mincnt=1, xscale='log', yscale='log', norm=LogNorm(),
                     linewidths=0.2, cmap='Greys', alpha=0.6)
    ax.hexbin(thrsh_ms_plt, thrsh_rs, gridsize=50, mincnt=1, xscale='log', yscale='log', norm=LogNorm(),
              linewidths=0.2, cmap='viridis')

    ax.text(0.8, 0.9, f'$z={z}$', bbox=dict(boxstyle="round,pad=0.3", fc='w', ec="k", lw=1, alpha=0.8),
            transform=ax.transAxes, horizontalalignment='right', fontsize=8)

    axlims_x.extend(ax.get_xlim())
    axlims_y.extend(ax.get_ylim())

    # Label axes
    if i == 2:
        ax.set_xlabel(r'$M_{\star}/M_\odot$')
    if j == 0:
        ax.set_ylabel('$R_{1/2}/$ckpc')

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

fig.savefig('plots/HalfMassRadius_all_snaps_compact.png', dpi=300,
            bbox_inches='tight')

plt.close(fig)
