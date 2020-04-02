#!/cosma/home/dp004/dc-rope1/.conda/envs/flares-env/bin/python
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import numba as nb
from matplotlib.colors import LogNorm
import matplotlib.gridspec as gridspec
from scipy.spatial import cKDTree
import eagle_IO as E
import seaborn as sns

matplotlib.use('Agg')

sns.set_style('whitegrid')


# @nb.njit(nogil=True, parallel=True)
def get_parts_in_aperture(all_poss, mets, cent, tree, app):

    # Get galaxy particle indices
    query = tree.query_ball_point(cent, r=app)

    # Get galaxy particle indices
    seps = all_poss[query, :] - cent
    rs2 = seps[:, 0] ** 2 + seps[:, 1] ** 2 + seps[:, 2] ** 2

    # Get particle positions and masses
    gal_rs = np.sqrt(rs2)
    gal_mets = mets[query]

    return gal_rs, gal_mets


# @nb.jit(nogil=True, parallel=True)
def get_r_and_met(all_poss, mets, gal_cops, gal_hmr, tree):

    # Loop over galaxies centres
    rs_dict = {}
    mets_dict = {}
    for (ind, cop), hmr in zip(enumerate(gal_cops), gal_hmr):

        # Get particles and masses
        gal_rs, gal_mets = get_parts_in_aperture(all_poss, mets, cop, tree, app=0.03)
        if len(gal_rs) != 0:
            rs_dict[ind], mets_dict[ind] = gal_rs / hmr, gal_mets

    return np.concatenate(list(mets_dict.values())), np.concatenate(list(rs_dict.values()))


regions = []
for reg in range(0, 2):

    if reg < 10:
        regions.append('000' + str(reg))
    else:
        regions.append('00' + str(reg))

snaps = ['003_z012p000', '004_z011p000', '005_z010p000',
         '006_z009p000', '007_z008p000', '008_z007p000',
         '009_z006p000', '010_z005p000', '011_z004p770']
axlims_x = []
axlims_y = []

# Define comoving softening length in kpc
csoft = 0.001802390 / 0.677

# Define part type
part_type = 4

metallicity_dict = {}
rads_dict = {}
for snap in snaps:
    metallicity_dict[snap] = {}
    rads_dict[snap] = {}

for reg in regions:

    for snap in snaps:

        print(reg, snap)

        z_str = snap.split('z')[1].split('p')
        z = float(z_str[0] + '.' + z_str[1])

        path = '/cosma7/data/dp004/dc-love2/data/G-EAGLE/geagle_' + reg + '/data/'
        try:
            all_poss = E.read_array('SNAP', path, snap, 'PartType0/Coordinates', noH=True,
                                    physicalUnits=True, numThreads=8)
            mets = E.read_array('SNAP', path, snap, 'PartType0/Metallicity', noH=True,
                                  physicalUnits=True, numThreads=8)
            gal_cops = E.read_array('SUBFIND', path, snap, 'Subhalo/CentreOfPotential', noH=True,
                                    physicalUnits=True, numThreads=8)
            gal_hmr = E.read_array('SUBFIND', path, snap, 'Subhalo/HalfMassRad', noH=True,
                                    physicalUnits=True, numThreads=8)[:, 4]
            gal_masses = E.read_array('SUBFIND', path, snap, 'Subhalo/Stars/Mass', noH=True,
                                      physicalUnits=True, numThreads=8) * 10 ** 10

            print(len(gal_cops), 'before cut')
            gal_cops = gal_cops[gal_masses > 1e9]
            gal_hmr = gal_hmr[gal_masses > 1e9]
            gal_cops = gal_cops[gal_hmr / (csoft / (1 + z)) < 1.1]
            gal_hmr = gal_hmr[gal_hmr / (csoft / (1 + z)) < 1.1]
            print(len(gal_cops), 'after cut')

            tree = cKDTree(all_poss, leafsize=16, compact_nodes=False, balanced_tree=False)

            if gal_cops.shape[0] != 0:
                metallicity_dict[snap][reg], rads_dict[snap][reg] = get_r_and_met(all_poss, mets, gal_cops,
                                                                                  gal_hmr, tree)

        except OSError:
            continue
        except ValueError:
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

    xs = np.concatenate(list(rads_dict[snap].values()))
    metallicity_plt = np.concatenate(list(metallicity_dict[snap].values()))

    nH = 10**-1 * (metallicity_plt/0.002)**-0.64
    nH[np.where(nH > 10)] = 10

    # xs = xs[[metallicity_plt > 0]]
    # metallicity_plt = metallicity_plt[metallicity_plt > 0]

    cbar = ax.hexbin(xs, nH, gridsize=100, mincnt=1, xscale='log',
                     norm=LogNorm(), linewidths=0.2, cmap='viridis')

    ax.text(0.1, 0.9, f'$z={z}$', bbox=dict(boxstyle="round,pad=0.3", fc='w', ec="k", lw=1, alpha=0.8),
            transform=ax.transAxes, horizontalalignment='left', fontsize=8)

    axlims_x.extend(ax.get_xlim())
    axlims_y.extend(ax.get_ylim())

    # Label axes
    if i == 2:
        ax.set_xlabel(r'$R/R_{1/2,\star}$')
    if j == 0:
        ax.set_ylabel('$n_{H}^{*}/\mathrm{cm}^{-3}$')

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

fig.savefig('plots/nH_profile.png',
            bbox_inches='tight')

plt.close(fig)