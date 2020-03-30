#!/cosma/home/dp004/dc-rope1/.conda/envs/flares-env/bin/python
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree
import matplotlib
import numba as nb
import eagle_IO as E
import seaborn as sns
from flares import flares
matplotlib.use('Agg')

sns.set_style('whitegrid')


# @nb.jit(nogil=True, parallel=True)
def get_parts_in_aperture(masses, cent, tree, app):

    # Get galaxy particle indices
    query = tree.query_ball_point(cent, r=app)

    # Get particle positions and masses
    gal_masses = masses[query]

    return np.sum(gal_masses)


def get_m(masses, gal_cops, tree):

    # Loop over galaxies centres
    ms = np.zeros_like(gal_cops)
    for ind, cop in enumerate(gal_cops):

        # Get particles and masses
        ms[ind] = get_parts_in_aperture(masses, cop, tree, app=0.03)

    return ms


def get_mass_data(path, snap, tag, reg, group="SUBFIND_GROUP", noH=True):

    path = '/cosma7/data/dp004/dc-love2/data/G-EAGLE/geagle_00' + reg + '/data/'
    try:
        all_poss = E.read_array('SNAP', path, snap, 'PartType4/Coordinates', noH=True,
                                physicalUnits=True, numThreads=8)
        masses = E.read_array('SNAP', path, snap, 'PartType4/Mass', noH=True,
                              physicalUnits=True, numThreads=8) * 10**10
        gal_cops = E.read_array('SUBFIND', path, snap, 'Subhalo/CentreOfPotential', noH=True,
                                physicalUnits=True, numThreads=8)
        gal_ms = E.read_array('SUBFIND', path, snap, 'Subhalo/Stars/Mass', noH=True,
                                physicalUnits=True, numThreads=8) * 10**10

        print(len(gal_cops), 'before cut')
        gal_cops = gal_cops[gal_ms > 0]
        M_dat = gal_ms[gal_ms > 0]
        print(len(gal_cops), 'after cut')

        tree = cKDTree(all_poss, leafsize=16, compact_nodes=False, balanced_tree=False)

        M_30 = get_m(masses, gal_cops, tree)

    except OSError:
        M_30 = np.full(100, 0.0)
        M_dat = np.full(100, 0.0)
    except ValueError:
        M_30 = np.full(100, 0.0)
        M_dat = np.full(100, 0.0)

    return M_dat, M_30


# Extarct M_subfinds
tag = "Subhalo/Stars/Mass"
snap = '010_z005p000'
group = "SUBFIND"
regions = []
for reg in range(0, 40):

    if reg < 10:
        regions.append('0' + str(reg))
    else:
        regions.append(str(reg))

M_subfind_dict = {}
M_30kpc_dict = {}
for reg in regions:

    M_subfind, M_30kpc = get_mass_data('/cosma/home/dp004/dc-rope1/FLARES/FLARES-1/G-EAGLE_' + reg + '/data/', snap,
                          tag, reg, group=group, noH=True)

    print('Nhalos', M_subfind.shape, M_30kpc.shape)

    M_subfind_dict[reg] = M_subfind
    M_30kpc_dict[reg] = M_30kpc

    # print('Minimums:', M_subfind.min(), M_30kpc.min())
    # print('Maximums:', M_subfind.max(), M_30kpc.max())
    # print('Sums:', np.sum(M_subfind), np.sum(M_30kpc), np.sum(M_subfind) / np.sum(M_30kpc) * 100)

M_subfind = np.concatenate(list(M_subfind_dict.values()))
M_30kpc = np.concatenate(list(M_30kpc_dict.values()))
M_subfind = M_subfind[np.where(M_subfind != 0.0)]
M_30kpc = M_30kpc[np.where(M_30kpc != 0.0)]
print('Nhalos', len(M_subfind), len(M_30kpc))
# Set up plot
fig = plt.figure()
ax = fig.add_subplot(111)

bins = np.logspace(np.log10(np.min((M_subfind.min(), M_30kpc.min()))),
                   np.log10(np.max((M_subfind.max(), M_30kpc.max()))),
                   40)

interval = bins[1:] - bins[:-1]

# Histogram the DMLJ halo masses
H, bins = np.histogram(M_subfind, bins=bins)
H_hr, _ = np.histogram(M_30kpc, bins=bins)

# Compute bin centres
bin_cents = bins[1:] - ((bins[1] - bins[0]) / 2)

# Plot each histogram
ax.loglog(bin_cents, H/interval, label='SUBFIND')
ax.loglog(bin_cents, H_hr/interval, linestyle='--', label='All particles in 30 pkpc')

# Label axes
ax.set_xlabel(r'$M_{\star}/M_\odot$')
ax.set_ylabel(r'$dN/dM$')

# Get and draw legend
handles, labels = ax.get_legend_handles_labels()
ax.legend(handles, labels)

# Save figure
fig.savefig('plots/GSMF_mass_comp_' + snap + '15kpc.png', bbox_inches='tight')
