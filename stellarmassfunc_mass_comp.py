#!/cosma/home/dp004/dc-rope1/.conda/envs/flares-env/bin/python
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree
import matplotlib
import numba as nb
import eagle_IO.eagle_IO as E
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
    ms = np.zeros(gal_cops.size)
    for ind, cop in enumerate(gal_cops):

        # Get particles and masses
        ms[ind] = get_parts_in_aperture(masses, cop, tree, app=0.03)

    return ms


def get_mass_data(path, snap, tag, reg, group="SUBFIND_GROUP", noH=True):

    gal_ms = E.read_array('SUBFIND', path, snap, 'Subhalo/ApertureMeasurements/Mass/030kpc',
                          numThreads=8) * 10**10 / 0.6777

    return gal_ms


# Extarct M_insts
tag = "Subhalo/Stars/Mass"
snap = '010_z005p000'
group = "SUBFIND"
regions = []
for reg in range(0, 1):

    if reg < 10:
        regions.append('0' + str(reg))
    else:
        regions.append(str(reg))

M_inst = []
M_std = []
for reg in regions:

    M_std.extend(get_mass_data('/cosma/home/dp004/dc-rope1/FLARES/FLARES-1/G-EAGLE_' + reg + '/data/', snap,
                                 tag, reg, group=group, noH=True))
    M_inst.extend(get_mass_data('/cosma7/data/dp004/FLARES/FLARES-1/FLARES_00_instantFB/data/', snap,
                                 tag, reg, group=group, noH=True))

M_inst = np.array(M_inst)
M_std = np.array(M_std)
M_inst = M_inst[np.where(M_inst != 0.0)]
M_std = M_std[np.where(M_std != 0.0)]
print('Nhalos', len(M_inst), len(M_std))
# Set up plot
fig = plt.figure()
ax = fig.add_subplot(111)

bins = np.logspace(np.log10(np.min((M_inst.min(), M_std.min()))),
                   np.log10(np.max((M_inst.max(), M_std.max()))),
                   40)

interval = bins[1:] - bins[:-1]

# Histogram the DMLJ halo masses
H, bins = np.histogram(M_inst, bins=bins)
H_hr, _ = np.histogram(M_std, bins=bins)

# Compute bin centres
bin_cents = bins[1:] - ((bins[1] - bins[0]) / 2)

# Plot each histogram
ax.loglog(bin_cents, H/interval, label='Instantaneous')
ax.loglog(bin_cents, H_hr/interval, linestyle='--', label='30 Myr Delay')

# Label axes
ax.set_xlabel(r'$M_{\star}/M_\odot$')
ax.set_ylabel(r'$dN/dM$')

# Get and draw legend
handles, labels = ax.get_legend_handles_labels()
ax.legend(handles, labels)

# Save figure
fig.savefig('plots/GSMF_mass_comp_' + snap + '15kpc.png', bbox_inches='tight')
