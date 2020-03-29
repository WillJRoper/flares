#!/cosma/home/dp004/dc-rope1/.conda/envs/flares-env/bin/python
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import numba as nb
import eagle_IO as E
import seaborn as sns
from flares import flares
matplotlib.use('Agg')

sns.set_style('whitegrid')


@nb.njit(nogil=True, parallel=True)
def get_parts_in_aperture(all_poss, masses, cent, app):

    # Get galaxy particle indices
    seps = all_poss - cent
    rs2 = seps[:, 0]**2 + seps[:, 1]**2 + seps[:, 2]**2
    cond = rs2 < app**2

    # Get particle positions and masses
    gal_masses = masses[cond]

    return np.sum(gal_masses)


def get_m(all_poss, masses, gal_cops):

    # Loop over galaxies centres
    ms = np.full_like(gal_cops, -2)
    for ind, cop in enumerate(gal_cops):

        # Get particles and masses
        ms[ind] = get_parts_in_aperture(all_poss, masses, cop, app=0.03)

    return ms


def get_mass_data(path, snap, tag, reg, group="SUBFIND_GROUP", noH=True):

    # Extract mass data
    M_dat = E.read_array(group, path, snap, tag, noH=noH)

    path = '/cosma7/data/dp004/dc-love2/data/G-EAGLE/geagle_00' + reg + '/data/'
    try:
        all_poss = E.read_array('SNAP', path, snap, 'PartType4/Coordinates', noH=True,
                                physicalUnits=True, numThreads=8)
        masses = E.read_array('SNAP', path, snap, 'PartType4/Mass', noH=True,
                              physicalUnits=True, numThreads=8)
        gal_cops = E.read_array('SUBFIND', path, snap, 'Subhalo/CentreOfPotential', noH=True,
                                physicalUnits=True, numThreads=8)

        M_30 = get_m(all_poss, masses, gal_cops)

    except OSError:
        M_30 = np.full_like(M_dat, 0.0)
    except ValueError:
        M_30 = np.full_like(M_dat, 0.0)

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

    M_subfind_dict[reg] = M_subfind[np.where(M_subfind != 0.0)] * 10**10
    M_30kpc_dict[reg] = M_30kpc[np.where(M_30kpc != 0.0)] * 10**10

    print('Minimums:', M_subfind.min(), M_30kpc.min())
    print('Maximums:', M_subfind.max(), M_30kpc.max())
    print('Sums:', np.sum(M_subfind), np.sum(M_30kpc), np.sum(M_subfind) / np.sum(M_30kpc) * 100)

M_subfind = np.concatenate(list(M_subfind_dict.values()))
M_30kpc = np.concatenate(list(M_30kpc_dict.values()))

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
ax.loglog(bin_cents, H/interval, label='"SUBFIND')
ax.loglog(bin_cents, H_hr/interval, linestyle='--', label='All particles in 30 pkpc')

# Label axes
ax.set_xlabel(r'$M_{\star}/M_\odot$')
ax.set_ylabel(r'$dN/dM$')

# Get and draw legend
handles, labels = ax.get_legend_handles_labels()
ax.legend(handles, labels)

# Save figure
fig.savefig('plots/GSMF_mass_comp_' + snap + '.png', bbox_inches='tight')
