#!/cosma/home/dp004/dc-rope1/.conda/envs/flares-env/bin/python
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import astropy.units as u
import astropy.constants as const
from matplotlib.colors import LogNorm
import eagle_IO.eagle_IO as E
import seaborn as sns
from scipy.optimize import curve_fit
from scipy.spatial import ConvexHull
# from flares import flares
matplotlib.use('Agg')

sns.set_style('whitegrid')


# @jit
def _sphere(coords, a, b, c, r):
    # Equation of a sphere

    x, y, z = coords[:, 0], coords[:, 1], coords[:, 2]

    return (x - a) ** 2 + (y - b) ** 2 + (z - c) ** 2 - r ** 2


def spherical_region(sim, snap):
    """
    Inspired from David Turner's suggestion
    """

    dm_cood = E.read_array('PARTDATA', sim, snap, '/PartType1/Coordinates',
                           noH=True, physicalUnits=False, numThreads=4)  # dm particle coordinates

    hull = ConvexHull(dm_cood)

    cen = [np.median(dm_cood[:, 0]), np.median(dm_cood[:, 1]), np.median(dm_cood[:, 2])]
    pedge = dm_cood[hull.vertices]  # edge particles
    y_obs = np.zeros(len(pedge))
    p0 = np.append(cen, 14. / 0.677700)

    popt, pcov = curve_fit(_sphere, pedge, y_obs, p0, method='lm', sigma=np.ones(len(pedge)) * 0.001)
    dist = np.sqrt(np.sum((pedge - popt[:3]) ** 2, axis=1))
    centre, radius, mindist = popt[:3], popt[3], np.min(dist)

    return centre, radius, mindist


def get_mass_data(path, snap, tag, group="SUBFIND_GROUP", noH=True, cut_bounds=True):

    # Extract mass data
    M_dat = E.read_array(group, path, snap, tag, noH=noH)

    # If boundaries to be eliminated
    if cut_bounds:
        centre, radius, mindist = spherical_region(path, snap)
        R_cop = E.read_array("SUBFIND", path, snap, "FOF/GroupCentreOfPotential", noH=noH)

        # Get the radius of each group
        R_cop -= centre
        radii = np.linalg.norm(R_cop, axis=1)
        M_dat = M_dat[np.where(radii < 14 / 0.677700)]

    return M_dat

tag = "Subhalo/HalfMassRad"
snap = '010_z005p000'
group = "SUBFIND"

# Extarct M_200s
M_200 = get_mass_data('/cosma/home/dp004/dc-rope1/FLARES/FLARES-1/G-EAGLE_00/data/', snap,
                      tag, group=group, noH=True, cut_bounds=True)[:, 4] * 1e3
# M_200_refDMO = get_mass_data('/cosma7/data/dp004/dc-love2/data/G-EAGLE/geagle_0032_hires/data/', snap,
#                       "FOF/Group_M_Crit200", group=group, noH=True, cut_bounds=True)
M_200_ref = get_mass_data('/cosma/home/dp004/dc-rope1/FLARES/FLARES-1/FLARES_00_REF/data/', snap,
                         tag, group=group, noH=True, cut_bounds=True)[:, 4] * 1e3
print(M_200.shape, M_200_ref.shape)
M_200 = M_200[np.where(M_200 != 0.0)]
M_200_ref = M_200_ref[np.where(M_200_ref != 0.0)]

print('Minimums:', M_200.min(), M_200_ref.min())
print('Maximums:', M_200.max(), M_200_ref.max())

# Set up plot
fig = plt.figure()
ax = fig.add_subplot(111)

bins = np.logspace(np.log10(np.min((M_200.min(), M_200_ref.min()))),
                   np.log10(np.max((M_200.max(), M_200_ref.max()))),
                   75)

interval = bins[1:] - bins[:-1]

# Histogram the DMLJ halo masses
H, bins = np.histogram(M_200, bins=bins)
H_ref, _ = np.histogram(M_200_ref, bins=bins)

# Compute bin centres
bin_cents = bins[1:] - ((bins[1] - bins[0]) / 2)

# Plot each histogram
ax.loglog(bin_cents, H/interval, label='"AGNdT9')
ax.loglog(bin_cents, H_ref/interval, linestyle='--', label='REFERENCE')

# Label axes
ax.set_xlabel(r'$M_{\star}/M_\odot$')
ax.set_ylabel(r'$dN/dM$')

# Get and draw legend
handles, labels = ax.get_legend_handles_labels()
ax.legend(handles, labels)

# Save figure
fig.savefig('plots/hmr_subgrid_comp' + snap + '.png', bbox_inches='tight')
