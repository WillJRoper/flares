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

M_200 = M_200[np.where(M_200 != 0.0)]
M_200_ref = M_200_ref[np.where(M_200_ref != 0.0)]

print('Minimums:', M_200.min(), M_200_ref.min())
print('Maximums:', M_200.max(), M_200_ref.max())

# Set up plot
fig = plt.figure()
ax = fig.add_subplot(111)

# Plot each histogram
cbar = ax.hexbin(M_200, M_200_ref, gridsize=100, mincnt=1, xscale='log', yscale='log',
                 norm=LogNorm(), linewidths=0.2, cmap='viridis', zorder=0)
ax.loglog((np.min((M_200.min(), M_200_ref.min())), np.max((M_200.min(), M_200_ref.min()))),
          (np.min((M_200.min(), M_200_ref.min())), np.max((M_200.min(), M_200_ref.min()))),
          linestyle='dashed', color='k')

# Label axes
ax.set_xlabel(r'$R_{1/2, \star, AGNdT9}/ [\mathrm{pkpc}]$')
ax.set_ylabel(r'$R_{1/2, \star, REF}/ [\mathrm{pkpc}]$')

fig.colorbar(cbar)

# Save figure
fig.savefig('plots/hmr_subgrid_comp' + snap + '.png', bbox_inches='tight')
