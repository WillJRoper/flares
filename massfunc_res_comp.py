#!/cosma/home/dp004/dc-rope1/.conda/envs/flares-env/bin/python
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import astropy.units as u
import astropy.constants as const
import eagle_IO as E
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
                           noH=False, physicalUnits=False, numThreads=4)  # dm particle coordinates

    hull = ConvexHull(dm_cood)

    cen = [np.median(dm_cood[:, 0]), np.median(dm_cood[:, 1]), np.median(dm_cood[:, 2])]
    pedge = dm_cood[hull.vertices]  # edge particles
    y_obs = np.zeros(len(pedge))
    p0 = np.append(cen, 14.)

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

tag = "FOF/Group_M_Crit200"
snap = '010_z005p000'
group = "SUBFIND_GROUP"

# Extarct M_200s
M_200 = get_mass_data('/cosma/home/dp004/dc-rope1/FLARES/FLARES-1/G-EAGLE_38/data/', snap,
                      tag, group=group, noH=True, cut_bounds=True)
# M_200_hrDMO = get_mass_data('/cosma7/data/dp004/dc-love2/data/G-EAGLE/geagle_0032_hires/data/', snap,
#                       "FOF/Group_M_Crit200", group=group, noH=True, cut_bounds=True)
M_200_hr = get_mass_data('/cosma/home/dp004/dc-rope1/FLARES/FLARES-HD/FLARES_HR_EMOSAICS_38/data/', snap,
                         tag, group=group, noH=True, cut_bounds=True)

M_200 = M_200[np.where(M_200 != 0.0)] * 10**10
M_200_hr = M_200_hr[np.where(M_200_hr != 0.0)] * 10**10
# M_200_hrDMO = M_200_hrDMO[np.where(M_200_hrDMO != 0.0)] * 10**10

print('Minimums:', M_200.min(), M_200_hr.min())
print('Maximums:', M_200.max(), M_200_hr.max())

# Set up plot
fig = plt.figure()
ax = fig.add_subplot(111)

bins = np.logspace(np.log10(np.min((M_200.min(), M_200_hr.min(), M_200_hrDMO.min()))),
                   np.log10(np.max((M_200.max(), M_200_hr.max(), M_200_hrDMO.max()))),
                   75)

interval = bins[1:] - bins[:-1]

# Histogram the DMLJ halo masses
H, bins = np.histogram(M_200, bins=bins)
H_hr, _ = np.histogram(M_200_hr, bins=bins)
# H_hrDMO, _ = np.histogram(M_200_hrDMO, bins=bins)

# Remove zeros for plotting
#H = H[np.where(H != 0)]
#interval1 = interval[np.where(H != 0)]
#H_hr = H_hr[np.where(H_hr != 0)]
#interval2 = interval[np.where(H_hr != 0)]

# Compute bin centres
bin_cents = bins[1:] - ((bins[1] - bins[0]) / 2)
#bin_cents1 = bin_cents[np.where(H != 0)]
#bin_cents2 = bin_cents[np.where(H_hr != 0)]

# Plot each histogram
ax.loglog(bin_cents, H/interval, label='"Standard" Resolution')
ax.loglog(bin_cents, H_hr/interval, linestyle='--', label='High Resolution')
# ax.loglog(bin_cents, H_hrDMO/interval, linestyle='--', label='High Resolution (DMO)')

# Label axes
ax.set_xlabel(r'$M_{200}/M_\odot$')
ax.set_ylabel(r'$dN/dM$')

# Get and draw legend
handles, labels = ax.get_legend_handles_labels()
ax.legend(handles, labels)

# Save figure
fig.savefig('plots/mass_func_res_comp' + snap + '.png', bbox_inches='tight')
