#!/cosma/home/dp004/dc-rope1/.conda/envs/flares-env/bin/python
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import astropy.units as u
import astropy.constants as const
import eagle_IO.eagle_IO as E
import seaborn as sns
from flares import flares
matplotlib.use('Agg')

sns.set_style('whitegrid')


def get_mass_data(path, snap, tag, group="SUBFIND", noH=True, cut_bounds=True):

    # Extract mass data
    M_dat = E.read_array(group, path, snap, tag, noH=noH, numThreads=8)[:, 4]

    # If boundaries to be eliminated
    if cut_bounds:
        centre, radius, mindist = flares.spherical_region(path, snap)
        R_cop = E.read_array("SUBFIND", path, snap, "Subhalo/CentreOfPotential",
                             noH=noH, numThreads=8)

        # Get the radius of each group
        R_cop -= centre
        radii = np.linalg.norm(R_cop, axis=1)
        M_dat = M_dat[np.where(radii < 14 / 0.677700)]

    return M_dat


# Extarct M_200s
tag = "Subhalo/ApertureMeasurements/Mass/030kpc"
snap = '010_z005p000'
group = "SUBFIND"

snaps = ['010_z005p000', ]

M_200 = []
M_200_hr = []

for snap in snaps:
    M_200.extend(get_mass_data('/cosma/home/dp004/dc-rope1/FLARES/FLARES-1/G-EAGLE_24/data/', snap,
                               tag, group=group, noH=True, cut_bounds=False))
    # M_200_hrDMO = get_mass_data('/cosma7/data/dp004/dc-love2/data/G-EAGLE/geagle_0032_hires/data/', snap,
    #                       tag, group=group, noH=True, cut_bounds=False)
    M_200_hr.extend(get_mass_data('/cosma/home/dp004/dc-rope1/FLARES/FLARES-HD/FLARES_HR_24/data/', snap,
                                  tag, group=group, noH=True, cut_bounds=False))

M_200 = np.array(M_200)
M_200_hr = np.array(M_200_hr)

M_200 = M_200[np.where(M_200 != 0.0)] * 10**10
M_200_hr = M_200_hr[np.where(M_200_hr != 0.0)] * 10**10
# M_200_hrDMO = M_200_hrDMO[np.where(M_200_hrDMO != 0.0)] #* 10**10

print('Minimums:', M_200.min(), M_200_hr.min())
print('Maximums:', M_200.max(), M_200_hr.max())
print('Sums:', np.sum(M_200), np.sum(M_200_hr), np.sum(M_200) / np.sum(M_200_hr) * 100)

# Set up plot
fig = plt.figure()
ax = fig.add_subplot(111)

bins = np.logspace(np.log10(np.min((M_200.min(), M_200_hr.min()))),
                   np.log10(np.max((M_200.max(), M_200_hr.max()))),
                   40)

interval = bins[1:] - bins[:-1]

# Histogram the DMLJ halo masses
H, bins = np.histogram(M_200, bins=bins)
H_hr, _ = np.histogram(M_200_hr, bins=bins)

H_cumsum = np.cumsum(H[::-1])
H_hr_cumsum = np.cumsum(H_hr[::-1])

# Remove zeros for plotting
#H = H[np.where(H != 0)]
#interval1 = interval[np.where(H != 0)]
#H_hr = H_hr[np.where(H_hr != 0)]
#interval2 = interval[np.where(H_hr != 0)]

# Compute bin centres
bin_cents = bins[1:] - ((bins[1] - bins[0]) / 2)
bin_cents_cumsum = bin_cents[::-1]
#bin_cents1 = bin_cents[np.where(H != 0)]
#bin_cents2 = bin_cents[np.where(H_hr != 0)]

# Plot each histogram
ax.loglog(bin_cents, H/interval, label='Standard')
ax.loglog(bin_cents, H_hr/interval, linestyle='--', label='High Resolution')

# ax.set_xlim(10**7.5, None)
# ax.set_ylim(None, 10**-5.5)

# Label axes
ax.set_xlabel(r'$M_{\star}/M_\odot$')
ax.set_ylabel(r'$dN/dM$')

# Get and draw legend
handles, labels = ax.get_legend_handles_labels()
ax.legend(handles, labels)

# Save figure
fig.savefig('plots/GSMF_res_comp_' + snap + '.png', bbox_inches='tight')

plt.close(fig)

# Set up plot
fig = plt.figure()
ax = fig.add_subplot(111)

# Plot each histogram
ax.loglog(bin_cents_cumsum, H_cumsum/interval, label='Standard')
ax.loglog(bin_cents, H_hr_cumsum/interval, linestyle='--', label='High Resolution')

# ax.set_xlim(10**7.5, None)
# ax.set_ylim(None, 10**-5.5)

# Label axes
ax.set_xlabel(r'$M_{\star}/M_\odot$')
ax.set_ylabel(r'$dN(>M)/dM$')

# Get and draw legend
handles, labels = ax.get_legend_handles_labels()
ax.legend(handles, labels)

# Save figure
fig.savefig('plots/GSMF_res_comp_' + snap + 'cumulative.png',
            bbox_inches='tight')
