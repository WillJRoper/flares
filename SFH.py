#!/cosma/home/dp004/dc-rope1/.conda/envs/flares-env/bin/python
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import astropy.units as u
import astropy.constants as cons
from astropy.cosmology import Planck13 as cosmo
from matplotlib.colors import LogNorm
import eagle_IO as E
import seaborn as sns
from flares import flares
matplotlib.use('Agg')

sns.set_style('whitegrid')


def calc_srf(z, a_born, mass):

    # Convert scale factor into redshift
    z_born = 1 / a_born - 1

    # Convert to time in Gyrs
    t = cosmo.age(z)
    t_born = cosmo.age(z_born)

    # Calculate the VR
    age = ((t - t_born) * u.Gyr).to(u.yr)

    # Calculate the SFR
    sfr = mass / age

    return sfr


regions = []
for reg in range(0, 2):

    if reg < 10:
        regions.append('000' + str(reg))
    else:
        regions.append('00' + str(reg))

snaps = ['000_z015p000', '001_z014p000', '002_z013p000', '003_z012p000', '004_z011p000', '005_z010p000',
             '006_z009p000', '007_z008p000', '008_z007p000', '009_z006p000', '010_z005p000', '011_z004p770']

zs_dict = {}
stellar_a_dict = {}
starmass_dict = {}
for snap in snaps:

    stellar_a_dict[snap] = {}
    starmass_dict[snap] = {}

for reg in regions:

    for snap in snaps:

        print(reg, snap)

        path = '/cosma7/data/dp004/dc-love2/data/G-EAGLE/geagle_' + reg + '/data/'

        stellar_a_dict[snap][reg] = E.read_array('SNAP', path, snap, 'PartType4/StellarFormationTime',
                                                  noH=True, numThreads=8)
        starmass_dict[snap][reg] = E.read_array('SNAP', path, snap, 'PartType4/Mass',
                                                noH=True, numThreads=8)

stellar_a = {}
starmass = {}
zs = {}
zs_plt = []
sfrs = {}
for snap in snaps:

    stellar_a[snap] = np.concatenate(list(stellar_a_dict[snap].values()))
    starmass[snap] = np.concatenate(list(starmass_dict[snap].values())) * 10**10
    z_str = snap.split('z')[1].split('p')
    z = float(z_str[0] + '.' + z_str[1])
    zs[snap] = np.full_like(starmass[snap], z)
    zs_plt.append(z)
    sfrs[snap] = calc_srf(z, stellar_a[snap], starmass[snap])

medians = np.zeros(len(snaps))
pcent84 = np.zeros(len(snaps))
pcent16 = np.zeros(len(snaps))
for ind, snap in enumerate(snaps):

    medians[ind] = np.median(sfrs[snap])
    pcent84[ind] = np.percentile(sfrs[snap], 84)
    pcent16[ind] = np.percentile(sfrs[snap], 16)

fig = plt.figure()
ax = fig.add_subplot(111)

ax.plot(zs, medians, linestyle='--', color='r')
ax.fill_between(zs, pcent16, pcent84, alpha=0.6, clor='g')

ax.set_xlabel('$z$')
ax.set_ylabel('SFR / $[M_\odot/\mathrm{yr}]$')

fig.savefig('plots/SFH.png')
