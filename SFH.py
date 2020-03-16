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


def calc_srf(z, a_born, mass, t_bin=100):

    # Convert scale factor into redshift
    z_born = 1 / a_born - 1

    # Convert to time in Gyrs
    t = cosmo.age(z)
    t_born = cosmo.age(z_born)

    # Calculate the VR
    age = (t - t_born).to(u.Myr)

    ok = np.where(age.value <= t_bin)[0]
    if len(ok) > 0:

        # Calculate the SFR
        sfr = np.sum(mass[ok]) / (t_bin * 1e6)

    else:
        sfr = 0.0

    return sfr


regions = []
for reg in range(0, 1):

    if reg < 10:
        regions.append('000' + str(reg))
    else:
        regions.append('00' + str(reg))

snaps = ['000_z015p000', '001_z014p000', '002_z013p000', '003_z012p000', '004_z011p000', '005_z010p000',
         '006_z009p000', '007_z008p000', '008_z007p000', '009_z006p000', '010_z005p000', '011_z004p770']

zs_dict = {}
stellar_a_dict = {}
starmass_dict = {}
halo_id_part_inds = {}
for snap in snaps:

    stellar_a_dict[snap] = {}
    starmass_dict[snap] = {}
    halo_id_part_inds[snap] = {}

for reg in regions:

    for snap in snaps:

        print(reg, snap)

        path = '/cosma7/data/dp004/dc-love2/data/G-EAGLE/geagle_' + reg + '/data/'
        try:
            starmass_dict[snap][reg] = E.read_array('SNAP', path, snap, 'PartType4/Mass',
                                                    noH=True, numThreads=8) * 10**10
            stellar_a_dict[snap][reg] = E.read_array('SNAP', path, snap, 'PartType4/StellarFormationTime',
                                                     noH=True, numThreads=8)
            part_ids = E.read_array('SNAP', path, snap, 'PartType4/ParticleIDs', numThreads=8)
            group_ids = E.read_array('PARTDATA', path, snap, 'PartType4/SubGroupNumber', numThreads=8)
        except:
            continue

        ind_to_pid = {}
        pid_to_ind = {}
        for ind, pid in enumerate(part_ids):
            ind_to_pid[ind] = pid
            pid_to_ind[pid] = ind
            if ind % 10000000 == 0:
                print('Mapping particle IDs to index:', pid, 'to', ind, 'of', len(part_ids), end='\r')

        halo_id_part_inds = {}
        for pid, simid in zip(part_ids, group_ids):
            simid = int(simid)
            if simid == 2 ** 30:
                continue
            try:
                halo_id_part_inds.setdefault(simid, set()).update({pid_to_ind[pid]})
            except KeyError:
                ind_to_pid[ind + 1] = pid
                pid_to_ind[pid] = ind + 1
                halo_id_part_inds.setdefault(simid, set()).update({pid_to_ind[pid]})

# Get halos which are in the distribution at the z=4.77
halos_in_pop = {}
for reg in regions:
    for grp in halo_id_part_inds['011_z004p770'][reg].keys():
        parts = list(halo_id_part_inds['011_z004p770'][reg][grp])
        if np.sum(starmass_dict['011_z004p770'][reg][parts]) > 10**9:
            halos_in_pop.setdefault(reg, []).append(grp)

sfrs_gals = {}
for snap in halo_id_part_inds.keys():
    sfrs_gals[snap] = {}
    z_str = snap.split('z')[1].split('p')
    z = float(z_str[0] + '.' + z_str[1])
    for reg in halo_id_part_inds[snap].keys():
        sfrs_gals[snap][reg] = {}
        for grp in halo_id_part_inds[snap][reg].keys():
            parts = list(halo_id_part_inds[snap][reg][grp])
            sfrs_gals[snap][reg][grp] = calc_srf(z, stellar_a_dict[snap][reg][parts], starmass_dict[snap][reg][parts])
zs = {}
zs_plt = []
sfrs = {}
for snap in snaps:

    print(snap)
    z_str = snap.split('z')[1].split('p')
    z = float(z_str[0] + '.' + z_str[1])
    # zs[snap] = np.full_like(starmass[snap], z)
    zs_plt.append(z)
    sfrs[snap] = np.concatenate([list(sfrs_gals[snap][reg].values()) for reg in sfrs_gals[snap].keys()])

# hex_sfrs = np.concatenate(list(sfrs.values()))
# hex_zs = np.concatenate(list(zs.values()))

medians = np.zeros(len(snaps))
pcent84 = np.zeros(len(snaps))
pcent16 = np.zeros(len(snaps))
for ind, snap in enumerate(snaps):

    print(ind, snap)

    medians[ind] = np.median(sfrs[snap])
    pcent84[ind] = np.percentile(sfrs[snap], 84)
    pcent16[ind] = np.percentile(sfrs[snap], 16)

fig = plt.figure()
ax = fig.add_subplot(111)

# cbar = ax.hexbin(hex_zs, hex_sfrs, gridsize=100, mincnt=1, norm=LogNorm(), yscale='log',
#                  linewidths=0.2, cmap='Greys', zorder=0)
ax.plot(zs_plt, medians, linestyle='--', color='r')
ax.fill_between(zs_plt, pcent16, pcent84, alpha=0.4, color='g')

ax.set_xlabel('$z$')
ax.set_ylabel('SFR / $[M_\odot/\mathrm{yr}]$')

ax.set_yscale('log')

# cax = fig.colorbar(cbar, ax=ax)
# cax.ax.set_ylabel(r'$N$')

fig.savefig('plots/SFH.png')
