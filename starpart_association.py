#!/cosma/home/dp004/dc-rope1/.conda/envs/flares-env/bin/python
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.gridspec as gridspec
import eagle_IO.eagle_IO as E
import seaborn as sns
import pickle
matplotlib.use('Agg')


sns.set_style('whitegrid')

load = False
regions = []
for reg in range(0, 40):

    if reg < 10:
        regions.append('0' + str(reg))
    else:
        regions.append(str(reg))

snaps = ['000_z015p000', '001_z014p000', '002_z013p000', '003_z012p000', '004_z011p000', '005_z010p000',
         '006_z009p000', '007_z008p000', '008_z007p000', '009_z006p000', '010_z005p000', '011_z004p770']

if not load:
    axlims_x = []
    axlims_y = []

    n_in_grpids_dict = {}
    n_in_subgrpids_dict = {}
    totn_dict = {}
    totm_dict = {}
    m_in_grpids_dict = {}
    m_in_subgrpids_dict = {}
    for snap in snaps:

        n_in_grpids_dict[snap] = 0
        n_in_subgrpids_dict[snap] = 0
        m_in_grpids_dict[snap] = 0
        m_in_subgrpids_dict[snap] = 0
        totn_dict[snap] = 0
        totm_dict[snap] = 0

    for reg in regions:

        n_in_grpids_dict[reg] = 0
        n_in_subgrpids_dict[reg] = 0
        m_in_grpids_dict[reg] = 0
        m_in_subgrpids_dict[reg] = 0
        totn_dict[reg] = 0
        totm_dict[reg] = 0

        for snap in snaps:

            print(reg, snap)

            path = '/cosma/home/dp004/dc-rope1/FLARES/FLARES-1/G-EAGLE_' + reg + '/data'

            try:
                grpids = E.read_array('PARTDATA', path, snap, 'PartType4/GroupNumber', numThreads=8)
                sub_grpids = E.read_array('PARTDATA', path, snap, 'PartType4/SubGroupNumber', numThreads=8)
                mass = E.read_array('PARTDATA', path, snap, 'PartType4/Mass', numThreads=8)
                totmass = E.read_array('SNAP', path, snap, 'PartType4/Mass', numThreads=8) / 0.6777
            except OSError:
                print("OsError")
                continue
            except ValueError:
                print("ValueError")
                continue

            totn_dict[snap] += totmass.size
            totm_dict[snap] += np.sum(totmass)

            totn_dict[reg] += totmass.size
            totm_dict[reg] += np.sum(totmass)

            ingrpinds = grpids != 2**30
            insubgrpinds = sub_grpids != 2**30
            n_in_grpids_dict[snap] += grpids[ingrpinds].size
            n_in_subgrpids_dict[snap] += sub_grpids[insubgrpinds].size
            m_in_grpids_dict[snap] += np.sum(mass[ingrpinds])
            m_in_subgrpids_dict[snap] += np.sum(mass[insubgrpinds])

            n_in_grpids_dict[reg] += grpids[ingrpinds].size
            n_in_subgrpids_dict[reg] += sub_grpids[insubgrpinds].size
            m_in_grpids_dict[reg] += np.sum(mass[ingrpinds])
            m_in_subgrpids_dict[reg] += np.sum(mass[insubgrpinds])

    with open('associationdicts.pck', 'wb') as pfile1:
        pickle.dump({"n_in_grpids_dict": n_in_grpids_dict, "n_in_subgrpids_dict": n_in_subgrpids_dict,
                     "m_in_grpids_dict": m_in_grpids_dict, "m_in_subgrpids_dict": m_in_subgrpids_dict,
                     "totn_dict": totn_dict, "totm_dict": totm_dict}, pfile1)

else:
    with open('associationdicts.pck', 'rb') as pfile1:
        save_dict = pickle.load(pfile1)
    n_in_grpids_dict = save_dict["n_in_grpids_dict"]
    n_in_subgrpids_dict = save_dict["n_in_subgrpids_dict"]
    totn_dict = save_dict["totn_dict"]
    totm_dict = save_dict["totm_dict"]
    m_in_grpids_dict = save_dict["m_in_grpids_dict"]
    m_in_subgrpids_dict = save_dict["m_in_subgrpids_dict"]

zs = []
plt_n_notingrp = []
plt_n_notinsubgrp = []
plt_m_notingrp = []
plt_m_notinsubgrp = []
plt_totn = []
plt_totm = []
for snap in snaps:
    z_str = snap.split('z')[1].split('p')
    z = float(z_str[0] + '.' + z_str[1])
    zs.append(z)
    plt_n_notingrp.append(totn_dict[snap] - n_in_grpids_dict[snap])
    plt_n_notinsubgrp.append(totn_dict[snap] - n_in_subgrpids_dict[snap])
    plt_m_notingrp.append(totm_dict[snap] - m_in_grpids_dict[snap])
    plt_m_notinsubgrp.append(totm_dict[snap] - m_in_subgrpids_dict[snap])
    plt_totn.append(totn_dict[snap])
    plt_totm.append(totm_dict[snap])

plt_n_notingrp = np.array(plt_n_notingrp)
plt_n_notinsubgrp = np.array(plt_n_notinsubgrp)
plt_m_notingrp = np.array(plt_m_notingrp)
plt_m_notinsubgrp = np.array(plt_m_notinsubgrp)
plt_totn = np.array(plt_totn)
plt_totm = np.array(plt_totm)

# Set up plot
fig = plt.figure(figsize=(6, 3.3))
gs = gridspec.GridSpec(ncols=3, nrows=4)
gs.update(wspace=0.0, hspace=0.0)
ax1 = fig.add_subplot(gs[0, :])
ax2 = fig.add_subplot(gs[1, :])
ax3 = fig.add_subplot(gs[2, :])
ax4 = fig.add_subplot(gs[3, :])

ax1.plot(zs, plt_n_notingrp / plt_totn, label="Not in a Group", color='#300bff')
ax1.plot(zs, plt_n_notinsubgrp / plt_totn, label="Not in a Subgroup", linestyle='--', color='#ff990b')
ax2.plot(zs, plt_m_notingrp / plt_totm, label="Not in a Group", color='#300bff')
ax2.plot(zs, plt_m_notinsubgrp / plt_totm, label="Not in a Subgroup", linestyle='--', color='#ff990b')
ax3.semilogy(zs, plt_n_notingrp, label="Not in a Group", color='#300bff')
ax3.semilogy(zs, plt_n_notinsubgrp, label="Not in a Subgroup", linestyle='--', color='#ff990b')
ax4.semilogy(zs, plt_m_notingrp * 10**10, label="Not in a Group", color='#300bff')
ax4.semilogy(zs, plt_m_notinsubgrp * 10**10, label="Not in a Subgroup", linestyle='--', color='#ff990b')

ax4.set_xlabel("$z$")
ax1.set_ylabel("$N_{out}/N_{tot}$")
ax2.set_ylabel("$M_{out}/M_{tot}$")
ax3.set_ylabel("$N_{out}$")
ax4.set_ylabel("$M_{out}/M_{\odot}$")

handles, labels = ax1.get_legend_handles_labels()
ax1.legend(handles, labels, loc='best')

fig.savefig("plots/star_particle_association.png", bbox_inches='tight')

plt.close(fig)

regs_ovdens = np.loadtxt("region_overdensity.txt", dtype=float)

ov_dens = []
plt_n_notingrp = []
plt_n_notinsubgrp = []
plt_m_notingrp = []
plt_m_notinsubgrp = []
plt_totn = []
plt_totm = []
for reg in regions:
    ov_dens.append(regs_ovdens[int(reg)])
    plt_n_notingrp.append(totn_dict[reg] - n_in_grpids_dict[reg])
    plt_n_notinsubgrp.append(totn_dict[reg] - n_in_subgrpids_dict[reg])
    plt_m_notingrp.append(totm_dict[reg] - m_in_grpids_dict[reg])
    plt_m_notinsubgrp.append(totm_dict[reg] - m_in_subgrpids_dict[reg])
    plt_totn.append(totn_dict[reg])
    plt_totm.append(totm_dict[reg])

sinds = np.argsort(ov_dens)
ov_dens = np.array(ov_dens)[sinds]
plt_n_notingrp = np.array(plt_n_notingrp)[sinds]
plt_n_notinsubgrp = np.array(plt_n_notinsubgrp)[sinds]
plt_m_notingrp = np.array(plt_m_notingrp)[sinds]
plt_m_notinsubgrp = np.array(plt_m_notinsubgrp)[sinds]
plt_totn = np.array(plt_totn)[sinds]
plt_totm = np.array(plt_totm)[sinds]

# Set up plot
fig = plt.figure(figsize=(6, 3.3))
gs = gridspec.GridSpec(ncols=3, nrows=4)
gs.update(wspace=0.0, hspace=0.0)
ax1 = fig.add_subplot(gs[0, :])
ax2 = fig.add_subplot(gs[1, :])
ax3 = fig.add_subplot(gs[2, :])
ax4 = fig.add_subplot(gs[3, :])

ax1.plot(ov_dens, plt_n_notingrp / plt_totn, label="Not in a Group", color='#300bff')
ax1.plot(ov_dens, plt_n_notinsubgrp / plt_totn, label="Not in a Subgroup", linestyle='--', color='#ff990b')
ax2.plot(ov_dens, plt_m_notingrp / plt_totm, label="Not in a Group", color='#300bff')
ax2.plot(ov_dens, plt_m_notinsubgrp / plt_totm, label="Not in a Subgroup", linestyle='--', color='#ff990b')
ax3.semilogy(ov_dens, plt_n_notingrp, label="Not in a Group", color='#300bff')
ax3.semilogy(ov_dens, plt_n_notinsubgrp, label="Not in a Subgroup", linestyle='--', color='#ff990b')
ax4.semilogy(ov_dens, plt_m_notingrp * 10**10, label="Not in a Group", color='#300bff')
ax4.semilogy(ov_dens, plt_m_notinsubgrp * 10**10, label="Not in a Subgroup", linestyle='--', color='#ff990b')

ax4.set_xlabel("$\Delta$")
ax1.set_ylabel("$N_{out}/N_{tot}$")
ax2.set_ylabel("$M_{out}/M_{tot}$")
ax3.set_ylabel("$N_{out}$")
ax4.set_ylabel("$M_{out}/M_{\odot}$")

handles, labels = ax1.get_legend_handles_labels()
ax1.legend(handles, labels, loc='best')

fig.savefig("plots/star_particle_association_environ.png", bbox_inches='tight')

