#!/cosma/home/dp004/dc-rope1/.conda/envs/flares-env/bin/python
import numpy as np
import matplotlib
from eagle_IO import eagle_IO as E
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from matplotlib.collections import LineCollection
import h5py
import seaborn as sns
matplotlib.use('Agg')


sns.set_style('white')

path = '/cosma/home/dp004/dc-rope1/FLARES/FLARES-1/G-EAGLE_06/data'
snap = '009_z006p000'
prog_snap = '008_z007p000'
desc_snap = '010_z005p000'
part_type = 1

# Define comoving softening length in kpc
csoft = 0.001802390 / 0.6777

hdf = h5py.File('/cosma/home/dp004/dc-rope1/FLARES/FLARES-1/MergerGraphs/GEAGLE_06/SubMgraph_009_z006p000.hdf5', 'r')

nprogs = hdf['nProgs'][...]

# Get insane halo
nprog = 224
insane_ind = nprogs == nprog
halo_ind = hdf['MEGA_halo_IDs'][insane_ind][0]
insane_ID = hdf['SUBFIND_halo_IDs'][halo_ind]
prog_start_ind = hdf['prog_start_index'][halo_ind]
desc_start_ind = hdf['desc_start_index'][halo_ind]
ndesc = hdf['nDescs'][halo_ind]
progs = hdf['Prog_haloIDs'][prog_start_ind: prog_start_ind + nprog]
descs = hdf['Desc_haloIDs'][desc_start_ind: desc_start_ind + nprog]

hdf.close()

# Load data for luminosities
snap_poss = E.read_array('PARTDATA', path, snap, 'PartType1/Coordinates', numThreads=8)
desc_poss = E.read_array('PARTDATA', path, desc_snap, 'PartType1/Coordinates', numThreads=8)
prog_poss = E.read_array('PARTDATA', path, prog_snap, 'PartType1/Coordinates', numThreads=8)

# Get this halos COP
subfind_grp_ids = E.read_array('SUBFIND', path, snap, 'Subhalo/GroupNumber', numThreads=8)
subfind_subgrp_ids = E.read_array('SUBFIND', path, snap, 'Subhalo/SubGroupNumber', numThreads=8)
cops = E.read_array('SUBFIND', path, snap, 'Subhalo/CentreOfPotential', numThreads=8)

# Convert to group.subgroup ID format
subfind_halo_ids = np.zeros(subfind_grp_ids.size, dtype=float)
for (ind, g), sg in zip(enumerate(subfind_grp_ids), subfind_subgrp_ids):
    subfind_halo_ids[ind] = float(str(int(g)) + '.%05d' % int(sg))
    
cop = cops[subfind_halo_ids == insane_ID, :]

# Extract the halo IDs (group names/keys) contained within this snapshot
grp_ids = E.read_array('PARTDATA', path, snap, 'PartType' + str(part_type) + '/GroupNumber', numThreads=8)
subgrp_ids = E.read_array('PARTDATA', path, snap, 'PartType' + str(part_type) + '/SubGroupNumber', numThreads=8)

# Convert to group.subgroup ID format
halo_ids = np.zeros(grp_ids.size, dtype=float)
for (ind, g), sg in zip(enumerate(grp_ids), subgrp_ids):
    halo_ids[ind] = float(str(int(g)) + '.%05d' % int(sg))
    
grp_ids = E.read_array('PARTDATA', path, prog_snap, 'PartType' + str(part_type) + '/GroupNumber', numThreads=8)
subgrp_ids = E.read_array('PARTDATA', path, prog_snap, 'PartType' + str(part_type) + '/SubGroupNumber', numThreads=8)

# Convert to group.subgroup ID format
prog_ids = np.zeros(grp_ids.size, dtype=float)
for (ind, g), sg in zip(enumerate(grp_ids), subgrp_ids):
    prog_ids[ind] = float(str(int(g)) + '.%05d' % int(sg))
    
grp_ids = E.read_array('PARTDATA', path, desc_snap, 'PartType' + str(part_type) + '/GroupNumber', numThreads=8)
subgrp_ids = E.read_array('PARTDATA', path, desc_snap, 'PartType' + str(part_type) + '/SubGroupNumber', numThreads=8)

# Convert to group.subgroup ID format
desc_ids = np.zeros(grp_ids.size, dtype=float)
for (ind, g), sg in zip(enumerate(grp_ids), subgrp_ids):
    desc_ids[ind] = float(str(int(g)) + '.%05d' % int(sg))
    
snap_pos = snap_poss[halo_ids == insane_ID, :] - cop
prog_pos = []
desc_pos = []
for prog in progs:
    print("prog", prog)
    prog_pos.extend(prog_poss[prog_ids == prog, :] - cop)
for desc in descs:
    print("desc", desc)
    desc_pos.extend(desc_poss[desc_ids == desc, :] - cop)

width = 0.5

prog_pos = np.array(prog_pos)
desc_pos = np.array(desc_pos)

bins = np.linspace(-width / 2, width / 2, int(width / csoft))
print(len(bins), "pixels in", width, "Mpc")

fig = plt.figure()
ax1 = fig.add_subplot(131)
ax2 = fig.add_subplot(132)
ax3 = fig.add_subplot(133)

H, _, _ = np.histogram2d(prog_pos[:, 0], prog_pos[:, 1], bins=bins)

ax1.imshow(np.arcsinh(H), cmap='Greys_r', extent=[-width / 2, width / 2, -width / 2, width / 2])

H, _, _ = np.histogram2d(snap_pos[:, 0], snap_pos[:, 1], bins=bins)

ax2.imshow(np.arcsinh(H), cmap='Greys_r', extent=[-width / 2, width / 2, -width / 2, width / 2])

H, _, _ = np.histogram2d(desc_pos[:, 0], desc_pos[:, 1], bins=bins)

ax3.imshow(np.arcsinh(H), cmap='Greys_r', extent=[-width / 2, width / 2, -width / 2, width / 2])

ax1.set_title("Progenitor")
ax2.set_title("Current")
ax3.set_title("Descendant")

fig.savefig("plots/Insanityprobe.png", bbox_inches='tight')
