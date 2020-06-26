#!/cosma/home/dp004/dc-rope1/.conda/envs/flares-env/bin/python
import numpy as np
import matplotlib
from eagle_IO import eagle_IO as E
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
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
print(insane_ID)
prog_start_ind = hdf['Prog_Start_Index'][halo_ind]
desc_start_ind = hdf['Desc_Start_Index'][halo_ind]
ndesc = hdf['nDescs'][halo_ind]
progs = hdf['prog_halo_ids'][prog_start_ind: prog_start_ind + nprog]
descs = hdf['desc_halo_ids'][desc_start_ind: desc_start_ind + ndesc]

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
print(cop)
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

grp_ids = E.read_array('SUBFIND', path, prog_snap, 'Subhalo/GroupNumber', numThreads=8)
subgrp_ids = E.read_array('SUBFIND', path, prog_snap, 'Subhalo/SubGroupNumber', numThreads=8)

# Convert to group.subgroup ID format
subfind_prog_ids = np.zeros(grp_ids.size, dtype=float)
for (ind, g), sg in zip(enumerate(grp_ids), subgrp_ids):
    subfind_prog_ids[ind] = float(str(int(g)) + '.%05d' % int(sg))
    
grp_ids = E.read_array('PARTDATA', path, desc_snap, 'PartType' + str(part_type) + '/GroupNumber', numThreads=8)
subgrp_ids = E.read_array('PARTDATA', path, desc_snap, 'PartType' + str(part_type) + '/SubGroupNumber', numThreads=8)

# Convert to group.subgroup ID format
desc_ids = np.zeros(grp_ids.size, dtype=float)
for (ind, g), sg in zip(enumerate(grp_ids), subgrp_ids):
    desc_ids[ind] = float(str(int(g)) + '.%05d' % int(sg))

grp_ids = E.read_array('SUBFIND', path, desc_snap, 'Subhalo/GroupNumber', numThreads=8)
subgrp_ids = E.read_array('SUBFIND', path, desc_snap, 'Subhalo/SubGroupNumber', numThreads=8)

# Convert to group.subgroup ID format
subfind_desc_ids = np.zeros(grp_ids.size, dtype=float)
for (ind, g), sg in zip(enumerate(grp_ids), subgrp_ids):
    subfind_desc_ids[ind] = float(str(int(g)) + '.%05d' % int(sg))

prog_cops = E.read_array('SUBFIND', path, prog_snap, 'Subhalo/CentreOfPotential', numThreads=8)
desc_cops = E.read_array('SUBFIND', path, desc_snap, 'Subhalo/CentreOfPotential', numThreads=8)
    
snap_pos = snap_poss[halo_ids == insane_ID, :] - cop
prog_pos = []
desc_pos = []
prog_cop = np.zeros((len(progs), 3))
desc_cop = np.zeros((len(descs), 3))
for ind, prog in enumerate(progs):
    print("prog", prog)
    prog_pos.extend(prog_poss[prog_ids == prog, :] - cop)
    prog_cop[ind] = prog_cops[subfind_prog_ids == prog, :] - cop[0]
for ind, desc in enumerate(descs):
    print("desc", desc)
    desc_pos.extend(desc_poss[desc_ids == desc, :] - cop)
    desc_cop[ind] = desc_cops[subfind_desc_ids == desc, :] - cop[0]

width = 2.5

prog_pos = np.array(prog_pos)
desc_pos = np.array(desc_pos)
print(desc_cop.shape)
print(prog_pos.shape, snap_pos.shape, desc_pos.shape)
prog_len = prog_pos.shape[0]
snap_len = snap_pos.shape[0]
desc_len = desc_pos.shape[0]

bins = np.linspace(-width / 2, width / 2, int(width / csoft))
print(len(bins), "pixels in", width, "Mpc")

fig = plt.figure()
ax1 = fig.add_subplot(131)
ax2 = fig.add_subplot(132)
ax3 = fig.add_subplot(133)

H, _, _ = np.histogram2d(prog_pos[:, 0], prog_pos[:, 1], bins=bins)

print("prog", np.sum(H), np.sum(H) / prog_len)

ax1.imshow(np.arcsinh(H.T), cmap='Greys_r', extent=[-width / 2, width / 2, -width / 2, width / 2], origin='lower')
ax1.scatter(prog_cop[:, 0], prog_cop[:, 1], marker='.', color='r', s=5, alpha=0.4)

H, _, _ = np.histogram2d(snap_pos[:, 0], snap_pos[:, 1], bins=bins)

print("current", np.sum(H), np.sum(H) / snap_len)

ax2.imshow(np.arcsinh(H.T), cmap='Greys_r', extent=[-width / 2, width / 2, -width / 2, width / 2], origin='lower')
ax2.scatter(0, 0, marker='.', color='r', s=5, alpha=0.4)

H, _, _ = np.histogram2d(desc_pos[:, 0], desc_pos[:, 1], bins=bins)

print("desc", np.sum(H), np.sum(H) / desc_len)

ax3.imshow(np.arcsinh(H.T), cmap='Greys_r', extent=[-width / 2, width / 2, -width / 2, width / 2], origin='lower')
ax3.scatter(desc_cop[:, 0], desc_cop[:, 1], marker='.', color='r', s=5, alpha=0.4)

ax1.tick_params(axis='both', left=False, top=False, right=False, bottom=False, labelleft=False,
               labeltop=False, labelright=False, labelbottom=False)
ax2.tick_params(axis='both', left=False, top=False, right=False, bottom=False, labelleft=False,
               labeltop=False, labelright=False, labelbottom=False)
ax3.tick_params(axis='both', left=False, top=False, right=False, bottom=False, labelleft=False,
               labeltop=False, labelright=False, labelbottom=False)

ax1.set_title("Progenitor")
ax2.set_title("Current")
ax3.set_title("Descendant")

fig.savefig("plots/Insanityprobe.png", bbox_inches='tight')

plt.close(fig)

x, y, z = snap_pos[:, 0], snap_pos[:, 1], snap_pos[:, 2]

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(snap_pos[:, 0], snap_pos[:, 1], snap_pos[:, 2], c='r', marker='.')

h, yedges, zedges = np.histogram2d(y, z, bins=50)
h = h.transpose()
normalized_map = plt.cm.Greys_r(h/h.max())

yy, zz = np.meshgrid(yedges, zedges)
xpos = np.min(x) - 0.2 # Plane of histogram
xflat = np.full_like(yy, xpos) 

p = ax.plot_surface(xflat, yy, zz, facecolors=normalized_map, rstride=1, cstride=1, shade=False)

h, xedges, zedges = np.histogram2d(x, z, bins=50)
h = h.transpose()
normalized_map = plt.cm.Greys_r(h/h.max())

xx, zz = np.meshgrid(xedges, zedges)
ypos = np.min(y) - 0.2 # Plane of histogram
yflat = np.full_like(xx, ypos)

p = ax.plot_surface(xx, yflat, zz, facecolors=normalized_map, rstride=1, cstride=1, shade=False)

h, xedges, yedges = np.histogram2d(x, y, bins=50)
h = h.transpose()
normaliyed_map = plt.cm.Greys_r(h/h.max())

xx, yy = np.meshgrid(xedges, yedges)
zpos = np.min(z) - 0.2 # Plane of histogram
zflat = np.full_like(xx, zpos)

p = ax.plot_surface(xx, yy, zflat, facecolors=normaliyed_map, rstride=1, cstride=1, shade=False)

fig.savefig("plots/projectiontest.png", bbox_inches='tight')
