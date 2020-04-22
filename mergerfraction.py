#!/cosma/home/dp004/dc-rope1/.conda/envs/flares-env/bin/python
import matplotlib as ml
ml.use('Agg')
import numpy as np
import eagle_IO as E
import h5py


reg = '00'
snap = '010_z005p000'

graphpath = '/cosma/home/dp004/dc-rope1/FLARES/FLARES-1/MergerGraphs/GEAGLE_' + reg + '/'

path = '/cosma/home/dp004/dc-rope1/FLARES/FLARES-1/G-EAGLE_' + reg + '/data'

# Get halo IDs and halo data
subgrp_ids = E.read_array('SUBFIND', path, snap, 'Subhalo/SubGroupNumber', numThreads=8)
grp_ids = E.read_array('SUBFIND', path, snap, 'Subhalo/GroupNumber', numThreads=8)
gal_ms = E.read_array('SUBFIND', path, snap, 'Subhalo/ApertureMeasurements/Mass/030kpc',
                      noH=True, physicalUnits=True, numThreads=8)[:, 4] * 10 ** 10

# Remove particles not associated to a subgroup
okinds = np.logical_and(subgrp_ids != 1073741824, np.logical_and(gal_ms > 1e8))
gal_ms = gal_ms[okinds]
grp_ids = grp_ids[okinds]
subgrp_ids = subgrp_ids[okinds]
halo_ids = np.zeros(grp_ids.size, dtype=float)
for (ind, g), sg in zip(enumerate(grp_ids), subgrp_ids):
    halo_ids[ind] = float(str(int(g)) + '.%05d' % int(sg))

# Initialise merger counters
no_merger = 0
merger = 0

# Open graph file
hdf = h5py.File(graphpath + 'SubMgraph_' + snap + '.hdf5', 'r')

# Loop over galaxies in the file
for gal in halo_ids:

    print(gal, end='\r')
    prog_conts = hdf[gal]['prog_stellar_mass_contribution'][...] * 10 ** 10
    nProg = len(prog_conts[prog_conts > 0])

    if np.sum(hdf[gal]['prog_stellar_mass_contribution'][...] * 10 ** 10) == 0:
        continue

    if nProg > 1:
        merger += 1
    elif nProg == 1:
        no_merger += 1
    else:
        continue

print(merger)
print(no_merger)
print(merger / no_merger)