"""
Download data from periodic box simulations
"""

import sys
import numpy as np
from eagle_IO import eagle_IO as E

import flares
from h5py_utilities import write_data_h5py, create_group_h5py, check_h5py

fl = flares.flares()
fname = 'data/periodic.h5'

overwrite = True
threads = 8

for box in ['ref','agn']:
    create_group_h5py(fname, box)
    create_group_h5py(fname, '%s/mstar'%box)
    create_group_h5py(fname, '%s/sfr'%box)
    create_group_h5py(fname, '%s/centrals'%box)


for tag in fl.ref_tags[:2]:
    print(tag)

    ## Reference ##
    if (check_h5py(fname, 'ref/mstar/%s'%tag) is False) |\
                            (overwrite == True):

        mstar = E.read_array("SUBFIND", fl.ref_directory, tag, "/Subhalo/Stars/Mass", 
                            numThreads=threads, noH=True)
        write_data_h5py(fname, 'ref/mstar', tag, data=mstar, overwrite=True)


    if (check_h5py(fname, 'ref/sfr/%s'%tag) is False)  |\
                            (overwrite == True):
                                
        sfr = E.read_array("SUBFIND", fl.ref_directory, tag, "/Subhalo/StarFormationRate", 
                          numThreads=threads, noH=True)
        write_data_h5py(fname, 'ref/sfr', tag, data=sfr, overwrite=True)


    if (check_h5py(fname, 'ref/centrals/%s'%tag) is False) |\
                            (overwrite == True):

        centrals = (E.read_array("SUBFIND", fl.ref_directory, tag, "/Subhalo/SubGroupNumber", 
                                numThreads=threads, noH=True) == 0)
        write_data_h5py(fname, 'ref/centrals', tag, data=centrals, overwrite=True)


#     ## AGNdT9 ## 
#     if (check_h5py(fname, 'agn/mstar/%s'%tag) is False)  |\
#                             (overwrite == True):
# 
#         mstar = E.read_array("SUBFIND", fl.agn_directory, tag, "/Subhalo/Stars/Mass", 
#                             numThreads=threads, noH=True)
#         write_data_h5py(fname, 'agn/mstar', tag, data=mstar, overwrite=True)
#         
#         
#     if (check_h5py(fname, 'agn/sfr/%s'%tag) is False) |\
#                             (overwrite == True):
# 
#         sfr = E.read_array("SUBFIND", fl.agn_directory, tag, "/Subhalo/StarFormationRate", 
#                           numThreads=threads, noH=True)
#         write_data_h5py(fname, 'agn/sfr', tag, data=sfr, overwrite=True)
# 
# 
#     if (check_h5py(fname, 'agn/centrals/%s'%tag) is False) |\
#                             (overwrite == True):
# 
#         centrals = (E.read_array("SUBFIND", fl.agn_directory, tag, "/Subhalo/SubGroupNumber", 
#                                 numThreads=threads, noH=True) == 0)
#         write_data_h5py(fname, 'agn/centrals', tag, data=centrals, overwrite=True)


