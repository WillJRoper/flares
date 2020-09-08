#!/cosma/home/dp004/dc-rope1/.conda/envs/flares-env/bin/python
import glob
import re
import timeit
from functools import partial

import h5py
import numpy as np
import schwimmbad


def get_files(fileType, path, tag):
    
    if fileType in ['FOF', 'FOF_PARTICLES']:
        files = glob.glob("%s/groups_%s/group_tab_%s*.hdf5" % (path, tag, tag))
    elif fileType in ['SNIP_FOF', 'SNIP_FOF_PARTICLES']:
        files = glob.glob("%s/groups_snip_%s/group_snip_tab_%s*.hdf5" % (path, tag, tag))
    elif fileType in ['SUBFIND', 'SUBFIND_GROUP', 'SUBFIND_IDS']:
        files = glob.glob("%s/groups_%s/eagle_subfind_tab_%s*.hdf5" % (path, tag, tag))
    elif fileType in ['SNIP_SUBFIND', 'SNIP_SUBFIND_GROUP', 'SNIP_SUBFIND_IDS']:
        files = glob.glob("%s/groups_snip_%s/eagle_subfind_snip_tab_%s*.hdf5" % (path, tag, tag))
    elif fileType in ['SNIP']:
        files = glob.glob("%s/snipshot_%s/snip_%s*.hdf5" % (path, tag, tag))
    elif fileType in ['SNAP']:
        files = glob.glob("%s/snapshot_%s/snap_%s*.hdf5" % (path, tag, tag))
    elif fileType in ['PARTDATA']:
        files = glob.glob("%s/particledata_%s/eagle_subfind_particles_%s*.hdf5" % (path, tag, tag))
    elif fileType in ['SNIP_PARTDATA']:
        files = glob.glob("%s/particledata_snip_%s/eagle_subfind_snip_particles_%s*.hdf5" % (path, tag, tag))
    else:
        files = []
        ValueError("Type of files not supported")

    print("There are", len(files), "of the form", files[0])

    return files


# def get_datasets(fileType, path, tag):


def get_attrs_datasets(fileType, path, tag):

    # Get all the files
    files = get_files(fileType, path, tag)

    for file in files:

        # Initialise lists to store data that needs to be extracted
        attr_keys = []
        group_attr_keys = []
        dset_keys = []

        with h5py.File(file, 'r') as hf:

            # Get attributes
            root_attrs = list(hf.attrs.keys())
            header_attrs = list(hf['Header'].attrs.keys())

            attr_keys.extend(root_attrs)
            attr_keys.extend(header_attrs)

            # Get datasets
            root_datasets = list(hf.keys())
            header_datasets = list(hf['Header'].keys())

            print(root_datasets)

            for key in root_datasets:
                root_key_datasets = list(hf[key].keys())
                for key1 in root_key_datasets:
                    dset_keys.append(key + "/" + key1)
                root_key_attrs_datasets = list(hf[key].attrs.keys())
                for key1 in root_key_attrs_datasets:
                    dset_keys.append((key, key1))

        print("----------------------------------")
        print(file)
        print(attr_keys)
        print(group_attr_keys)
        print(dset_keys)


path = "/cosma7/data/dp004/FLARES/FLARES-HD/FLARES_HR_26/data/"
tag = "010_z005p000"
fileType = "SUBFIND"

get_attrs_datasets(fileType, path, tag)
