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


def read_groupdset_hdf5(f, key):
    """

    Args:
        ftype (str)
        directory (str)
        tag (str)
        dataset (str)
    """

    num = f.split(".")[-2]

    with h5py.File(f, 'r') as hf:
        dat = hf[key][...]
        value = dat
        file_num = num
        attr_key = key

    return file_num, attr_key, value


def read_attr_hdf5(f, key):
    """

    Args:
        ftype (str)
        directory (str)
        tag (str)
        dataset (str)
    """

    num = f.split(".")[-2]

    with h5py.File(f, 'r') as hf:
        dat = hf[key[0]].attrs[key[1]]
        value = dat
        file_num = num
        attr_key = key

    return file_num, attr_key, value


def read_rootattr_hdf5(f, key):
    """

    Args:
        ftype (str)
        directory (str)
        tag (str)
        dataset (str)
    """

    num = f.split(".")[-2]

    with h5py.File(f, 'r') as hf:
        dat = hf.attrs[key]
        value = dat
        file_num = num
        attr_key = key

    return file_num, attr_key, value


def read_multi(fileType, path, tag, numThreads=8):

    start = timeit.default_timer()

    results = {}

    key_dict, files = get_attrs_datasets(fileType, path, tag)

    for key in key_dict["groupattr"]:

        if numThreads == 1:
            pool = schwimmbad.SerialPool()
        elif numThreads == -1:
            pool = schwimmbad.MultiPool()
        else:
            pool = schwimmbad.MultiPool(processes=numThreads)

        lg = partial(read_attr_hdf5, key=key)
        dat = list(pool.map(lg, files))
        pool.close()
        results[key] = dat

    for key in key_dict["rootattr"]:

        if numThreads == 1:
            pool = schwimmbad.SerialPool()
        elif numThreads == -1:
            pool = schwimmbad.MultiPool()
        else:
            pool = schwimmbad.MultiPool(processes=numThreads)

        lg = partial(read_rootattr_hdf5, key=key)
        dat = list(pool.map(lg, files))
        pool.close()
        results[key] = dat

    for key in key_dict["groupdset"]:

        if numThreads == 1:
            pool = schwimmbad.SerialPool()
        elif numThreads == -1:
            pool = schwimmbad.MultiPool()
        else:
            pool = schwimmbad.MultiPool(processes=numThreads)

        lg = partial(read_attr_hdf5, key=key)
        dat = list(pool.map(lg, files))
        pool.close()
        results[key] = dat

    print(results)

def get_attrs_datasets(fileType, path, tag):

    # Get all the files
    files = get_files(fileType, path, tag)

    keys_dict = {}

    for file in files:

        # Initialise lists to store data that needs to be extracted
        attr_keys = []
        group_attr_keys = []
        dset_keys = []

        with h5py.File(file, 'r') as hf:

            # Get attributes
            root_attrs = list(hf.attrs.keys())

            attr_keys.extend(root_attrs)

            # Get datasets
            root_groups = list(hf.keys())

            for key in root_groups:
                root_key_datasets = list(hf[key].keys())
                for key1 in root_key_datasets:
                    dset_keys.append(key + "/" + key1)
                root_key_attrs_datasets = list(hf[key].attrs.keys())
                for key1 in root_key_attrs_datasets:
                    group_attr_keys.append((key, key1))

        keys_dict[file] = {"rootattr": attr_keys, "groupattr": group_attr_keys, "groupdset": dset_keys}

    return keys_dict, files


path = "/cosma7/data/dp004/FLARES/FLARES-HD/FLARES_HR_26/data/"
tag = "010_z005p000"
fileType = "SUBFIND"

read_multi(fileType, path, tag, numThreads=8)
