import glob
import re
import timeit
from functools import partial

import h5py
import numpy as np
import schwimmbad


def read_hdf5(f, dataset):
    with h5py.File(f, 'r') as hf:
        dat = np.array(hf.get(dataset))
        if dat.ndim == 0: return np.array([])

    return dat


def get_files(fileType, directory, tag):
    """
    Fetch filename

    Args:
        fileType (str)
        directory (str)
        tag (str)
    """

    if fileType in ['FOF', 'FOF_PARTICLES']:
        files = glob.glob("%s/groups_%s/group_tab_%s*.hdf5" % (directory, tag, tag))
    elif fileType in ['SNIP_FOF', 'SNIP_FOF_PARTICLES']:
        files = glob.glob("%s/groups_snip_%s/group_snip_tab_%s*.hdf5" % (directory, tag, tag))
    elif fileType in ['SUBFIND', 'SUBFIND_GROUP', 'SUBFIND_IDS']:
        files = glob.glob("%s/groups_%s/eagle_subfind_tab_%s*.hdf5" % (directory, tag, tag))
    elif fileType in ['SNIP_SUBFIND', 'SNIP_SUBFIND_GROUP', 'SNIP_SUBFIND_IDS']:
        files = glob.glob("%s/groups_snip_%s/eagle_subfind_snip_tab_%s*.hdf5" % (directory, tag, tag))
    elif fileType in ['SNIP']:
        files = glob.glob("%s/snipshot_%s/snip_%s*.hdf5" % (directory, tag, tag))
    elif fileType in ['SNAP']:
        files = glob.glob("%s/snapshot_%s/snap_%s*.hdf5" % (directory, tag, tag))
    elif fileType in ['PARTDATA']:
        files = glob.glob("%s/particledata_%s/eagle_subfind_particles_%s*.hdf5" % (directory, tag, tag))
    elif fileType in ['SNIP_PARTDATA']:
        files = glob.glob("%s/particledata_snip_%s/eagle_subfind_snip_particles_%s*.hdf5" % (directory, tag, tag))
    else:
        ValueError("Type of files not supported")

    return sorted(files, key=lambda x: int(re.findall("(\d+)", x)[-2]))


def apply_physicalUnits_conversion(f, dataset, dat, verbose=True):
    """

    Args:
        f (str)
        dat (array)
    """
    with h5py.File(f, 'r') as hf:
        exponent = hf[dataset].attrs['aexp-scale-exponent']
        a = hf['Header'].attrs['ExpansionFactor']

    if exponent != 0.:
        if verbose:
            print("Converting to physical units. (Multiplication by a^%f, a=%f)" % (exponent, a))
        return dat * pow(a, exponent)
    else:
        if verbose:
            print("Converting to physical units. No conversion needed!")
        return dat


def apply_hfreeUnits_conversion(f, dataset, dat, verbose=True):
    """

    Args:
        f (str)
        dat (array)
    """
    with h5py.File(f, 'r') as hf:
        exponent = hf[dataset].attrs['h-scale-exponent']
        h = hf['Header'].attrs['HubbleParam']

    if exponent != 0.:
        if verbose:
            print("Converting to h-free units. (Multiplication by h^%f, h=%f)" % (exponent, h))
        return dat * pow(h, exponent)
    else:
        if verbose:
            print("Converting to h-free units. No conversion needed!")
        return dat


def apply_CGSUnits_conversion(f, dataset, dat, verbose=True):
    """
    A function to convert to CGS units.

    :param f:
    :param dataset:
    :param dat:
    :param verbose:
    :return:
    """

    with h5py.File(f, 'r') as hf:
        cgs = hf[dataset].attrs['CGSConversionFactor']

    if cgs != 1.0:
        if verbose:
            print("Converting to CGS units. (Multiplication by CGS conversion factor = %f)" % cgs)
        return dat * cgs
    else:
        if verbose:
            print("Converting to CGS units. No conversion needed!")
        return dat


def read_array(ftype, directory, tag, dataset, numThreads=1, noH=False, physicalUnits=False, CGS=False, verbose=True):
    """
   
    Args:
        ftype (str)
        directory (str)
        tag (str)
        dataset (str)
        numThreads (int)
        noH (bool)
        physicalUnits (bool)
    """
    start = timeit.default_timer()

    files = get_files(ftype, directory, tag)

    if numThreads == 1:
        pool = schwimmbad.SerialPool()
    elif numThreads == -1:
        pool = schwimmbad.MultiPool()
    else:
        pool = schwimmbad.MultiPool(processes=numThreads)

    lg = partial(read_hdf5, dataset=dataset)
    dat = np.concatenate(list(pool.map(lg, files)), axis=0)
    pool.close()

    stop = timeit.default_timer()

    print("Reading in '{}' for z = {} using {} thread(s) took {}s".format(dataset,
                                                                          np.round(read_header(ftype, directory, tag, dataset='Redshift'), 3),
                                                                          numThreads, np.round(stop - start, 6)))

    if noH:
        dat = apply_hfreeUnits_conversion(files[0], dataset, dat, verbose=verbose)

    if physicalUnits:
        dat = apply_physicalUnits_conversion(files[0], dataset, dat, verbose=verbose)
    if CGS:
        dat = apply_CGSUnits_conversion(files[0], dataset, dat, verbose=verbose)
    return dat


def read_header(ftype, directory, tag, dataset):
    """
   
    Args:
        ftype (str)
        directory (str)
        tag (str)
        dataset (str)
    """

    files = get_files(ftype, directory, tag)
    with h5py.File(files[0], 'r') as hf:
        hdr = hf['Header'].attrs[dataset]

    return hdr


if __name__ == "__main__":
    directory = '/cosma5/data/Eagle/ScienceRuns/Planck1/L0050N0752/PE/S15_AGNdT9/data'
    tag = '003_z008p988'

    z = read_header('SUBFIND', directory, tag, 'Redshift')
    mstar = read_array('SUBFIND', directory, tag, "/Subhalo/Stars/Mass", numThreads=1, verbose=True)
    print(mstar.shape)
    print(mstar[:10])