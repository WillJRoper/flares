#!/cosma/home/dp004/dc-rope1/.conda/envs/flares-env/bin/python
import glob
import re
import timeit
from functools import partial

import h5py
import numpy as np
import schwimmbad


def get_files(path, tag):

    files = glob.glob("%s/groups_%s/group_tab_%s*.hdf5" % (path, tag, tag))

    print(files)


path = "/cosma7/data/dp004/FLARES/FLARES-HD/FLARES_HR_26/data/"
tag = "010_z005p000"

get_files(path, tag)
