#!/cosma/home/dp004/dc-rope1/.conda/envs/flares-env/bin/python
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import astropy.units as u
from matplotlib.colors import LogNorm
import matplotlib.gridspec as gridspec
import eagle_IO as E
import seaborn as sns
import pickle
import itertools
matplotlib.use('Agg')

sns.set_style('whitegrid')


def get_parts_in_gal(ID, poss, IDs):

    # Get particle positions
    gal_poss = poss[IDs == ID, :]

    return gal_poss


def get_parts_around_gal(all_poss, mean, lim):

    # Get galaxy particle indices
    seps = np.abs(all_poss - mean)
    xcond = seps[:, 0] < lim
    ycond = seps[:, 1] < lim
    zcond = seps[:, 2] < lim

    # Get particle positions
    surnd_poss = all_poss[np.logical_and(xcond, ycond, zcond), :]

    return surnd_poss


def create_img(ID, res, all_poss, poss, IDs):

    # Get the galaxy particle positions
    gal_poss = get_parts_in_gal(ID, poss, IDs)

    # Compute mean particle position
    mean = gal_poss.mean(axis=0)

    # Centre galaxy on mean
    gal_poss -= mean

    # Find the max and minimum position on each axis
    xmax, xmin = np.max(gal_poss[:, 0]), np.min(gal_poss[:, 0])
    ymax, ymin = np.max(gal_poss[:, 1]), np.min(gal_poss[:, 1])
    zmax, zmin = np.max(gal_poss[:, 2]), np.min(gal_poss[:, 2])

    # Set up lists of mins and maximums
    mins = [xmin, ymin, zmin]
    maxs = [xmax, ymax, ymax]

    # Compute extent 3D
    lim = np.max([np.abs(xmin), np.abs(ymin), np.abs(zmin), xmax, ymax, zmax]) + 0.05

    # Get the surrounding distribution
    surnd_poss = get_parts_around_gal(all_poss, mean, lim)

    # Centre particle distribution
    surnd_poss -= mean

    # Set up dictionaries to store images
    galimgs = {}
    surundimgs = {}
    extents = {}

    for (i, j) in [(0, 1), (0, 2), (1, 2)]:

        # Compute extent for the 2D square image
        dim = np.max([np.abs(mins[i]), np.abs(mins[j]), maxs[i], maxs[j]]) + 0.05
        extents[str(i) + '-' + str(j)] = [-dim, dim, -dim, dim]
        posrange = ((-dim, dim), (-dim, dim))

        # Create images
        galimgs[str(i) + '-' + str(j)], _, _ = np.histogram2d(gal_poss[:, [i, j]], bins=dim / res, range=posrange)
        surundimgs[str(i) + '-' + str(j)], _, _ = np.histogram2d(surnd_poss[:, [i, j]], bins=dim / res, range=posrange)


    return galimgs, surundimgs, extents


def img_main(path, reg, snap, res, part_type):

    # Load all necessary arrays
    subgrp_ids = E.read_array('PARTDATA', path, snap, 'PartType' + str(part_type) + '/SubGroupNumber', numThreads=8)

    # Get IDs


    # Get the images
    galimgs, surundimgs, extents = create_img(ID, res, all_poss, poss, subgrp_ids)

