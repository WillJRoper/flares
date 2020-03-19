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

    # Get galaxy particle indices
    gal_inds = np.where(IDs == ID)[0]

    # Get particle positions
    gal_poss = poss[gal_inds, :]

    return gal_poss


def get_parts_around_gal(ID, poss, IDs):
    # Get galaxy particle indices
    gal_inds = np.where(IDs == ID)[0]

    # Get particle positions
    gal_poss = poss[gal_inds, :]

    return gal_poss


def create_img(ID, res, )


