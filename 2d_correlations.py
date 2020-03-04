#!/cosma/home/dp004/dc-rope1/.conda/envs/flares-env/bin/python
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import astropy.units as u
import astropy.constants as const
import eagle_IO as E
import seaborn as sns
from flares import flares
matplotlib.use('Agg')

sns.set_style('whitegrid')

# Load the groups and arrays names
groups = np.genfromtxt('HDF5_Groups.txt', dtype=str, delimiter='\n')
arrnames = np.genfromtxt('HDF5_ArrNames.txt', dtype=str, delimiter='\n')

for g, an in zip(groups, arrnames):

    print(g[2:-1] + '/' + an[2:-1])

    if g[2:-1]