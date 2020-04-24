#!/cosma/home/dp004/dc-rope1/.conda/envs/flares-env/bin/python
import numpy as np
import eagle_IO.eagle_IO as E
from scipy.optimize import curve_fit
from scipy.spatial import ConvexHull


def _sphere(coords, a, b, c, r):

    # Equation of a sphere
    x, y, z = coords[:, 0], coords[:, 1], coords[:, 2]

    return (x - a) ** 2 + (y - b) ** 2 + (z - c) ** 2 - r ** 2


def spherical_region(sim, snap):
    """
    Inspired from David Turner's suggestion
    """

    dm_cood = E.read_array('PARTDATA', sim, snap, '/PartType1/Coordinates',
                           noH=True, physicalUnits=False, numThreads=8)  # dm particle coordinates

    hull = ConvexHull(dm_cood)

    cen = [np.median(dm_cood[:, 0]), np.median(dm_cood[:, 1]), np.median(dm_cood[:,2])]
    pedge = dm_cood[hull.vertices]  #edge particles
    y_obs = np.zeros(len(pedge))
    p0 = np.append(cen, 15 / 0.677)

    popt, pcov = curve_fit(_sphere, pedge, y_obs, p0, method='lm', sigma=np.ones(len(pedge)) * 0.001)
    dist = np.sqrt(np.sum((pedge-popt[:3])**2, axis=1))
    centre, radius, mindist = popt[:3], popt[3], np.min(dist)

    return centre, radius, mindist


def get_fullbox_gal_cents(reg, snap='011_z004p770', mass_lim=10**9.5):

    # Open region centres
    reg_cents = np.loadtxt('Region_cents.txt', dtype=float)
    print(reg_cents)
    this_region = reg_cents[int(reg), :]

    # Define the path
    path = '/cosma/home/dp004/dc-rope1/FLARES/FLARES-1/G-EAGLE_' + reg + '/data'

    # Get the galaxy data
    gal_ms = E.read_array('SUBFIND', path, snap, 'Subhalo/ApertureMeasurements/Mass/030kpc',
                          noH=True, physicalUnits=False, numThreads=8)[:, 4] * 10 ** 10
    gal_cops = E.read_array('SUBFIND', path, snap, 'Subhalo/CentreOfPotential', noH=True,
                            physicalUnits=False, numThreads=8)

    # Perform mass cut
    gal_cops = gal_cops[gal_ms > mass_lim]

    # Get region centre
    centre, radius, mindist = spherical_region(path, snap)
    gal_cops -= centre

    # Calculate the full box
    gal_cops += this_region

    # Save a text file
    np.savetxt('galaxy_cents.txt', gal_cops)


get_fullbox_gal_cents(reg=0, snap='011_z004p770', mass_lim=10**9.5)
