import numpy as np
import  eagle_IO.eagle_IO as E
import matplotlib.pyplot as plt


def get_img(path, snap):

    # Get redshift
    z_str = snap.split('z')[1].split('p')
    z = float(z_str[0] + '.' + z_str[1])

    # Define comoving softening length in kpc
    soft = 0.001802390 / 0.677 * 1 / (1 + z)

    # Get spatial data
    star_poss = E.read_array('SNAP', path, snap, 'PartType4/Coordinates', noH=True,
                            physicalUnits=True, numThreads=8)
    dm_poss = E.read_array('SNAP', path, snap, 'PartType4/Coordinates', noH=True,
                            physicalUnits=True, numThreads=8)
    gas_poss = E.read_array('SNAP', path, snap, 'PartType4/Coordinates', noH=True,
                            physicalUnits=True, numThreads=8)

    # Get median
    cent = np.median(dm_poss, axis=0)

    # Centre coordiantes
    star_poss -= cent
    dm_poss -= cent
    gas_poss -= cent

    # Define the bins
    bins = np.arange(-1, 1, soft)

    # Get the image
    H_gas, xgas, ygas = np.histogram2d(star_poss[:, 0], star_poss[:, 1], bins=bins)
    H_dm, xdm, ydm = np.histogram2d(star_poss[:, 0], star_poss[:, 1], bins=bins)
    if len(star_poss) != 0:
        H_star, xstar, ystar = np.histogram2d(star_poss[:, 0], star_poss[:, 1], bins=bins)
    else:
        H_star = np.zeros_like(H_dm)

    return np.arcsinh(H_dm), np.arcsinh(H_gas), np.arcsinh(H_star)

def mainimg(path, snap):

    # Get imgs
    H_dm, H_gas, H_star = get_img(path, snap)

    # Define the extent
    extent = [-1, 1, -1, 1]

    # Set up plot
    fig = plt.figure()
    ax1 = fig.add_subplot(131)
    ax2 = fig.add_subplot(132)
    ax3 = fig.add_subplot(133)

    # Plot images
    ax1.imshow(H_dm, extent=extent, cmap='plasma')
    ax2.imshow(H_gas, extent=extent, cmap='plasma')
    ax3.imshow(H_star, extent=extent, cmap='plasma')

    # Turn off the axis
    ax1.axis(False)
    ax2.axis(False)
    ax3.axis(False)

    # Label the plots
    ax1.set_title('Dark Matter')
    ax2.set_title('Gas')
    ax3.set_title('Stars')

    # Save the figure
    fig.savefig('galaxytest/gal_' + snap + '.png', bbox_inches='tight')

snaps = ['000_z015p000', '001_z014p000', '002_z013p000', '003_z012p000', '004_z011p000', '005_z010p000',
         '006_z009p000', '007_z008p000', '008_z007p000', '009_z006p000', '010_z005p000', '011_z004p770']

path = '/cosma/home/dp004/dc-rope1/cosma7/FLARES/galaxy_test/data/'

for snap in snaps:
    mainimg(path, snap)
