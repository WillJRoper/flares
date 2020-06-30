#!/cosma/home/dp004/dc-rope1/.conda/envs/flares-env/bin/python
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from matplotlib.collections import LineCollection
import h5py
import eagle_IO.eagle_IO as E
import seaborn as sns
matplotlib.use('Agg')


sns.set_style('whitegrid')


def main():

    regions = []
    for reg in range(33, 34):
        if reg < 10:
            regions.append('0' + str(reg))
        else:
            regions.append(str(reg))

    # snaps = ['005_z010p000', '006_z009p000', '007_z008p000', '008_z007p000', '009_z006p000', '010_z005p000']
    snaps = ['005_z010p000', ]

    reg_snaps = []
    for reg in reversed(regions):

        for snap in snaps:
            reg_snaps.append((reg, snap))

    mine = []
    aswins = []

    aswins_path = '/cosma7/data/dp004/dc-payy1/my_files/flares_pipeline/data/flares.hdf5'

    a_hdf = h5py.File(aswins_path, 'r')

    for reg, snap in reg_snaps:

        print(reg, snap)

        my_path = '/cosma/home/dp004/dc-rope1/FLARES/FLARES-1/WebbData/GEAGLE_' + reg + '/RestUV' + snap + '.hdf5'

        try:
            aswins.extend(a_hdf[f'{reg}/{snap}/Galaxy/BPASS_2.2.1/Chabrier300/Luminosity/DustModelI/FUV'][...])

            my_hdf = h5py.File(my_path, 'r')
            mine.extend(my_hdf['FAKE.TH.FUV']['Aperture_Luminosity_30kpc'][:, 0])

            my_hdf.close()
        except OSError:
            print("No File")
        except KeyError:
            print("Key Error")

    a_hdf.close()

    print("lengths", len(mine), len(aswins))

    bins = np.logspace(26, 31, 200)
    bin_wid = bins[1] - bins[0]
    bin_cents = bins[1:] - bin_wid / 2

    # Set up figure
    fig = plt.figure()
    ax = fig.add_subplot(111)

    H, _ = np.histogram(mine, bins=bins)

    ax.loglog(bin_cents, H, label='Will')

    H, _ = np.histogram(aswins, bins=bins)

    ax.loglog(bin_cents, H, label='Aswin', linestyle='--')

    ax.set_xlabel("$L_{FUV} / $(erg s$^{-1}$ Hz$^{-1}$)")
    ax.set_ylabel("$N$")

    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels, loc='best')

    fig.savefig("plots/mine_aswin_comp.png")


main()
