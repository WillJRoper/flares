#!/cosma/home/dp004/dc-rope1/.conda/envs/flares-env/bin/python
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import h5py
import os
import gc
import sys
matplotlib.use('Agg')


def get_ns(graphpath, snap):

    hdf = h5py.File(graphpath + snap + '.hdf5', 'r')

    nprog = hdf['nProgs'][...]
    ndesc = hdf['nDescs'][...]

    halo_ids = hdf['SUBFIND_halo_IDs'][...]

    hdf.close()

    return nprog[halo_ids >= 0], ndesc[halo_ids >= 0]


def main():

    regions = []
    for reg in range(0, 40):
        if reg < 10:
            regions.append('0' + str(reg))
        else:
            regions.append(str(reg))

    snaps = ['000_z015p000', '001_z014p000', '002_z013p000', '003_z012p000', '004_z011p000', '005_z010p000',
             '006_z009p000', '007_z008p000', '008_z007p000', '009_z006p000', '010_z005p000', '011_z004p770']

    reg_snaps = []
    for reg in reversed(regions):

        for snap in snaps:
            reg_snaps.append((reg, snap))

    nprogs = []
    ndescs = []
    nprogs_dict = {}
    ndescs_dict = {}

    for reg, snap in reg_snaps:

        graphpath = '/cosma/home/dp004/dc-rope1/FLARES/FLARES-1/MergerGraphs/GEAGLE_' + reg + 'SubMgraph_'

        nprog, ndesc = get_ns(graphpath, snap)
        nprogs.extend(nprog)
        ndescs.extend(ndesc)
        nprogs_dict.setdefault(reg, []).extend(nprog)
        ndescs_dict.setdefault(reg, []).extend(ndesc)

    progbins, progcounts = np.unique(nprogs)
    descbins, desccounts = np.unique(ndescs)

    fig = plt.figure()
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)

    ax1.bar(progbins, progcounts, width=1, alpha=0.9, color='b', ec='b')
    ax2.bar(descbins, desccounts, width=1, alpha=0.9, color='b', ec='b')

    # Set y-axis scaling to logarithmic
    ax1.set_yscale('log')
    ax2.set_yscale('log')

    # Ensure tick labels are integers
    ax1.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax2.xaxis.set_major_locator(MaxNLocator(integer=True))

    # Label axes
    ax1.set_xlabel(r'$N_{dProg}$')
    ax1.set_ylabel(r'$N$')
    ax2.set_xlabel(r'$N_{dDesc}$')
    ax2.set_ylabel(r'$N$')

    # Save the plot as a png
    fig.savefig('plots/ProgDescHist.png', bbox_inches='tight')

    plt.close(fig)

    regs_ovdens = np.loadtxt("region_overdensity.txt", dtype=float)
    reg_bins = np.digitize(regs_ovdens, bins=np.linspace(-0.5, 1, 6))
    print(reg_bins)
    nprogs_environ = {}
    ndescs_environ = {}

    # for reg in regions:
    #
    # fig = plt.figure()
    # ax1 = fig.add_subplot(121)
    # ax2 = fig.add_subplot(122)
    #
    # for reg in regions:
    #
    #     progbins, progcounts = np.unique(nprogs)
    #     descbins, desccounts = np.unique(ndescs)
    #
    #     ax1.bar(progbins, progcounts, width=1, alpha=0.9, color='b', ec='b')
    #     ax2.bar(descbins, desccounts, width=1, alpha=0.9, color='b', ec='b')
    #
    # # Set y-axis scaling to logarithmic
    # ax1.set_yscale('log')
    # ax2.set_yscale('log')
    #
    # # Ensure tick labels are integers
    # ax1.xaxis.set_major_locator(MaxNLocator(integer=True))
    # ax2.xaxis.set_major_locator(MaxNLocator(integer=True))
    #
    # # Label axes
    # ax1.set_xlabel(r'$N_{dProg}$')
    # ax1.set_ylabel(r'$N$')
    # ax2.set_xlabel(r'$N_{dDesc}$')
    # ax2.set_ylabel(r'$N$')
    #
    # # Draw legend
    # handles, labels = ax1.get_legend_handles_labels()
    # ax1.legend(handles, labels)
    #
    # # Save the plot as a png
    # fig.savefig('plots/ProgDescHist.png', bbox_inches='tight')
