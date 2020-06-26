#!/cosma/home/dp004/dc-rope1/.conda/envs/flares-env/bin/python
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from matplotlib.collections import LineCollection
import h5py
import seaborn as sns
matplotlib.use('Agg')


sns.set_style('whitegrid')


def multiline(xs, ys, c, ax=None, **kwargs):
    """Plot lines with different colorings

    Parameters
    ----------
    xs : iterable container of x coordinates
    ys : iterable container of y coordinates
    c : iterable container of numbers mapped to colormap
    ax (optional): Axes to plot on.
    kwargs (optional): passed to LineCollection

    Notes:
        len(xs) == len(ys) == len(c) is the number of line segments
        len(xs[i]) == len(ys[i]) is the number of points for each line (indexed by i)

    Returns
    -------
    lc : LineCollection instance.
    """

    # find axes
    ax = plt.gca() if ax is None else ax

    # create LineCollection
    segments = [np.column_stack([x, y]) for x, y in zip(xs, ys)]
    lc = LineCollection(segments, **kwargs)

    # set coloring of line segments
    #    Note: I get an error if I pass c as a list here... not sure why.
    lc.set_array(np.asarray(c))

    # add lines to axes and rescale
    #    Note: adding a collection doesn't autoscalee xlim/ylim
    ax.add_collection(lc)
    ax.autoscale()
    return lc


def get_ns(graphpath, snap):

    hdf = h5py.File(graphpath + snap + '.hdf5', 'r')

    nprog = hdf['nProgs'][...]
    ndesc = hdf['nDescs'][...]
    prog_start_index = hdf['Prog_Start_Index'][...]
    desc_start_index = hdf['Desc_Start_Index'][...]
    prog_conts = hdf['prog_stellar_mass_contribution'][...]
    desc_conts = hdf['desc_stellar_mass_contribution'][...]
    nprogs = []
    ndescs = []

    hdf.close()

    for npg, nd, pstart, dstart in zip(ndesc, nprog, prog_start_index, desc_start_index):
        pconts = prog_conts[pstart: pstart + npg]
        dconts = desc_conts[dstart: dstart + nd]
        nprogs.append(len(pconts[pconts > 0]))
        ndescs.append(len(dconts[dconts > 0]))

    return np.array(nprogs), np.array(ndescs)


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

        graphpath = '/cosma/home/dp004/dc-rope1/FLARES/FLARES-1/MergerGraphs/GEAGLE_' + reg + '/SubMgraph_'

        nprog, ndesc = get_ns(graphpath, snap)
        nprogs.extend(nprog)
        ndescs.extend(ndesc)
        nprogs_dict.setdefault(reg, []).extend(nprog)
        ndescs_dict.setdefault(reg, []).extend(ndesc)

    progbins, progcounts = np.unique(nprogs, return_counts=True)
    descbins, desccounts = np.unique(ndescs, return_counts=True)

    fig = plt.figure(figsize=(12, 6))
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
    bin_edges = np.linspace(regs_ovdens.min(), regs_ovdens.max(), 6)
    reg_bins = np.digitize(regs_ovdens, bins=bin_edges)
    bin_wid = bin_edges[1] - bin_edges[0]
    bin_cents = bin_edges[1:] - bin_wid / 2
    print(reg_bins)
    nprogs_environ = {}
    ndescs_environ = {}

    for reg, bin_ind in zip(regions, reg_bins):

        nprogs_environ.setdefault(bin_ind, []).extend(nprogs_dict[reg])
        ndescs_environ.setdefault(bin_ind, []).extend(ndescs_dict[reg])

    fig = plt.figure(figsize=(12, 6))
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)

    colors = matplotlib.cm.plasma(np.linspace(0, 1, 6))

    plt_prog_xs = []
    plt_prog_ys = []
    plt_desc_xs = []
    plt_desc_ys = []

    for (bin_ind, ovden), col in zip(enumerate(bin_cents), colors):

        progbins, progcounts = np.unique(nprogs_environ[bin_ind + 1], return_counts=True)
        descbins, desccounts = np.unique(ndescs_environ[bin_ind + 1], return_counts=True)

        ax1.plot(progbins, progcounts, color=col)
        ax2.plot(descbins, desccounts, color=col)

        plt_prog_xs.extend(progbins)
        plt_prog_ys.extend(progcounts)
        plt_desc_xs.extend(descbins)
        plt_desc_ys.extend(desccounts)

    lc = multiline(plt_prog_xs, plt_prog_ys, bin_cents, cmap='plasma', ax=ax1)
    lc = multiline(plt_desc_xs, plt_desc_ys, bin_cents, cmap='plasma', ax=ax2)

    axcb = fig.colorbar(lc)
    axcb.set_label('$\Delta$')

    # Set y-axis scaling to logarithmic
    ax1.set_yscale('log')
    ax2.set_yscale('log')
    ax1.set_xscale('log')
    ax2.set_xscale('log')

    # Ensure tick labels are integers
    ax1.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax2.xaxis.set_major_locator(MaxNLocator(integer=True))

    # Label axes
    ax1.set_xlabel(r'$N_{dProg}$')
    ax1.set_ylabel(r'$N$')
    ax2.set_xlabel(r'$N_{dDesc}$')
    ax2.set_ylabel(r'$N$')

    # Draw legend
    handles, labels = ax1.get_legend_handles_labels()
    ax1.legend(handles, labels)

    # Save the plot as a png
    fig.savefig('plots/ProgDescHist_environ.png', bbox_inches='tight')


main()
