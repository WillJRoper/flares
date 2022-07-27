import h5py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import matplotlib.gridspec as gridspec
import cmasher as cmr


# Open file
hdf = h5py.File("/Users/willroper/Downloads/DustModelI.h5", "r")

# Define colors
cols = [("Webb.NIRCAM.F150W", "Webb.NIRCAM.F200W"),
        ("Webb.NIRCAM.F200W", "Webb.NIRCAM.F277W"),
        ("Webb.NIRCAM.F277W", "Webb.NIRCAM.F356W"),
        ("Webb.NIRCAM.F356W", "Webb.NIRCAM.F444W"),
        ("Webb.NIRCAM.F410W", "Webb.NIRCAM.F444W"),
        ("Webb.NIRCAM.F430W", "Webb.NIRCAM.F444W"),
        ("Webb.NIRCAM.F444W", "Webb.MIRI.F560W"),
        ("Webb.MIRI.F560W", "Webb.MIRI.F770W")]

# Load redshifts
zs = hdf["z"][...]

# Define plot
ncols = len(cols)
fig = plt.figure()
gs = gridspec.GridSpec(nrows=1, ncols=ncols + 1,
                       width_ratios=[20, ] * ncols + [1, ])
gs.update(wspace=0.0, hspace=0.0)
axes = []
cax = fig.add_subplot(gs[-1])
for i in range(ncols):
    axes.append(fig.add_subplot(gs[i]))
    if i > 0:
        axes[i].tick_params("y", left=False, right=False, labelleft=False,
                            labelright=False)
    axes[i].set_ylim(np.min(zs), np.max(zs))
    axes[i].set_xlim(np.min(zs), np.max(zs))
    axes[i].set_xlabel("$z$")
    if i == 0:
        axes[i].set_ylabel("$z$")

# Define plotting parameters
norm = TwoSlopeNorm(vmin=-2, vcenter=0, vmax=2)
extent = [np.min(zs), np.max(zs), np.min(zs), np.max(zs)]

# Loop over colors
for ax, (c1, c2) in zip(axes, cols):

    print(c1, "-", c2)

    # Get fluxes
    f1 = hdf[c1][...]
    f2 = hdf[c2][...]

    # Get color
    col = 2.5 * np.log10(f1 / f2)

    print(col.shape)

    # Compute grid
    XX, YY = np.meshgrid(col, col)

    # Compute residual
    resi = XX - YY

    print(np.min(resi), np.max(resi))

    # Plot heat map
    im = ax.imshow(resi, extent=extent, cmap=cmr.seaweed, norm=norm)


cbar = fig.colorbar(im, cax)
cbar.set_label(r"$A-B(z_{x}) - A-B(z_{y}) $")

fig.savefig("color_confusion.png", bbox_inches="tight", dpi=300)
