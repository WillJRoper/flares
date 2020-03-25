#!/cosma/home/dp004/dc-rope1/.conda/envs/flares-env/bin/python
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib
import astropy.units as u
import eagle_IO as E
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import pickle
import os
import gc
from utilities import calc_ages, get_Z_LOS
from astropy.cosmology import Planck13 as cosmo
from webb_imgs import createSimpleImgs, createPSFdImgs
import seaborn as sns
os.environ['FLARE'] = '/cosma7/data/dp004/dc-wilk2/flare'
import FLARE.filters
from SynthObs.SED import models
matplotlib.use('Agg')


sns.set_style('white')

# Define SED model
model = models.define_model('BPASSv2.2.1.binary/ModSalpeter_300',
                            path_to_SPS_grid = FLARE.FLARE_dir + '/data/SPS/nebular/3.0/') # DEFINE SED GRID -
model.dust_ISM = ('simple', {'slope': -1.0})
model.dust_BC = ('simple', {'slope': -1.0})
filters = FLARE.filters.NIRCam
F = FLARE.filters.add_filters(filters, new_lam = model.lam)

# --- create new L grid for each filter. In units of erg/s/Hz
model.create_Lnu_grid(F)


def create_img(gal_poss, arc_res, ini_width, gal_ms, gal_ages, gal_mets, gal_smls, gas_mets, gas_poss, gas_ms, gas_sml,
               lkernel, kbins, conv, redshift, NIRCfs, model, F, output, psf):

    # Set up dictionaries to store images
    galimgs = {}
    extents = {}
    ls = {}

    for (i, j) in [(0, 1), (0, 2), (1, 2)]:

        # Define dimensions array
        if i == 0 and j == 1:
            k = 2
        elif i == 0 and j == 2:
            k = 1
        else:
            k = 0
        dimens = np.array([i, j, k])

        gal_met_surfden = get_Z_LOS(gal_poss, gas_poss, gas_ms, gas_mets, gas_sml, dimens, lkernel, kbins, conv)

        galimgs[str(i) + '-' + str(j)] = {}
        ls[str(i) + '-' + str(j)] = {}

        for f in NIRCfs:

            result = createSimpleImgs(gal_poss[:, i], gal_poss[:, j], gal_ms, gal_ages, gal_mets, gal_met_surfden,
                                      gal_smls, redshift, arc_res, ini_width, f, model, F, output)
            galimgs[str(i) + '-' + str(j)][f] = createPSFdImgs(result[0], arc_res, f, redshift, result[-1])
            extents[str(i) + '-' + str(j)], ls[str(i) + '-' + str(j)][f] = result[1: -1]

    return galimgs, extents, ls


def img_main(path, snap, reg, arc_res, model, F, output=True, psf=True, npart_lim=10**3, dim=0.1, load=True,
             conv=1, scale=0.1, NIRCfs=(None, )):

    # Get the redshift
    z_str = snap.split('z')[1].split('p')
    z = float(z_str[0] + '.' + z_str[1])

    model.create_Fnu_grid(F, z, cosmo)

    # Define stellar particle type
    part_type = 4

    kinp = np.load('/cosma/home/dp004/dc-rope1/cosma7/FLARES/flares/los_extinction/kernel_sph-anarchy.npz',
                   allow_pickle=True)
    lkernel = kinp['kernel']
    header = kinp['header']
    kbins = header.item()['bins']

    if load:

        with open('UVimg_data/stellardata_reg' + reg + '_snap'
                  + snap + '_npartgreaterthan' + str(npart_lim) + '.pck', 'rb') as pfile1:
            save_dict = pickle.load(pfile1)

        gal_ages = save_dict['gal_ages']
        gal_mets = save_dict['gal_mets']
        gal_ms = save_dict['gal_ms']
        gas_mets = save_dict['gas_mets']
        gas_ms = save_dict['gas_ms']
        gal_smls = save_dict['gal_smls']
        gas_smls = save_dict['gas_smls']
        all_gas_poss = save_dict['all_gas_poss']
        all_gal_poss = save_dict['all_gal_poss']
        means = save_dict['means']

    else:

        # Load all necessary arrays
        subgrp_ids = E.read_array('PARTDATA', path, snap, 'PartType' + str(part_type) + '/SubGroupNumber', numThreads=8)
        all_poss = E.read_array('SNAP', path, snap, 'PartType' + str(part_type) + '/Coordinates', noH=True, numThreads=8)
        part_ids = E.read_array('SNAP', path, snap, 'PartType' + str(part_type) + '/ParticleIDs', numThreads=8)
        gal_sml = E.read_array('SNAP', path, snap, 'PartType' + str(part_type) + '/SmoothingLength', numThreads=8)
        group_part_ids = E.read_array('PARTDATA', path, snap, 'PartType' + str(part_type) + '/ParticleIDs', numThreads=8)
        grp_ids = E.read_array('PARTDATA', path, snap, 'PartType' + str(part_type) + '/GroupNumber', numThreads=8)
        # gal_ids = E.read_array('SUBFIND', path, snap, 'Subhalo/SubGroupNumber', numThreads=8)
        # gal_gids = E.read_array('SUBFIND', path, snap, 'Subhalo/GroupNumber', numThreads=8)
        # gal_cops = E.read_array('SUBFIND', path, snap, 'Subhalo/CentreOfPotential', noH=True, numThreads=8)
        halo_ids = np.zeros_like(grp_ids, dtype=float)
        for (ind, g), sg in zip(enumerate(grp_ids), subgrp_ids):
            halo_ids[ind] = float(str(g) + '.' + str(sg + 1))

        # # Get centre of potentials
        # gal_cop = {}
        # for cop, g, sg in zip(gal_cops, gal_gids, gal_ids):
        #     gal_cop[float(str(g) + '.' + str(sg + 1))] = cop

        # Translate ID into indices
        ind_to_pid = {}
        pid_to_ind = {}
        for ind, pid in enumerate(part_ids):
            if pid in group_part_ids:
                ind_to_pid[ind] = pid
                pid_to_ind[pid] = ind

        # Get the IDs above the npart threshold
        ids, counts = np.unique(halo_ids, return_counts=True)
        ids = set(ids[counts > npart_lim])

        # Get the particles in the halos
        halo_id_part_inds = {}
        for pid, simid in zip(group_part_ids, halo_ids):
            if simid not in ids:
                continue
            if int(str(simid).split('.')[1]) == 2**30:
                continue
            try:
                halo_id_part_inds.setdefault(simid, set()).update({pid_to_ind[pid]})
            except KeyError:
                ind_to_pid[len(part_ids) + 1] = pid
                pid_to_ind[pid] = len(part_ids) + 1
                halo_id_part_inds.setdefault(simid, set()).update({pid_to_ind[pid]})

        del group_part_ids, halo_ids, pid_to_ind, ind_to_pid, subgrp_ids, part_ids

        gc.collect()

        print('There are', len(ids), 'galaxies above the cutoff')

        # If there are no galaxies exit
        if len(ids) == 0:
            return

        # Load data for luminosities
        a_born = E.read_array('SNAP', path, snap, 'PartType4/StellarFormationTime', noH=True, numThreads=8)
        metallicities = E.read_array('SNAP', path, snap, 'PartType4/SmoothedMetallicity', noH=True, numThreads=8)
        masses = E.read_array('SNAP', path, snap, 'PartType4/Mass', noH=True, numThreads=8) * 10**10

        # Calculate ages
        ages = calc_ages(z, a_born)

        # Get the position of each of these galaxies
        gal_ages = {}
        gal_mets = {}
        gal_ms = {}
        gal_smls = {}
        all_gal_poss = {}
        means = {}
        for id in ids:
            parts = list(halo_id_part_inds[id])
            all_gal_poss[id] = all_poss[parts, :]
            gal_ages[id] = ages[parts]
            gal_mets[id] = metallicities[parts]
            gal_ms[id] = masses[parts]
            gal_smls[id] = gal_sml[parts]
            means[id] = all_gal_poss[id].mean(axis=0)

        print('Got galaxy properties')

        del ages, all_poss, metallicities, masses, gal_sml, halo_id_part_inds, a_born

        gc.collect()

        # Get gas particle information
        gsubgrp_ids = E.read_array('PARTDATA', path, snap, 'PartType0/SubGroupNumber', numThreads=8)
        gas_all_poss = E.read_array('SNAP', path, snap, 'PartType0/Coordinates', noH=True, numThreads=8)
        gpart_ids = E.read_array('SNAP', path, snap, 'PartType0/ParticleIDs', numThreads=8)
        ggroup_part_ids = E.read_array('PARTDATA', path, snap, 'PartType0/ParticleIDs', numThreads=8)
        ggrp_ids = E.read_array('PARTDATA', path, snap, 'PartType0/GroupNumber', numThreads=8)
        gas_metallicities = E.read_array('SNAP', path, snap, 'PartType0/SmoothedMetallicity', noH=True, numThreads=8)
        gas_smooth_ls = E.read_array('SNAP', path, snap, 'PartType0/SmoothingLength', noH=True, numThreads=8)
        gas_masses = E.read_array('SNAP', path, snap, 'PartType0/Mass', noH=True, numThreads=8) * 10**10
        ghalo_ids = np.zeros_like(ggrp_ids, dtype=float)
        for (ind, g), sg in zip(enumerate(ggrp_ids), gsubgrp_ids):
            ghalo_ids[ind] = float(str(g) + '.' + str(sg + 1))

        print('Got halo IDs')
        set_ggroup_part_ids = set(ggroup_part_ids)
        # Translate ID into indices
        gind_to_pid = {}
        gpid_to_ind = {}
        for ind, pid in enumerate(gpart_ids):
            if pid in set_ggroup_part_ids:
                gind_to_pid[ind] = pid
                gpid_to_ind[pid] = ind

        print('Made ID dicts')

        # Get the particles in the halos
        ghalo_id_part_inds = {}
        for pid, simid in zip(ggroup_part_ids, ghalo_ids):
            if simid not in ids:
                continue
            if int(str(simid).split('.')[1]) == 2**30:
                continue
            try:
                ghalo_id_part_inds.setdefault(simid, set()).update({gpid_to_ind[pid]})
            except KeyError:
                gind_to_pid[len(gpart_ids) + 1] = pid
                gpid_to_ind[pid] = len(gpart_ids) + 1
                ghalo_id_part_inds.setdefault(simid, set()).update({gpid_to_ind[pid]})

        print('Got particle IDs')

        del ggroup_part_ids, ghalo_ids, gpid_to_ind, gind_to_pid, gsubgrp_ids, gpart_ids

        gc.collect()

        # Get the position of each of these galaxies
        gas_mets = {}
        gas_ms = {}
        gas_smls = {}
        all_gas_poss = {}
        for id in ids:
            gparts = list(ghalo_id_part_inds[id])
            all_gas_poss[id] = gas_all_poss[gparts, :]
            gas_mets[id] = gas_metallicities[gparts]
            gas_ms[id] = gas_masses[gparts]
            gas_smls[id] = gas_smooth_ls[gparts]

        print('Got gas properties')

        del gas_smooth_ls, gas_masses, gas_metallicities, gas_all_poss, ghalo_id_part_inds

        gc.collect()

        save_dict = {'gal_ages': gal_ages, 'gal_mets': gal_mets, 'gal_ms': gal_ms, 'gas_mets': gas_mets,
                     'gas_ms': gas_ms, 'gal_smls': gal_smls, 'gas_smls': gas_smls, 'all_gas_poss': all_gas_poss,
                     'all_gal_poss': all_gal_poss, 'means': means}

        with open('UVimg_data/stellardata_reg' + reg + '_snap'
                  + snap + '_npartgreaterthan' + str(npart_lim) + '.pck', 'wb') as pfile1:
            pickle.dump(save_dict, pfile1)

        print('Written out properties')

    print('Extracted galaxy positions')

    axlabels = [r'$x$', r'$y$', r'$z$']

    # Create images for these galaxies
    for id in gal_ages.keys():

        print('Computing images for', id)

        # Get the images
        galimgs, extents, ls = create_img(all_gal_poss[id], arc_res, dim, gal_ms[id], gal_ages[id],
                                          gal_mets[id], gal_smls[id], gas_mets[id], all_gas_poss[id], gas_ms[id],
                                          gas_smls[id], lkernel, kbins, conv, z, NIRCfs, model, F, output, psf)

        # Loop over dimensions
        for key in galimgs.keys():

            # Set up figure
            if len(NIRCfs) == 1:
                fig = plt.figure(111)
                axes = [fig.add_subplot(111)]
            elif len(NIRCfs) < 6:
                fig = plt.figure(figsize=(5, int(len(NIRCfs) * 22/5)))
                gs = gridspec.GridSpec(1, len(NIRCfs))
                gs.update(wspace=0.0, hspace=0.0)
                axes = []
                for i in range(len(NIRCfs)):
                    axes.append(fig.add_subplot(gs[0, i]))
            else:
                fig = plt.figure(figsize=(5*2, int(len(NIRCfs)/2 * 22/5)))
                gs = gridspec.GridSpec(2, int(len(NIRCfs/2)))
                gs.update(wspace=0.0, hspace=0.0)
                axes = []
                for i in range(len(NIRCfs)):
                    if i < len(NIRCfs) / 2:
                        axes.append(fig.add_subplot(gs[0, i]))
                    else:
                        axes.append(fig.add_subplot(gs[1, i]))

            # Set up parameters for drawing the scale line
            dim = extents[key][0] * 2
            right_side = dim - (dim * 0.1)
            vert = dim - (dim * 0.15)
            lab_vert = vert + (dim * 0.1) * 5 / 8
            lab_horz = right_side - scale / 2

            # Draw images
            for ax, f in zip(axes, NIRCfs):

                # Plot image with zeroed background
                ax.imshow(np.zeros_like(galimgs[key][f]), extent=extents[key], cmap='Greys_r')
                im = ax.imshow(np.log10(galimgs[key][f]), extent=extents[key], cmap='Greys_r')

                # Draw scale line
                ax.plot([right_side - scale, right_side], [vert, vert], color='w', linewidth=0.5)

                # Label scale
                ax.text(lab_horz, lab_vert, str(int(scale*1e3)) + ' ckpc', horizontalalignment='center',
                        fontsize=2, color='w')

                # Draw text
                ax.text(0.1, 0.9, f, bbox=dict(boxstyle="round,pad=0.3", fc='w', ec="k", lw=1, alpha=0.8),
                        transform=ax.transAxes, horizontalalignment='left', fontsize=4)

                # Remove ticks
                ax.tick_params(axis='both', left=False, top=False, right=False, bottom=False, labelleft=False,
                                labeltop=False, labelright=False, labelbottom=False)

                # Add colorbars
                cax = inset_axes(ax, width="50%", height="3%", loc='lower left')
                cbar = fig.colorbar(im, cax=cax, orientation="horizontal")

                # Label colorbars
                cbar.ax.set_xlabel(r'$\log_{10}(\mathrm{counts})$', fontsize=2, color='w', labelpad=1.0)
                cbar.ax.xaxis.set_label_position('top')
                cbar.outline.set_edgecolor('w')
                cbar.outline.set_linewidth(0.05)
                cbar.ax.tick_params(axis='x', length=1, width=0.2, pad=0.01, labelsize=2, color='w', labelcolor='w')

            fig.savefig('plots/webbimages/Webb_reg' + str(reg) + '_snap' + snap +
                        '_gal' + str(id).split('.')[0] + 'p' + str(id).split('.')[1] + '_coords' + key + '_PSF'
                        + str(psf) + '_' + '_'.join(NIRCfs) + '.png', bbox_inches='tight', dpi=600)

            plt.close(fig)


# Define comoving softening length in Mpc
csoft = 0.001802390/0.677

# Define image width
width = 3.

# Define resolution
arc_res = 0.031
print(width / arc_res, 'pixels in', width, 'arcseconds')

npart_lim = 10**4
NIRCfs = ('F115W', 'F150W', 'F200W')

regions = []
for reg in range(0, 38):
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

for i in range(len(reg_snaps)):

    print(reg_snaps[i][0], reg_snaps[i][1])

    # Define region variables
    reg = reg_snaps[i][0]
    snap = reg_snaps[i][1]
    path = '/cosma/home/dp004/dc-rope1/FLARES/FLARES-1/G-EAGLE_' + reg + '/data/'

    files = os.listdir('UVimg_data/')
    print(files)

    if 'stellardata_reg' + reg + '_snap' + snap + '_npartgreaterthan' + str(npart_lim) + '.pck' in files:
        load = True
    else:
        continue
        # load = False

    # try:
    #     img_main(path, snap, reg, arc_res, model, F, output=True, psf=True, npart_lim=npart_lim, dim=width, load=load,
    #              conv=(u.solMass/u.Mpc**2).to(u.g/u.cm**2), scale=0.1, NIRCfs=NIRCfs)
    # except ValueError:
    #     print('ValueError')
    #     continue
    # except KeyError:
    #     print('KeyError')
    #     continue
    # except OSError:
    #     print('OSError')
    #     continue
    img_main(path, snap, reg, arc_res, model, F, output=True, psf=True, npart_lim=npart_lim, dim=width, load=load,
             conv=(u.solMass / u.Mpc ** 2).to(u.g / u.cm ** 2), scale=0.1, NIRCfs=NIRCfs)
