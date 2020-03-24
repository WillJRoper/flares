import numpy as np
from photutils import CircularAperture, RectangularAperture, EllipticalAperture, aperture_photometry
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import astropy.units as u
import astropy.constants as const
from scipy.interpolate import interp1d
from scipy.optimize import curve_fit
from jwst_backgrounds import jbt
from astropy.cosmology import WMAP9 as cosmo
import os
import seaborn as sns


# Change plot astetics using seaborn
sns.set_context('notebook', font_scale=1.0)
sns.set_style('whitegrid')


def SNRfinder(img, aperture, appnum, apprad, scale, seed=10000):

    # Set random seed
    if seed is int:
        np.random.seed(seed)

    # Define the dimensions of the object image for which a SNR is to be computed
    ndim = img.shape[0]

    # Get the signal from the img
    if aperture == 'Elliptical1.5':
        sig_app = EllipticalAperture([np.floor(ndim / 2.), np.floor(ndim / 2.)], a=apprad * 1.5, b=apprad)
    elif aperture == 'Elliptical2.0':
        sig_app = EllipticalAperture([np.floor(ndim / 2.), np.floor(ndim / 2.)], a=apprad * 2, b=apprad)
    elif aperture == 'Rectangle':
        sig_app = RectangularAperture([np.floor(ndim / 2.), np.floor(ndim / 2.)], w=apprad, h=apprad)
    else:
        sig_app = CircularAperture([np.floor(ndim / 2.), np.floor(ndim / 2.)], r=apprad)

    # Create the photometry table
    sig_phot_table = aperture_photometry(img, sig_app)

    # Define the signal flux from the photometry table
    true_signal = sig_phot_table['aperture_sum'].data / sig_app.area()

    # Create large array of random noise
    noise = np.random.normal(loc=0.0, scale=scale, size=[6 * ndim, 6 * ndim])

    # Create appnum random aperture positions
    apposs = np.zeros([appnum, 2])
    iposs = np.random.randint(apprad, high=noise.shape[0]-apprad, size=appnum)
    jposs = np.random.randint(apprad, high=noise.shape[0]-apprad, size=appnum)
    for ind, i, j in zip(range(0, appnum), iposs, jposs):
        apposs[ind, 0] = i
        apposs[ind, 1] = j

    # Create the apertures
    if aperture == 'Circular':
        apertures = CircularAperture(apposs, r=apprad)
    elif aperture == 'Elliptical1.5':
        apertures = EllipticalAperture(apposs, a=apprad*1.5, b=apprad)
    elif aperture == 'Elliptical2.0':
        apertures = EllipticalAperture(apposs, a=apprad*2, b=apprad)
    elif aperture == 'Rectangle':
        apertures = RectangularAperture(apposs, w=apprad, h=apprad)

    # Create the photometry table
    phot_table = aperture_photometry(noise, apertures)

    # Store the aperture sums in the dictionary
    appflux = phot_table['aperture_sum'].data / apertures.area()

    # Extract the noise
    derived_noise = np.std(appflux)

    # Add noise to the image to get the noisy signal
    noisyimg = np.random.normal(loc=0.0, scale=derived_noise, size=[ndim, ndim])
    noisyimg += img

    if aperture == 'Elliptical1.5':
        noisy_sig_ap = EllipticalAperture([noisyimg.shape[0]/2., noisyimg.shape[1]/2.], a=apprad*1.5, b=apprad)
    elif aperture == 'Elliptical2.0':
        noisy_sig_ap = EllipticalAperture([noisyimg.shape[0]/2., noisyimg.shape[1]/2.], a=apprad*2, b=apprad)
    elif aperture == 'Rectangle':
        noisy_sig_ap = RectangularAperture([noisyimg.shape[0]/2., noisyimg.shape[1]/2.], w=apprad, h=apprad)
    else:
        noisy_sig_ap = CircularAperture([np.floor(ndim / 2.), np.floor(ndim / 2.)], r=apprad)

    # Create the photometry table
    noisysig_phot_table = aperture_photometry(noisyimg, noisy_sig_ap)

    # Define the signal flux from the photometry table
    noisy_sig = noisysig_phot_table['aperture_sum'].data / noisy_sig_ap.area()

    # Compute the SNR
    SNR = np.abs(true_signal / derived_noise)

    # Find the ratio of true signal to noisy signal
    sig_ratio = true_signal / noisy_sig

    return SNR, derived_noise, sig_ratio


def SNRheatmap(img, aperture, appnum, minrad, maxrad, minscale, maxscale, res, haloID, galID, npart, kpc_res, NIRCf=None):

    # Create SNR array
    SNRs = np.zeros([res, res])

    # Create arrays for computing the SNR map
    rs = np.linspace(minrad, maxrad, res)
    scales = np.linspace(minscale, maxscale, res)

    # Loop over radii
    count = 0
    for i, r in enumerate(rs):

        # Loop over over scales
        for j, scale in enumerate(scales):

            count += 1
            print('%.2f'%(count/len(rs)**2*100) + '%')
            SNRs[i, j], noise, ratio = SNRfinder(img, aperture, appnum, r, scale)

    # Use log scaling if the max SNR is large
    if SNRs.max() > 300:

        # Set up the figure
        fig = plt.figure()
        ax = fig.add_subplot(111)

        # Plot the contour map
        locator = ticker.LogLocator(base=10)
        cax = ax.contourf(rs, scales, SNRs, levels=res*10, locator=locator, cmap='viridis')
        cs = ax.contour(rs, scales, SNRs, levels=[1, 10], locator=locator, cmap='summer')

        ax.clabel(cs, inline=1, fontsize=10)
        cbar = fig.colorbar(cax)

        cbar.ax.set_ylabel(r'$\log$(SNR)', rotation=270, labelpad=15)

        # Label axes
        ax.set_xlabel(r'$R_{ap}/kpc$')
        ax.set_ylabel(r'$\sigma/\mathrm{erg} \ \mathrm{s}^{-1}  \ \mathrm{Hz}^{-1}$')

        if filter is not None:
            plt.savefig('SNR_maps/With PSF/logSNRmap_' + str(haloID) + '.' + str(galID) + '.' + str(npart)
                        + '.' + str(NIRCf) + '.png', dpi=600, bbox_inches='tight')
        else:
            plt.savefig('SNR_maps/Without PSF/logSNRmap_' + str(haloID) + '.' + str(galID) + '.' + str(npart)
                        + '.png', dpi=600, bbox_inches='tight')

    # Set up the figure
    fig = plt.figure()
    ax = fig.add_subplot(111)

    # Plot the contour map
    cax = ax.contourf(rs, scales, SNRs, levels=res*10, cmap='viridis')
    cs = ax.contour(rs, scales, SNRs, levels=[1, 5, 10, 20], cmap='summer')

    ax.clabel(cs, inline=1, fontsize=10)
    cbar = fig.colorbar(cax)

    cbar.ax.set_ylabel(r'SNR', rotation=270, labelpad=15)

    # Label axes
    ax.set_xlabel(r'$R_{ap}/pix$')
    ax.set_ylabel(r'$\sigma/\mathrm{erg} \ \mathrm{s}^{-1}  \ \mathrm{Hz}^{-1}$')

    if filter is not None:
        plt.savefig('SNR_maps/With PSF/SNRmap_' + str(haloID) + '.' + str(galID) + '.' + str(npart)
                    + '.' + str(NIRCf) + '.png', dpi=600, bbox_inches='tight')
    else:
        plt.savefig('SNR_maps/Without PSF/SNRmap_' + str(haloID) + '.' + str(galID) + '.' + str(npart)
                    + '.png', dpi=600, bbox_inches='tight')

    return


def sersic(x, y, L0, hr):
    return L0 * np.exp(-np.sqrt(x**2+y**2)/hr)


def fakeGal(L0, hr, res):

    # Define x and y arrays
    x = np.linspace(-res/2., res/2., res)
    y = np.linspace(-res/2., res/2., res)

    # Get meshgrid from xs and ys
    xs, ys = np.meshgrid(x, y)

    # Compute the sersic profile galaxy
    img = sersic(xs, ys, L0, hr)

    return img


def true_noise(ra, dec, day, f, filters, arc_res, redshift, electrons=True):
    ''' A fucntion used to interface with jwst_backgrounds to extarct background
        levels at L2 in electrons per second.

    :param ra: The right ascension in degrees. (float)
    :param dec: The declination in degrees. (float)
    :param day: The day of observation. (int)
    :param f: Filter ID. (str)
    :param filters: An instance of the FLARE.filters class. (cls)
    :param arc_res: The pixel resolution in arcseconds. (float)
    :param electrons:
    :return:
    '''

    # Extract the background curve from jwst_backgrounds
    bg = jbt.background(ra, dec, float(f.split('.')[-1][1:4]) / 100)  # calculate background
    day_ind = np.argwhere(bg.bkg_data['calendar'] == day)[0][0]  # identify index of day

    # Extract the background array and wavelengths at which it is sampled
    bkg_curve_jwstb = bg.bkg_data['total_bg'][day_ind, :]
    wavelengths = bg.bkg_data['wave_array']

    # Define mirror area in cm**2
    m_area = 25 * u.m**2

    # Extract wavelengths from filters object and convert to m
    bkg_lam = filters.filter(f).lam * u.Angstrom

    # Interpolate background curve to get the same resolution as transmission curve
    interpfunc = interp1d((wavelengths * u.micron).to(u.m), bkg_curve_jwstb, kind='cubic')
    interp_lam = bkg_lam.to(u.m)
    bkg_curve = interpfunc(interp_lam)

    # interp_lam = bkg_lam
    bkg_curve = bkg_curve * u.MJy * u.steradian ** -1

    # Multiply the angular area of a pixel to remove the inverse steradians
    bkg_curve = bkg_curve * (arc_res**2 * u.arcsecond**2).to(u.steradian)

    # Convert flux to SI
    bkg_curve = bkg_curve.to(u.J * u.s**-1 * u.m**-2 * u.Hz**-1)

    # Extract this filters transmission curve (wavelength must be in Angstrom)
    T_lambda = filters.filter(f).T

    if electrons:

        # Prepare the integrand in units of electrons /  (m3 * s)
        integrand = (bkg_curve * const.h**-1 * interp_lam**-1 * T_lambda).decompose()

        # Integrate over wavelength
        bkg = np.trapz(integrand, x=interp_lam)

        # Multiply by the area of a pixel in m2 to get electrons / s
        bkg *= m_area

    else:

        # Prepare the integrand in nJy
        integrand = (bkg_curve * const.c * interp_lam**-2 * T_lambda)

        # Integrate over wavelength
        bkg = np.trapz(integrand, x=interp_lam).to(u.nJy/u.s)

    # Get standard deviation of the noise distribution
    mean = bkg.value
    stdev = mean

    return stdev, T_lambda, interp_lam, wavelengths, bkg_curve_jwstb


# -------------------------------------------------

# fig = plt.figure()
# ax = fig.add_subplot(111)
#
# for size in [0.1, 0.5, 1, 1.5, 2]:
#
#     img = fakeGal(10, size, 39)
#
#     # Define the dimensions of the object image for which a SNR is to be computed
#     ndim = img.shape[0]
#
#     SNR10, noise10, ratio = SNRfinder(img, 'Circular', 10000, 5, 10)
#     rs = np.linspace(0.1, 17, 100)
#     SNRs = np.zeros(rs.size)
#     noises = np.zeros(rs.size)
#     for ind2, rad in enumerate(rs):
#         print(ind2)
#
#         SNRs[ind2], noises[ind2], ratio = SNRfinder(img, 'Circular', 1000, rad, noise10)
#
#     ax.plot(rs, SNRs, label=r'$h_{R}=$' + str(size) + ' pix')
#
# ax.set_xlabel(r'$R_{app}$')
# ax.set_ylabel(r'SNR')
#
# ax.grid(True)
#
# handles, labels = ax.get_legend_handles_labels()
# ax.legend(handles, labels)
#
# plt.savefig('FakeSNRNoise10AppSizeComp.png', dpi=300, bbox_inches='tight')

# -------------------------------------------------

# # Cycle through galaxies making images
# for haloID, halopath in zip(haloIDs, halopaths):
#     for i in galIDs[haloID]:
#         print('Halo: ', haloID, ' Galaxy: ', i)
#         gimg, npart = createPSFdImgs(halopath, i, redshift, kpc_res, 0.031, width, shortF='F150W')
#
#         # Define the dimensions of the object image for which a SNR is to be computed
#         ndim = gimg.shape[0]
#
#         SNR10, noise10, ratio = SNRfinder(gimg, 'Circular', 10000, 5, 10)
#         rs = np.linspace(0.1, ndim/2., 1000)
#         SNRs = np.zeros(rs.size)
#         noises = np.zeros(rs.size)
#         for ind2, rad in enumerate(rs):
#
#             SNRs[ind2], noises[ind2], ratio = SNRfinder(gimg, 'Circular', 1000, rad, noise10)
#
#         print(rs[SNRs.argmax()])
#         fig = plt.figure()
#         ax = fig.add_subplot(111)
#
#         ax.plot(rs, SNRs)
#
#         ax.set_xlabel(r'$R_{app}$')
#         ax.set_ylabel(r'SNR')
#
#         ax.grid(True)
#
#         plt.savefig('SNRNoise10AppSizeComp. ' + str(haloID) + '.' + str(i) + '.PSF.png', dpi=300, bbox_inches='tight')

# -------------------------------------------------

# bestSNRs = np.zeros(len(scales))
# bestRs = np.zeros(len(scales))
#
# for ind1, scale in enumerate(scales):
#
#         # Define the dimensions of the object image for which a SNR is to be computed
#         ndim = gimg.shape[0]
#         rs = np.linspace(0.5, ndim/2., 100)
#         SNRs = np.zeros(rs.size)
#         noises = np.zeros(rs.size)
#         for ind2, rad in enumerate(rs):
#
#             SNRs[ind2], noises[ind2], ratio = SNRfinder(gimg, 'Circular', 1000, rad, scale)
#
#         bestSNRs[ind1] = SNRs.max()
#         bestRs[ind1] = rs[SNRs.argmax()]
#
#         print(ind1)
#
# fig = plt.figure()
# ax = fig.add_subplot(111)
# axx = ax.twiny()
#
# locs = ax.get_xticks()
# ax.plot(scales, bestSNRs)
# axx.set_xticks(bestRs)
#
# ax.set_xlabel(r'$\sigma$')
# ax.set_ylabel(r'SNR')
#
# # handles, labels = ax.get_legend_handles_labels()
# # ax.legend(handles, labels)
#
# ax.grid(True)
#
# plt.savefig('SNRNoiseScaleAppRComp.png', dpi=300, bbox_inches='tight')
#
# print('-----------------------------------------')


# --------------------------------------------------------------

# scales = [1, 5, 10, 20, 30, 50, 100]
# for scale in scales:
#     # Cycle through galaxies making images
#     for haloID, halopath in zip(haloIDs, halopaths):
#         for i in galIDs[haloID]:
#             print('Halo: ', haloID, ' Galaxy: ', i)
#             rs = np.linspace(0.001, 50, 100)
#             gimg, Himg, resi = createSimpleImgs(halopath, snapshot, 7, redshift, kpc_res, width)
#             apps = {}
#
#             fig = plt.figure()
#             ax = fig.add_subplot(111)
#
#             for aperture in ['Circular', 'Elliptical1.5', 'Elliptical2.0']:
#                 SNRs = np.zeros(rs.size)
#                 noises = np.zeros(rs.size)
#                 for ind, r in enumerate(rs):
#                     SNRs[ind], noises[ind], ratio = SNRfinder(Himg, aperture, 1000, r, scale)
#                     print(SNRs[ind], Himg.max(), Himg.max()/SNRs[ind])
#                     print(aperture, ind)
#
#                 ax.plot(rs, SNRs, label=aperture)
#
#             ax.set_xlabel(r'$R_{ap}$')
#             ax.set_ylabel(r'SNR')
#
#             handles, labels = ax.get_legend_handles_labels()
#             ax.legend(handles, labels)
#
#             ax.grid(True)
#
#             plt.savefig('SNRApertureComp_' + str(scale) + '.png', dpi=300, bbox_inches='tight')
#
#             # Define the dimensions of the object image for which a SNR is to be computed
#             ndim = gimg.shape[0]
#
#             sig_app = CircularAperture([np.floor(ndim / 2.), np.floor(ndim / 2.)], r=10)
#
#             # Create the photometry table
#             sig_phot_table = aperture_photometry(gimg, sig_app)
#
#             # Define the signal flux from the photometry table
#             signal = sig_phot_table['aperture_sum'].data / sig_app.area()
#
#             noise = np.random.normal(loc=0.0, scale=noises.max(), size=Himg.shape)
#             noisyimg = gimg + noise
#
#             plt.figure()
#             plt.imshow(noisyimg)
#             plt.show()
#
#             print('-----------------------------------------')
#             break
#         break

# -------------------------------------------------------------

# # Cycle through galaxies making images
# for haloID, halopath in zip(haloIDs, halopaths):
#     for i in galIDs[haloID]:
#         print('Halo: ', haloID, ' Galaxy: ', i)
#         gimg, Himg, resi, npart = createSimpleImgs(halopath, snapshot, 7, redshift, kpc_res, width)
#         rs = np.linspace(0.001, gimg.shape[0]/2., 1000)
#         break
#     break
#
# fig = plt.figure()
# ax = fig.add_subplot(111)
#
# scales = np.linspace(1, 25, 5)
# for scale in scales:
#
#     SNRs = np.zeros(rs.size)
#     noises = np.zeros(rs.size)
#
#     for ind, r in enumerate(rs):
#         SNRs[ind], noises[ind], ratio = SNRfinder(gimg, 'Circular', 1000, r, scale)
#         print(ind)
#
#     print(rs[SNRs.argmax()])
#     ax.plot(rs, SNRs, label=r'$\sigma= $' + str(scale))
#
# ax.set_xlabel(r'$R_{ap}$')
# ax.set_ylabel(r'SNR')
#
# handles, labels = ax.get_legend_handles_labels()
# ax.legend(handles, labels)
#
# ax.grid(True)
#
# plt.savefig('SNRvRap_' + str(haloID) + '.' + str(i) + '.png', dpi=300, bbox_inches='tight')
# fig.clf()
#
# fig1 = plt.figure()
# ax1 = fig1.add_subplot(111)
#
# ndim = gimg.shape[0]
#
# sig_app = CircularAperture([np.floor(ndim / 2.), np.floor(ndim / 2.)], r=rs[SNRs.argmax()])
# sig_app.plot()
#
# # Create the photometry table
# sig_phot_table = aperture_photometry(gimg, sig_app)
#
# # Define the signal flux from the photometry table
# signal = sig_phot_table['aperture_sum'].data / sig_app.area()
#
# ax1.imshow(gimg, extent=[0, gimg.shape[0], 1, gimg.shape[0]])
# plt.show()


