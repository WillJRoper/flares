import numpy as np
from astropy.io.fits import open
import astropy.constants as const
import astropy.units as u
from noisegenerator import *
from PSFgenerator import genPSFs
from utilities import calc_ages, get_Z_LOS
from astropy.convolution import convolve_fft, Gaussian2DKernel
from photutils import CircularAperture, aperture_photometry
import matplotlib.pyplot as plt
from astropy.cosmology import Planck13 as cosmo
import numba as nb
import os
# os.environ['WEBBPSF_PATH'] = '/Users/willroper/anaconda3/envs/webbpsf-env/share/webbpsf-data/'
# os.environ['PYSYN_CDBS'] = '/Users/willroper/anaconda3/envs/webbpsf-env/share/pysynphot-data/cdbs/'
os.environ['FLARE'] = '/cosma7/data/dp004/dc-wilk2/flare'
# import webbpsf
import FLARE.filters
from SynthObs.SED import models


# Define SED model
model = models.define_model('BPASSv2.2.1.binary/ModSalpeter_300',
                            path_to_SPS_grid = FLARE.FLARE_dir + '/data/SPS/nebular/3.0/') # DEFINE SED GRID -
model.dust_ISM = ('simple', {'slope': -1.0})
model.dust_BC = ('simple', {'slope': -1.0})
filters = FLARE.filters.NIRCam
F = FLARE.filters.add_filters(filters)


# @nb.njit(nogil=True)
def createSimpleImgs(X, Y, masses, ages, metals, gal_met_surfden, smls, redshift, arc_res, ini_width,
					 NIRCf=None, model=model, F=F, output=False):
	''' A function that takes simulation data and produced a image with the galaxy in the centre applying
		smoothing based on the resolution of the simualtion.

	:param imgpath: The path to the halo data used to make this image. (str)
	:param galID: The ID of the galaxy in this halo. (int)
	:param redshift: Redshift. (float)
	:param arc_res: Pixel resolution in arcseconds. (float)
	:param ini_width: The approximate width of the image in arcseconds. (float, int)
	:param NIRCf: Filter ID. (str)
	:param model: An instance of the models class form SynthObs. (cls)
	:param F: An instance of FLARE.filters class from FLARE. (cls)
	:param output: A boolean controlling whether the print statements are executed. (Boolean)
	:return:
	'''

	# Compute the initial number of pixels along a single dimension of the image
	ini_Ndim = int(ini_width / arc_res)

	# Ndim must odd for convolution with the PSF
	if ini_Ndim % 2 != 0:
		Ndim = ini_Ndim
	else:
		Ndim = ini_Ndim + 1

	# Compute the width from the number of pixels along a single dimension (Ndim)
	width = Ndim * arc_res
	if output:
		print('Width= ', width, 'arcsecond')

	# Find the number of arcseconds per kpc at the current redshift using astropy and 'throw away' units
	arcsec_per_kpc_proper = cosmo.arcsec_per_kpc_proper(redshift).to(u.arcsec / u.Mpc).value

	# Calculate width in kpc to use for the extent of the image
	kpc_proper_per_arcmin = cosmo.kpc_proper_per_arcmin(redshift)
	mpc_width = ((width * u.arcsec).to(u.arcmin) * kpc_proper_per_arcmin).to(u.Mpc).value
	extent = [-mpc_width / 2, mpc_width / 2, -mpc_width / 2, mpc_width / 2]

	# Convert star positions to angular positons in arcseconds
	X *= arcsec_per_kpc_proper
	Y *= arcsec_per_kpc_proper

	# Extract the luminosity for the desired filter
	if NIRCf is not None:
		
		# Calculate optical depth of ISM and birth cloud
		tauVs_ISM = (10 ** 5.2) * gal_met_surfden
		tauVs_BC = 2.0 * (metals / 0.01)
		
		# Extract the flux in nanoJansky
		L = (models.generate_Fnu_array(model, masses, ages, metals, tauVs_ISM,
									   tauVs_BC, F, 'JWST.NIRCAM.' + NIRCf) * u.nJy).decompose()

		# Extarct the transmission curve and wavelengths
		T_curve = FLARE.filters.filter('JWST.NIRCAM.' + NIRCf).T
		lams = (FLARE.filters.filter('JWST.NIRCAM.' + NIRCf).lam * u.Angstrom).to(u.m)

		# Convert the flux of each star to electrons per second
		Fs = L.value * np.trapz(T_curve * const.h.value**-1 * lams.value**-1, x=lams.value) * 25
		L = Fs

	else:  # if there is no filter all stars have the same luminosity
		L = np.ones_like(X)

	# Find the rough centre based on the distribution of the star particles weighted by observed luminosity
	centre_x = np.sum(X * L) / np.sum(L)
	centre_y = np.sum(Y * L) / np.sum(L)

	# Compute offset from centre
	X -= centre_x
	Y -= centre_y

	# Print the number of particles for this galaxy
	org_nstars = len(L)
	# print('number of particles: ', org_nstars)

	# Compute the boolean array for inclusion within the image dimenisons
	s = (np.fabs(X) < width/2.) & (np.fabs(Y) < width/2.)

	# Mask star positions and luminosities which fall outside of the inclusion region defined by Ndim
	X = X[s]
	Y = Y[s]
	L = L[s]

	# Print the number of particles within the image area
	if output:
		print('number of particles in image area: ', len(L))

	# Print the fraction of the stars included in the image from those of the entire galaxy
	if output:
		print('fraction of stars included: ', len(L) / org_nstars)

	# =============== Compute the gaussian smoothed image ===============

	# Define x and y positions for the gaussians
	Gx, Gy = np.meshgrid(np.linspace(-width/2., width/2., Ndim), np.linspace(-width/2., width/2., Ndim))

	# Initialise the image array
	gsmooth_img = np.zeros((Ndim, Ndim))

	# Define the miniimum smoothing for 0.1kpc in arcseconds
	smooth = smls * cosmo.arcsec_per_kpc_proper(redshift).to(u.arcsec / u.Mpc).value

	# Define the image reduction size for sub images
	if arc_res == 0.031:
		sub_size = 8
	else:
		sub_size = 4

	# Get the image pixel coordinates along each axis
	ax_coords = np.linspace(-width/2., width/2., Ndim)

	# Loop over each star computing the smoothed gaussian distribution for this particle
	for x, y, l, sml in zip(X, Y, L, smooth):

		# Get this star's position within the image
		x_img, y_img = (np.abs(ax_coords - x)).argmin(), (np.abs(ax_coords - y)).argmin()

		# Define sub image over which to compute the smooothing for this star (1/4 of the images size)
		# NOTE: this drastically speeds up image creation
		sub_xlow, sub_xhigh = x_img-int(Ndim/sub_size), x_img+int(Ndim/sub_size) + 1
		sub_ylow, sub_yhigh = y_img-int(Ndim/sub_size), y_img+int(Ndim/sub_size) + 1

		# Compute the image
		g = np.exp(-(((Gx[sub_ylow:sub_yhigh, sub_xlow:sub_xhigh] - x) ** 2
					+ (Gy[sub_ylow:sub_yhigh, sub_xlow:sub_xhigh] - y) ** 2)
					/ (2.0 * sml ** 2)))

		# Get the sum of the gaussian
		gsum = np.sum(g)

		# If there are stars within the image in this gaussian add it to the image array
		if gsum > 0:
			gsmooth_img[sub_ylow:sub_yhigh, sub_xlow:sub_xhigh] += g * l / gsum

	if output:
		print(NIRCf, 'Image finished')

	return gsmooth_img, extent, L, Ndim


def createPSFdImgs(img, arc_res, filter, redshift, Ndim):
	''' A function for convolution of webbPSF point spread function with images.

	:param img: The image for convolution. (array)[Ndim, Ndim]
	:param arc_res: Pixel resolution in arcseconds. (float)
	:param filter: Filter ID. (str)
	:param redshift: Redshift. (float)
	:param Ndim: Number of pixels along a single dimension of the image. (int)
	:return:
	'''

	# Try to load the PSF
	try:
		psf = open('JWSTPSFs/' + filter + '_' + str(arc_res) + '_z=' + str(redshift) + '_' + str(Ndim) + '_PSF.fits')

	except FileNotFoundError:  # if it doesnt exist compute the PSF, save it and load it
		psf = genPSFs(filter, Ndim, arc_res, redshift)

	# Convolve the PSF with the image
	convolved_img = convolve_fft(img, psf[0].data)
	psf.close()

	return convolved_img


def createNoisyPSFdImgs(img, elec_noise=None, apprad=2., scale=1., seed=10000, expo_t=2866):
	''' A function for adding noise to Webb observations.

	:param img: The noiseless image. (array)[Ndim, Ndim]
	:param elec_noise: The standard deviation of the noise distribution in the same units as the image. (float)
	:param expo_t: The exposure time in seconds. (float)
	:return:
	'''

	# Get the SNR ratio
	if elec_noise is not None:
		noise_sigma = elec_noise
		SNR = 1

	else:
		SNR, noise_sigma, sig_ratio = SNRfinder(img, 'Circular', 1000, apprad, scale, seed)

	# Simulate exposure time
	noise = np.random.poisson(noise_sigma * expo_t, img.shape)

	# Add noise to the image
	nimg = img * expo_t + noise

	return nimg, SNR, noise_sigma, noise


def fakeGal(L0, hr, res):

	# Define x and y arrays
	x = np.linspace(-res/2., res/2., res)
	y = np.linspace(-res/2., res/2., res)

	# Get meshgrid from xs and ys
	xs, ys = np.meshgrid(x, y)

	# Compute the sersic profile galaxy
	img = L0 * np.exp(-np.sqrt(xs**2+ys**2)/hr)

	return img


def createHSTImg(imgpath, galID, redshift, kpc_res, arc_res, width, shortF=None, HSTf=None):

	# Make sure resolution is valid for HST or resampled HST
	assert (arc_res == 0.06 or arc_res == 0.03), 'Invalid resolution for HST WFC3, either 0.06 or 0.03 (resampled)'

	# Get simple image for valid resolutions
	img, nstars = createSimpleImgs(imgpath, galID, redshift, kpc_res, arc_res, width, shortF, longF=None)

	Ndim = img.shape[0]

	# ================ Apply the PSF to each image ================
	# Import the psf file from Tiny Tim downloads
	psf = open('HSTPSFs/' + HSTf + '_PSF.fits')[0].data

	# Convolve images with the PSF
	if Ndim % 2 == 0:
		if psf.shape[0] % 2 == 0:
			img = convolve_fft(img[1:, 1:], psf[1:, 1:])
		else:
			img = convolve_fft(img[1:, 1:], psf)
	else:
		if psf.shape[0] % 2 == 0:
			img = convolve_fft(img, psf[1:, 1:])
		else:
			img = convolve_fft(img, psf)

	print('Convolution complete')

	return img

