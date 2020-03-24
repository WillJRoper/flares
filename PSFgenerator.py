import os
os.environ['WEBBPSF_PATH'] = '/Users/willroper/anaconda3/envs/webbpsf-env/share/webbpsf-data/'
os.environ['PYSYN_CDBS'] = '/Users/willroper/anaconda3/envs/webbpsf-env/share/pysynphot-data/cdbs/'
import webbpsf


def genPSFs(NIRCf, Ndim, arc_res, redshift):
    """

    :param NIRCF:
    :param width:
    :param kpc_res:
    :param arc_res:
    :param redshift:
    :return:
    """

    # Compute the PSF
    nc = webbpsf.NIRCam()  # Assign JWST instrument to variable.
    nc.filter = NIRCf  # Set filter.
    psf = nc.calc_psf('JWSTPSFs/' + NIRCf + '_' + str(arc_res) + '_z=' + str(redshift) + '_' + str(Ndim)
                      + '_PSF.fits', oversample=1, fov_pixels=Ndim)  # Assign optical settings to variable.

    return psf


