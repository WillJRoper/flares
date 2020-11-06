"""
    All the functions listed here requires the generation of the particle
    information file.
"""

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname("__file__"), '..')))
from functools import partial
import schwimmbad
from SynthObs.SED import models
import FLARE
import FLARE.filters
from FLARE.photom import lum_to_M
import h5py
import utilities as util


def DTM_fit(Z, Age):
    """
    Fit function from L-GALAXIES dust modeling
    Formula uses Age in Gyr while the supplied Age is in Myr
    """

    D0, D1, alpha, beta, gamma = 0.008, 0.329, 0.017, -1.337, 2.122
    tau = 5e-5 / (D0 * Z)
    DTM = D0 + (D1 - D0) * (1. - np.exp(-alpha * (Z ** beta)
                                        * ((Age / (1e3 * tau)) ** gamma)))
    if np.isnan(DTM) or np.isinf(DTM):
        DTM = 0.

    return DTM


def get_data(ii, tag, inp='FLARES'):

    num = str(ii)
    if inp == 'FLARES':
        if len(num) == 1:
            num = '0' + num

        sim = rF"./data/FLARES_{num}_sp_info.hdf5"

    else:
        sim = rF"./data/EAGLE_{inp}_sp_info.hdf5"

    with h5py.File(sim, 'r') as hf:
        S_len = np.array(hf[tag + '/Galaxy'].get('S_Length'),
                         dtype=np.int64)
        G_len = np.array(hf[tag + '/Galaxy'].get('G_Length'),
                         dtype=np.int64)
        cops = np.array(hf[tag + '/Galaxy'].get("COP"),
                        dtype=np.float64)
        S_mass = np.array(hf[tag + '/Particle'].get('S_MassInitial'),
                          dtype=np.float64)
        S_Z = np.array(hf[tag + '/Particle'].get('S_Z_smooth'),
                       dtype=np.float64)
        S_age = np.array(hf[tag + '/Particle'].get('S_Age'),
                         dtype=np.float64) * 1e3
        S_los = np.array(hf[tag + '/Particle'].get('S_los'),
                         dtype=np.float64)
        G_Z = np.array(hf[tag + '/Particle'].get('G_Z_smooth'),
                       dtype=np.float64)
        S_sml = np.array(hf[tag + '/Particle'].get('S_sml'),
                         dtype=np.float64)
        G_sml = np.array(hf[tag + '/Particle'].get('G_sml'),
                         dtype=np.float64)
        G_mass = np.array(hf[tag + '/Particle'].get('G_Mass'),
                          dtype=np.float64)
        S_coords = np.array(hf[tag + '/Particle'].get('S_Coordinates'),
                           dtype=np.float64)
        G_coords = np.array(hf[tag + '/Particle'].get('G_Coordinates'),
                           dtype=np.float64)
        S_vels = np.array(hf[tag + '/Particle'].get('S_Vel'),
                          dtype=np.float64)
        G_vels = np.array(hf[tag + '/Particle'].get('G_Vel'),
                          dtype=np.float64)

    begin = np.zeros(len(S_len), dtype=np.int64)
    end = np.zeros(len(S_len), dtype=np.int64)
    begin[1:] = np.cumsum(S_len)[:-1]
    end = np.cumsum(S_len)

    gbegin = np.zeros(len(G_len), dtype=np.int64)
    gend = np.zeros(len(G_len), dtype=np.int64)
    gbegin[1:] = np.cumsum(G_len)[:-1]
    gend = np.cumsum(G_len)

    return S_mass, S_Z, S_age, S_los, G_Z, S_len, \
           G_len, G_sml, S_sml, G_mass, S_coords, G_coords, \
           S_vels, G_vels, cops, \
           begin, end, gbegin, gend


def lum(sim, kappa, tag, BC_fac, inp='FLARES', IMF='Chabrier_300', LF=True,
        filters=('FAKE.TH.FUV',), Type='Total', log10t_BC=7.,
        extinction='default', orientation="sim"):
    
    kinp = np.load('/cosma/home/dp004/dc-rope1/cosma7/FLARES/'
                   'flares/los_extinction/kernel_sph-anarchy.npz',
                   allow_pickle=True)
    lkernel = kinp['kernel']
    header = kinp['header']
    kbins = header.item()['bins']

    S_mass, S_Z, S_age, S_los, G_Z, S_len, \
    G_len, G_sml, S_sml, G_mass, S_coords, G_coords, \
    S_vels, G_vels, cops, \
    begin, end, gbegin, gend = get_data(sim, tag, inp)

    Lums = {f: np.zeros(len(S_mass), dtype=np.float64) for f in filters}

    model = models.define_model(
        F'BPASSv2.2.1.binary/{IMF}')  # DEFINE SED GRID -
    if extinction == 'default':
        model.dust_ISM = (
            'simple', {'slope': -1.})  # Define dust curve for ISM
        model.dust_BC = ('simple', {
            'slope': -1.})  # Define dust curve for birth cloud component
    elif extinction == 'Calzetti':
        model.dust_ISM = ('Starburst_Calzetti2000', {''})
        model.dust_BC = ('Starburst_Calzetti2000', {''})
    elif extinction == 'SMC':
        model.dust_ISM = ('SMC_Pei92', {''})
        model.dust_BC = ('SMC_Pei92', {''})
    elif extinction == 'MW':
        model.dust_ISM = ('MW_Pei92', {''})
        model.dust_BC = ('MW_Pei92', {''})
    elif extinction == 'N18':
        model.dust_ISM = ('MW_N18', {''})
        model.dust_BC = ('MW_N18', {''})
    else:
        ValueError("Extinction type not recognised")

    z = float(tag[5:].replace('p', '.'))

    # --- create rest-frame luminosities
    F = FLARE.filters.add_filters(filters, new_lam=model.lam)
    model.create_Lnu_grid(
        F)  # --- create new L grid for each filter. In units of erg/s/Hz

    for jj in range(len(begin)):

        # Extract values for this galaxy
        Masses = S_mass[begin[jj]: end[jj]]
        Ages = S_age[begin[jj]: end[jj]]
        Metallicities = S_Z[begin[jj]: end[jj]]
        gasMetallicities = G_Z[begin[jj]: end[jj]]
        gasSML = G_sml[begin[jj]: end[jj]]
        gasMasses = G_mass[begin[jj]: end[jj]]

        if orientation == "sim":

            starCoords = S_coords[:, begin[jj]: end[jj]].T - cops[:, jj]
            S_coords[:, begin[jj]: end[jj]] = starCoords.T

            MetSurfaceDensities = S_los[begin[jj]:end[jj]]

        elif orientation == "face-on":

            starCoords = S_coords[:, begin[jj]: end[jj]].T - cops[:, jj]
            gasCoords = G_coords[:, begin[jj]: end[jj]].T - cops[:, jj]
            gasVels = G_vels[:, begin[jj]: end[jj]].T

            # Get angular momentum vector
            ang_vec = util.ang_mom_vector(gasMasses, gasCoords, gasVels)

            # Rotate positions
            starCoords = util.get_rotated_coords(ang_vec, starCoords)
            gasCoords = util.get_rotated_coords(ang_vec, gasCoords)
            S_coords[:, begin[jj]: end[jj]] = starCoords.T

            MetSurfaceDensities = util.get_Z_LOS(starCoords, gasCoords,
                                                 gasMasses, gasMetallicities,
                                                 gasSML, (0, 1, 2),
                                                 lkernel, kbins)
        elif orientation == "side-on":

            starCoords = S_coords[:, begin[jj]: end[jj]].T - cops[:, jj]
            gasCoords = G_coords[:, begin[jj]: end[jj]].T - cops[:, jj]
            gasVels = G_vels[:, begin[jj]: end[jj]].T

            # Get angular momentum vector
            ang_vec = util.ang_mom_vector(gasMasses, gasCoords, gasVels)

            # Rotate positions
            starCoords = util.get_rotated_coords(ang_vec, starCoords)
            gasCoords = util.get_rotated_coords(ang_vec, gasCoords)
            S_coords[:, begin[jj]: end[jj]] = starCoords.T

            MetSurfaceDensities = util.get_Z_LOS(starCoords, gasCoords,
                                                 gasMasses, gasMetallicities,
                                                 gasSML, (2, 0, 1),
                                                 lkernel, kbins)
        else:
            MetSurfaceDensities = None
            print(orientation,
                  "is not an recognised orientation. "
                  "Accepted types are 'sim', 'face-on', or 'side-on'")

        # GMetallicities=G_Z[gbegin[jj]:gend[jj]]
        # Mage=np.nansum(Masses*Ages)/np.nansum(Masses)
        # Z=np.nanmean(GMetallicities)
        #
        # MetSurfaceDensities=DTM_fit(Z, Mage) * MetSurfaceDensities

        if Type == 'Total':
            # --- calculate V-band (550nm) optical depth for each star particle
            tauVs_ISM = kappa * MetSurfaceDensities
            tauVs_BC = BC_fac * (Metallicities / 0.01)
            fesc = 0.0

        elif Type == 'Pure-stellar':
            tauVs_ISM = np.zeros(len(Masses))
            tauVs_BC = np.zeros(len(Masses))
            fesc = 1.0

        elif Type == 'Intrinsic':
            tauVs_ISM = np.zeros(len(Masses))
            tauVs_BC = np.zeros(len(Masses))
            fesc = 0.0

        elif Type == 'Only-BC':
            tauVs_ISM = np.zeros(len(Masses))
            tauVs_BC = BC_fac * (Metallicities / 0.01)
            fesc = 0.0

        else:
            tauVs_ISM = None
            tauVs_BC = None
            fesc = None
            ValueError(F"Undefined Type {Type}")

        # --- calculate rest-frame Luminosity. In units of erg/s/Hz
        for f in filters:
            Lnu = models.generate_Lnu_array(model, Masses, Ages, Metallicities,
                                            tauVs_ISM, tauVs_BC, F, f, 
                                            fesc=fesc, log10t_BC=log10t_BC)
            Lums[f][begin[jj]: end[jj]] = Lnu
            
    Lums["coords"] = S_coords
    Lums["smls"] = S_sml
    Lums["begin"] = begin
    Lums["end"] = end

    return Lums  # , S_len + G_len


def flux(sim, kappa, tag, BC_fac, inp='FLARES', IMF='Chabrier_300',
         filters=FLARE.filters.NIRCam_W, Type='Total', log10t_BC=7.,
         extinction='default', orientation="sim"):
    
    kinp = np.load('/cosma/home/dp004/dc-rope1/cosma7/FLARES/'
                   'flares/los_extinction/kernel_sph-anarchy.npz',
                   allow_pickle=True)
    lkernel = kinp['kernel']
    header = kinp['header']
    kbins = header.item()['bins']

    S_mass, S_Z, S_age, S_los, G_Z, S_len, \
    G_len, G_sml, S_sml, G_mass, S_coords, G_coords, \
    S_vels, G_vels, cops, \
    begin, end, gbegin, gend = get_data(sim, tag, inp)

    Fnus = {f: np.zeros(len(S_mass), dtype=np.float64) for f in filters}

    model = models.define_model(
        F'BPASSv2.2.1.binary/{IMF}')  # DEFINE SED GRID -
    if extinction == 'default':
        model.dust_ISM = (
            'simple', {'slope': -1.})  # Define dust curve for ISM
        model.dust_BC = ('simple', {
            'slope': -1.})  # Define dust curve for birth cloud component
    elif extinction == 'Calzetti':
        model.dust_ISM = ('Starburst_Calzetti2000', {''})
        model.dust_BC = ('Starburst_Calzetti2000', {''})
    elif extinction == 'SMC':
        model.dust_ISM = ('SMC_Pei92', {''})
        model.dust_BC = ('SMC_Pei92', {''})
    elif extinction == 'MW':
        model.dust_ISM = ('MW_Pei92', {''})
        model.dust_BC = ('MW_Pei92', {''})
    elif extinction == 'N18':
        model.dust_ISM = ('MW_N18', {''})
        model.dust_BC = ('MW_N18', {''})
    else:
        ValueError("Extinction type not recognised")

    z = float(tag[5:].replace('p', '.'))
    F = FLARE.filters.add_filters(filters, new_lam=model.lam * (1. + z))

    cosmo = FLARE.default_cosmo()

    # --- create new Fnu grid for each filter. In units of nJy/M_sol
    model.create_Fnu_grid(F, z, cosmo)

    for jj in range(len(begin)):

        # Extract values for this galaxy
        Masses = S_mass[begin[jj]: end[jj]]
        Ages = S_age[begin[jj]: end[jj]]
        Metallicities = S_Z[begin[jj]: end[jj]]
        gasMetallicities = G_Z[begin[jj]: end[jj]]
        gasSML = G_sml[begin[jj]: end[jj]]
        gasMasses = G_mass[begin[jj]: end[jj]]

        if orientation == "sim":

            starCoords = S_coords[:, begin[jj]: end[jj]].T - cops[:, jj]
            S_coords[:, begin[jj]: end[jj]] = starCoords.T

            MetSurfaceDensities = S_los[begin[jj]:end[jj]]

        elif orientation == "face-on":

            starCoords = S_coords[:, begin[jj]: end[jj]].T - cops[:, jj]
            gasCoords = G_coords[:, begin[jj]: end[jj]].T - cops[:, jj]
            gasVels = G_vels[:, begin[jj]: end[jj]].T

            # Get angular momentum vector
            ang_vec = util.ang_mom_vector(gasMasses, gasCoords, gasVels)

            # Rotate positions
            starCoords = util.get_rotated_coords(ang_vec, starCoords)
            gasCoords = util.get_rotated_coords(ang_vec, gasCoords)
            S_coords[:, begin[jj]: end[jj]] = starCoords.T

            MetSurfaceDensities = util.get_Z_LOS(starCoords, gasCoords,
                                                 gasMasses, gasMetallicities,
                                                 gasSML, (0, 1, 2),
                                                 lkernel, kbins)
        elif orientation == "side-on":

            starCoords = S_coords[:, begin[jj]: end[jj]].T - cops[:, jj]
            gasCoords = G_coords[:, begin[jj]: end[jj]].T - cops[:, jj]
            gasVels = G_vels[:, begin[jj]: end[jj]].T

            # Get angular momentum vector
            ang_vec = util.ang_mom_vector(gasMasses, gasCoords, gasVels)

            # Rotate positions
            starCoords = util.get_rotated_coords(ang_vec, starCoords)
            gasCoords = util.get_rotated_coords(ang_vec, gasCoords)
            S_coords[:, begin[jj]: end[jj]] = starCoords.T

            MetSurfaceDensities = util.get_Z_LOS(starCoords, gasCoords,
                                                 gasMasses, gasMetallicities,
                                                 gasSML, (2, 0, 1),
                                                 lkernel, kbins)
        else:
            MetSurfaceDensities = None
            print(orientation,
                  "is not an recognised orientation. "
                  "Accepted types are 'sim', 'face-on', or 'side-on'")

        # GMetallicities=G_Z[gbegin[jj]:gend[jj]]
        #
        # Mage=Masses*Ages/np.nansum(Masses)
        # Z=np.nanmean(GMetallicities)
        # if kappa == 0:
        #     tauVs=kappa * MetSurfaceDensities
        # else:
        #     tauVs=DTM_fit(Z, Mage) * MetSurfaceDensities

        if Type == 'Total':
            # --- calculate V-band (550nm) optical depth for each star particle
            tauVs_ISM = kappa * MetSurfaceDensities
            tauVs_BC = BC_fac * (Metallicities / 0.01)
            fesc = 0.0

        elif Type == 'Pure-stellar':
            tauVs_ISM = np.zeros(len(Masses))
            tauVs_BC = np.zeros(len(Masses))
            fesc = 1.0

        elif Type == 'Intrinsic':
            tauVs_ISM = np.zeros(len(Masses))
            tauVs_BC = np.zeros(len(Masses))
            fesc = 0.0

        elif Type == 'Only-BC':
            tauVs_ISM = np.zeros(len(Masses))
            tauVs_BC = BC_fac * (Metallicities / 0.01)
            fesc = 0.0

        else:
            tauVs_ISM = None
            tauVs_BC = None
            fesc = None
            ValueError(F"Undefined Type {Type}")
            
        # --- calculate rest-frame Luminosity. In units of erg/s/Hz
        for f in filters:

            # --- calculate rest-frame flux of each object in nJy
            Fnu = models.generate_Fnu_array(model, Masses, Ages, Metallicities,
                                            tauVs_ISM, tauVs_BC, F, f, 
                                            fesc=fesc, log10t_BC=log10t_BC)

            Fnus[f][begin[jj]: end[jj]] = Fnu

    Fnus["coords"] = S_coords
    Fnus["smls"] = S_sml
    Fnus["begin"] = begin
    Fnus["end"] = end

    return Fnus


def get_lines(sim, kappa, tag, BC_fac, inp='FLARES', IMF='Chabrier_300',
              LF=False, lines='HI6563', Type='Total', log10t_BC=7.,
              extinction='default', orientation="sim"):

    kinp = np.load('/cosma/home/dp004/dc-rope1/cosma7/FLARES/'
                   'flares/los_extinction/kernel_sph-anarchy.npz',
                   allow_pickle=True)
    lkernel = kinp['kernel']
    header = kinp['header']
    kbins = header.item()['bins']

    S_mass, S_Z, S_age, S_los, G_Z, S_len, \
    G_len, G_sml, S_sml, G_mass, S_coords, G_coords, \
    S_vels, G_vels, cops, \
    begin, end, gbegin, gend = get_data(sim, tag, inp)

    # --- calculate intrinsic quantities
    if extinction == 'default':
        dust_ISM = ('simple', {'slope': -1.})  # Define dust curve for ISM
        dust_BC = ('simple', {
            'slope': -1.})  # Define dust curve for birth cloud component
    elif extinction == 'Calzetti':
        dust_ISM = ('Starburst_Calzetti2000', {''})
        dust_BC = ('Starburst_Calzetti2000', {''})
    elif extinction == 'SMC':
        dust_ISM = ('SMC_Pei92', {''})
        dust_BC = ('SMC_Pei92', {''})
    elif extinction == 'MW':
        dust_ISM = ('MW_Pei92', {''})
        dust_BC = ('MW_Pei92', {''})
    elif extinction == 'N18':
        dust_ISM = ('MW_N18', {''})
        dust_BC = ('MW_N18', {''})
    else:
        ValueError("Extinction type not recognised")

    lum = np.zeros(len(begin), dtype=np.float64)
    EW = np.zeros(len(begin), dtype=np.float64)

    # --- initialise model with SPS model and IMF.
    # Set verbose=True to see a list of available lines.
    m = models.EmissionLines(F'BPASSv2.2.1.binary/{IMF}', dust_BC=dust_BC,
                             dust_ISM=dust_ISM, verbose=False)
    for jj in range(len(begin)):

        # Extract values for this galaxy
        Masses = S_mass[begin[jj]: end[jj]]
        Ages = S_age[begin[jj]: end[jj]]
        Metallicities = S_Z[begin[jj]: end[jj]]
        gasMetallicities = G_Z[begin[jj]: end[jj]]
        gasSML = G_sml[begin[jj]: end[jj]]
        gasMasses = G_mass[begin[jj]: end[jj]]

        if orientation == "sim":

            MetSurfaceDensities = S_los[begin[jj]:end[jj]]

        elif orientation == "face-on":

            starCoords = S_coords[:, begin[jj]: end[jj]].T - cops[:, jj]
            gasCoords = G_coords[:, begin[jj]: end[jj]].T - cops[:, jj]
            gasVels = G_vels[:, begin[jj]: end[jj]].T

            # Get angular momentum vector
            ang_vec = util.ang_mom_vector(gasMasses, gasCoords, gasVels)

            # Rotate positions
            starCoords = util.get_rotated_coords(ang_vec, starCoords)
            gasCoords = util.get_rotated_coords(ang_vec, gasCoords)

            MetSurfaceDensities = util.get_Z_LOS(starCoords, gasCoords,
                                                 gasMasses,
                                                 gasMetallicities,
                                                 gasSML, (0, 1, 2),
                                                 lkernel, kbins)
        elif orientation == "side-on":

            starCoords = S_coords[:, begin[jj]: end[jj]].T - cops[:, jj]
            gasCoords = G_coords[:, begin[jj]: end[jj]].T - cops[:, jj]
            gasVels = G_vels[:, begin[jj]: end[jj]].T

            # Get angular momentum vector
            ang_vec = util.ang_mom_vector(gasMasses, gasCoords, gasVels)

            # Rotate positions
            starCoords = util.get_rotated_coords(ang_vec, starCoords)
            gasCoords = util.get_rotated_coords(ang_vec, gasCoords)

            MetSurfaceDensities = util.get_Z_LOS(starCoords, gasCoords,
                                                 gasMasses,
                                                 gasMetallicities,
                                                 gasSML, (2, 0, 1),
                                                 lkernel, kbins)
        else:
            MetSurfaceDensities = None
            print(orientation,
                  "is not an recognised orientation. "
                  "Accepted types are 'sim', 'face-on', or 'side-on'")

        if Type == 'Total':
            # --- calculate V-band (550nm) optical depth for each star particle
            tauVs_ISM = kappa * MetSurfaceDensities
            tauVs_BC = BC_fac * (Metallicities / 0.01)
            fesc = 0.0

        elif Type == 'Pure-stellar':
            tauVs_ISM = np.zeros(len(Masses))
            tauVs_BC = np.zeros(len(Masses))
            fesc = 1.0

        elif Type == 'Intrinsic':
            tauVs_ISM = np.zeros(len(Masses))
            tauVs_BC = np.zeros(len(Masses))
            fesc = 0.0

        elif Type == 'Only-BC':
            tauVs_ISM = np.zeros(len(Masses))
            tauVs_BC = BC_fac * (Metallicities / 0.01)
            fesc = 0.0

        else:
            tauVs_ISM = None
            tauVs_BC = None
            fesc = None
            ValueError(F"Undefined Type {Type}")

        o = m.get_line_luminosity(lines, Masses, Ages, Metallicities,
                                  tauVs_BC=tauVs_BC, tauVs_ISM=tauVs_ISM,
                                  verbose=False, log10t_BC=log10t_BC)

        lum[jj] = o['luminosity']
        EW[jj] = o['EW']

    return lum, EW


def get_lum(sim, kappa, tag, BC_fac, IMF='Chabrier_300',
            bins=np.arange(-24, -16, 0.5), inp='FLARES', LF=True,
            filters=('FAKE.TH.FUV'), Type='Total', log10t_BC=7.,
            extinction='default', orientation="sim"):
    try:
        Lums = lum(sim, kappa, tag, BC_fac=BC_fac, IMF=IMF, inp=inp, LF=LF,
                   filters=filters, Type=Type, log10t_BC=log10t_BC,
                   extinction=extinction, orientation=orientation)

    except Exception as e:
        Lums = np.ones(len(filters)) * np.nan
        print(e)

    if LF:
        tmp, edges = np.histogram(lum_to_M(Lums), bins=bins)
        return tmp

    else:
        return Lums


def get_lum_all(kappa, tag, BC_fac, IMF='Chabrier_300',
                bins=np.arange(-24, -16, 0.5), inp='FLARES', LF=True,
                filters=('FAKE.TH.FUV'), Type='Total', log10t_BC=7.,
                extinction='default', orientation="sim"):

    print(f"Getting luminosities for tag {tag} with kappa={kappa}")

    if inp == 'FLARES':
        df = pd.read_csv('weight_files/weights_grid.txt')
        weights = np.array(df['weights'])

        sims = np.arange(0, len(weights))

        calc = partial(get_lum, kappa=kappa, tag=tag, BC_fac=BC_fac, IMF=IMF,
                       bins=bins, inp=inp, LF=LF, filters=filters, Type=Type,
                       log10t_BC=log10t_BC, extinction=extinction,
                       orientation=orientation)

        pool = schwimmbad.MultiPool(processes=8)
        dat = np.array(list(pool.map(calc, sims)))
        pool.close()

        if LF:
            hist = np.sum(dat, axis=0)
            out = np.zeros(len(bins) - 1)
            err = np.zeros(len(bins) - 1)
            for ii, sim in enumerate(sims):
                err += np.square(np.sqrt(dat[ii]) * weights[ii])
                out += dat[ii] * weights[ii]

            return out, hist, np.sqrt(err)

        else:
            return dat

    else:
        out = get_lum(00, kappa=kappa, tag=tag, BC_fac=BC_fac, IMF=IMF,
                      bins=bins, inp=inp, LF=LF, filters=filters, Type=Type,
                      log10t_BC=log10t_BC, extinction=extinction,
                      orientation=orientation)

        return out


def get_flux(sim, kappa, tag, BC_fac, IMF='Chabrier_300', inp='FLARES',
             filters=FLARE.filters.NIRCam, Type='Total', log10t_BC=7.,
             extinction='default', orientation="sim"):

    try:
        Fnus = flux(sim, kappa, tag, BC_fac=BC_fac, IMF=IMF, inp=inp,
                    filters=filters, Type=Type, log10t_BC=log10t_BC,
                    extinction=extinction, orientation=orientation)

    except Exception as e:
        Fnus = np.ones(len(filters)) * np.nan
        print(e)

    return Fnus


def get_flux_all(kappa, tag, BC_fac, IMF='Chabrier_300', inp='FLARES',
                 filters=FLARE.filters.NIRCam, Type='Total', log10t_BC=7.,
                 extinction='default', orientation="sim"):

    print(f"Getting fluxes for tag {tag} with kappa={kappa}")

    if inp == 'FLARES':

        df = pd.read_csv('weight_files/weights_grid.txt')
        weights = np.array(df['weights'])

        sims = np.arange(0, len(weights))

        calc = partial(get_flux, kappa=kappa, tag=tag, BC_fac=BC_fac, IMF=IMF,
                       inp=inp, filters=filters, Type=Type,
                       log10t_BC=log10t_BC, extinction=extinction,
                       orientation=orientation)

        pool = schwimmbad.MultiPool(processes=8)
        out = np.array(list(pool.map(calc, sims)))
        pool.close()

    else:

        out = get_flux(00, kappa=kappa, tag=tag, BC_fac=BC_fac, IMF=IMF,
                       inp=inp, filters=filters, Type=Type,
                       log10t_BC=log10t_BC, extinction=extinction,
                       orientation=orientation)

    return out
