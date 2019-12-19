import os
import numpy as np
from astropy.cosmology import Planck13 as cosmo
from astropy import units as u
from scipy.optimize import curve_fit
from scipy.spatial import ConvexHull
import h5py
import schwimmbad
from functools import partial
import eagle_IO.eagle_IO as E
from numba import jit, njit

norm = np.linalg.norm
conv = (u.solMass/u.Mpc**2).to(u.solMass/u.pc**2)



class flares:

    def __init__(self,fname,sim_type='FLARES'):
        
        self.fname =fname
        self.sim_type = sim_type

        #Put down the sim root location here
        #self.directory = '/cosma7/data/dp004/dc-payy1/G-EAGLE/GEAGLE_'
        self.directory = '/cosma7/data/dp004/dc-payy1/G-EAGLE/'
        self.ref_directory = '/cosma5/data/Eagle/ScienceRuns/Planck1/L0100N1504/PE/REFERENCE/data'
        self.agn_directory = '/cosma5/data/Eagle/ScienceRuns/Planck1/L0050N0752/PE/S15_AGNdT9/data'

        self.cosmo = cosmo
        
        # self.data = "%s/data"%(os.path.dirname(os.path.realpath(__file__)))
        
        if self.sim_type == 'FLARES':
            self.radius = 14. #in cMpc/h
            self.halos = np.array([
                     '00','01','02','03','04',
                     '05','06','07','08','09',
                     '10','11','12','13','14',
                     '15','16','17','18','19',
                     '20','21','22','23','24',
                     '25','26','27','28','29',
                     '30','31','32','33','34',
                     '35','36','37','38','39'])

            self.tags = np.array([
                                  #'000_z015p000','001_z014p000','002_z013p000',
                                  #'003_z012p000','004_z011p000',
                                  '005_z010p000',
                                  '006_z009p000','007_z008p000','008_z007p000',
                                  '009_z006p000','010_z005p000'
                                  #,'011_z004p770'
                                  ])
            ## update with weights file location
            self.weights = '/cosma7/data/dp004/dc-love2/codes/ic_selection/weights_grid.txt'
        elif sim_type=="PERIODIC":
            self.tags = np.array(['001_z015p132','002_z009p993','003_z008p988',
                                  '004_z008p075','005_z007p050','006_z005p971',
                                  '007_z005p487','008_z005p037','009_z004p485'])
        else:
            raise ValueError("sim_type not recognised")



    # @jit
    def _sphere(self, coords, a, b, c, r):
    
        #Equation of a sphere
    
        x, y, z = coords[:,0], coords[:,1], coords[:,2]
    
        return (x-a)**2 + (y-b)**2 + (z-c)**2 - r**2


    def spherical_region(self, halo, snap):
        """
        Inspired from David Turner's suggestion
        """

        dm_cood = E.read_array('PARTDATA', halo, snap, '/PartType1/Coordinates',
        noH=False, physicalUnits=False, numThreads=4)  #dm particle coordinates

        hull = ConvexHull(dm_cood)

        cen = [np.median(dm_cood[:,0]), np.median(dm_cood[:,1]), np.median(dm_cood[:,2])]
        pedge = dm_cood[hull.vertices]  #edge particles
        y_obs = np.zeros(len(pedge))
        p0 = np.append(cen, self.radius)

        popt, pcov = curve_fit(self._sphere, pedge, y_obs, p0, method = 'lm', sigma = np.ones(len(pedge))*0.001)
        dist = np.sqrt(np.sum((pedge-popt[:3])**2, axis = 1))
        centre, radius, mindist = popt[:3], popt[3], np.min(dist)

        return centre, radius, mindist


    def calc_df(self, mstar, tag, volume, massBinLimits):

        hist, dummy = np.histogram(np.log10(mstar), bins = massBinLimits)
        hist = np.float64(hist)
        phi = (hist / volume) / (massBinLimits[1] - massBinLimits[0])
        phi_sigma = (np.sqrt(hist) / volume) /\
                    (massBinLimits[1] - massBinLimits[0]) # Poisson errors

        return phi, phi_sigma, hist


    def plot_df(self, ax, phi, phi_sigma, hist, massBins,
                label, color, hist_lim=10, lw=3, alpha=0.7, lines=True):

        kwargs = {}
        kwargs_lo = {}

        if lines:
            kwargs['lw']=lw

            kwargs_lo['lw']=lw
            kwargs_lo['linestyle']='dotted'
        else:
            kwargs['ls']=''
            kwargs['marker']='o'

            kwargs_lo['ls']=''
            kwargs_lo['marker']='o'
            kwargs_lo['markerfacecolor']='white'
            kwargs_lo['markeredgecolor']=color


        def yerr(phi,phi_sigma):

            p = phi
            ps = phi_sigma

            mask = (ps <= p)

            err_up = np.abs(np.log10(p) - np.log10(p + ps))
            err_lo = np.abs(np.log10(p) - np.log10(p - ps))

            return err_up, err_lo, mask


        err_up, err_lo, mask = yerr(phi,phi_sigma)

        err_lo[~mask] = 0.5
        err_up[~mask] = 0.5

        ax.errorbar(np.log10(massBins[phi > 0.]),
                np.log10(phi[phi > 0.]),
                yerr=[err_up[phi > 0.],
                      err_lo[phi > 0.]],
                uplims=(~mask[phi > 0.]),
                label=label, c=color, alpha=alpha, **kwargs)



    def load_property(self,name,arr_type='Subhalo'):

        if self.sim_type == "FLARES":
            out = {halo: {tag: None for tag in self.tags} for halo in self.halos}
            for halo in self.halos:
                for tag in self.tags:
                    out[halo][tag] = f['%s/%s/%s/%s'%(halo,tag,arr_type,name)][:]
        elif self.sim_type == "PERIODIC":
            out = {tag: None for tag in self.tags}
            for tag in self.tags:
                out[tag] = f['%s/%s/%s'%(tag,arr_type,name)][:]
        
        return out




    def create_group_hdf5(self, filename, obj_str):
        check = self.check_hdf5(filename, obj_str)
        with h5py.File(filename, 'a') as h5file:
            if check:
                print('Object already exists')
                return False
            else:
                h5file.create_group(obj_str)


    def check_hdf5(self, filename, obj_str):
        with h5py.File(filename, 'a') as h5file:
            if obj_str not in h5file:
                return False
            else:
                return True


    #def get_SFT(SFT, redshift):
    def get_star_formation_time(self, SFT, redshift):
    
        SFz = (1/SFT) - 1.
        SFz = self.cosmo.age(redshift).value - self.cosmo.age(SFz).value
        return SFz
    
    def get_age(self, arr, z, numThreads = 4):
    
        if numThreads == 1:
            pool = schwimmbad.SerialPool()
        elif numThreads == -1:
            pool = schwimmbad.MultiPool()
        else:
            pool = schwimmbad.MultiPool(processes=numThreads)
    
        calc = partial(get_SFT, redshift = z)
        Age = np.array(list(pool.map(calc,arr)))
    
        return Age



@jit()
def get_Z_LOS(s_cood, g_cood, g_mass, g_Z, g_sml, lkernel, kbins):

    """

    Compute the los metal surface density (in g/cm^2) for star particles inside the galaxy taking
    the z-axis as the los.
    Args:
        s_cood (3d array): stellar particle coordinates
        g_cood (3d array): gas particle coordinates
        g_mass (1d array): gas particle mass
        g_Z (1d array): gas particle metallicity
        g_sml (1d array): gas particle smoothing length

    """
    n = len(s_cood)
    Z_los_SD = np.zeros(n)
    #Fixing the observer direction as z-axis. Use make_faceon() for changing the
    #particle orientation to face-on
    xdir, ydir, zdir = 0, 1, 2
    for ii in range(n):

        thisspos = s_cood[ii]
        ok = (g_cood[:,zdir] > thisspos[zdir])
        thisgpos = g_cood[ok]
        thisgsml = g_sml[ok]
        thisgZ = g_Z[ok]
        thisgmass = g_mass[ok]
        x = thisgpos[:,xdir] - thisspos[xdir]
        y = thisgpos[:,ydir] - thisspos[ydir]

        b = np.sqrt(x*x + y*y)
        boverh = b/thisgsml

        ok = (boverh <= 1.)

        kernel_vals = np.array([lkernel[int(kbins*ll)] for ll in boverh[ok]])

        Z_los_SD[ii] = np.sum((thisgmass[ok]*thisgZ[ok]/(thisgsml[ok]*thisgsml[ok]))*kernel_vals) #in units of Msun/Mpc^2

    Z_los_SD*=conv #in units of Msun/pc^2

    return Z_los_SD
