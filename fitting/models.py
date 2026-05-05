import numpy as np
from scipy.integrate import simpson, quad, cumulative_trapezoid
from scipy.special import erf
import astropy.units as u
from colossus.halo import concentration
from colossus.lss import bias
from colossus.halo import profile_nfw
from colossus.halo import profile_outer

from fitting.constants import pc, Msun, cosmo, SQPI

# ====================
# Auxiliary functions
# ====================

def logistic(x, x0=1, k=10):
    return (1.0 + np.exp(-2.0 * k * (x - x0)))**(-1)

def gamma_pdf(Rs, sigma):
    return (Rs / sigma**2) * np.exp(-1.0 * (Rs / sigma))

def rayleigh_pdf(Rs, sigma):
    return (Rs / sigma**2) * np.exp(-0.5 * (Rs / sigma)**2)

# ====================
# Base models: sigma, delta_sigma with integration
# ====================

class BaseModelFast:

    def density_contrast(self):
        ''' density contrast delta(r) = rho(x)/rho_mean - 1 '''
        raise NotImplementedError('Must be defined in child class')

    def sigma(self, R, *params):

        p = params
        u_grid = np.linspace(0.0, 100.0, 500)
        radius_grid = np.hypot(u_grid[None, :], R[:, None])
        integrand_grid = self.density_contrast(radius_grid, *p)
        result = 2.0 * simpson(integrand_grid, u_grid, axis=1)

        return result

    def delta_sigma(self, R, *params):
        ''' integration from density contrast as eq B.16 of Cromer+2022 '''
        num_theta=200
        num_x=1000

        x_grid = np.linspace(1e-5, R.max(), num_x)
        #x_grid = np.geomspace(1e-5, R.max(), num_x)
        integrand_x = x_grid**2 * self.density_contrast(x_grid, *params)
        cumulative = cumulative_trapezoid(integrand_x, x_grid, initial=0.0)
        I1_interp = np.interp(R, x_grid, cumulative)

        theta = np.linspace(0.0, np.pi/2.0 - 1e-6, num_theta)
        denom = 4.0 * np.sin(theta) + 3.0 - np.cos(2.0 * theta)

        r_mesh = R[:, None] / np.cos(theta[None, :])

        integrand_theta = self.density_contrast(r_mesh, *params) / denom[None, :]
        I2 = simpson(integrand_theta, theta, axis=1)

        return (4.0 / R**2) * I1_interp - 4.0 * R * I2

class BaseModelQuad:

    def delta_sigma(self, R, *params):

        x_grid = np.linspace(0.0, R.max(), 1000)
        integrand = x_grid**2 * self.density_contrast(x_grid, *params)
        cumulative = cumulative_trapezoid(integrand, x_grid, initial=0.0)

        I1_interp = np.interp(R, x_grid, cumulative)

        result = np.zeros_like(R)

        for i, Ri in enumerate(R):
            def integrand2(theta):
                return self.density_contrast(Ri/np.cos(theta), *params) / (4.0*np.sin(theta) + 3 - np.cos(2.0*theta))

            I2,_ = quad(integrand2, 0.0, np.pi/2.0 - 1e-6)
            result[i] = (4.0/Ri**2)*I1_interp[i] - 4.0*Ri*I2

        return result

# =============================
#  DENSITY MODELS FOR CLUSTERS
# =============================

class NFW:
    def __init__(self, redshift, miss_pdf = rayleigh_pdf):
        self.redshift : float = redshift
        self.rho_c : float = cosmo.critical_density(redshift).to('M_sun/Mpc^3').value
        self.miss_pdf = miss_pdf

    def R_200(self, M200:float) -> float | list[float]:
        ''' R_200 in Mpc/h '''
        return (M200 * (3.0 / (4.0 * np.pi)) * 1.0 / (200.0 * self.rho_c))**(1/3)

    def c_200(self, M200:float) -> float:
        ''' c_200 [adim]'''
        return concentration.concentration(M=M200, mdef='200c', z=self.redshift, model='diemer19')

    def sigma_1h(self, R, M200):
        eps = 1e-6

        # if c200 is None:
        c200 = self.c_200(M200)

        deltac = (200.0/3.0) * ((c200**3) / (np.log(1.0 + c200) - (c200 / (1 + c200))))

        r200 = self.R_200(M200)
        x = (R * c200) / r200
        
        m1 = x < 1 - eps
        m2 = x > 1 + eps
        m3 = np.abs(x - 1) <= eps

        jota  = np.zeros_like(R)
        
        atanh = np.arctanh(np.sqrt((1.0 - x[m1]) / (1.0 + x[m1])))
        jota[m1] = (1.0 / (x[m1]**2 -1.0)) * (1.0 - (2.0 / np.sqrt(1.0 - x[m1]**2)) * atanh)

        atan = np.arctan(np.sqrt((x[m2] - 1.0) / (1.0 + x[m2])))
        jota[m2] = (1.0 / (x[m2]**2 - 1.0)) * (1.0 - (2.0 / np.sqrt(x[m2]**2 - 1.0)) * atan)

        jota[m3] = 1/3

        rs_m = r200/c200 # Mpc/h
        kapak = (2.0 * rs_m * deltac * self.rho_c) * (1e-12) #Msun/pc^2
        # Units M_sun/pc2
        return kapak * jota

    def dsigma_1h(self, R:np.ndarray[float], M200:float) -> np.ndarray[float]:
        '''
        Projected density contrast of NFW density model.
        M200 in solar masses
        R in h^-1 Mpc
        '''

        # if c200 is None:
        c200 = self.c_200(M200)

        deltac = (200.0/3.0) * (c200**3)/ (np.log(1.0 + c200) - c200 / (1.0 + c200))

        r200 = self.R_200(M200)
        x = (R * c200) / r200

        jota = np.zeros_like(x)

        m1 = x<1.0
        m2 = x>1.0
        m3 = ~(m1|m2) #not (m1 or m2) => not(x<1 or x>1) => not(x!=1) => x==1.0

        if np.any(m1):
            xm = x[m1]
            xm_sq = xm*xm

            sqrt_term = np.sqrt(1.0 - xm_sq)
            atanh = np.arctanh(np.sqrt((1.0 - xm) / (1.0 + xm)))

            jota[m1] = (
                (4.0 * atanh) / (xm_sq * sqrt_term)
                + (2.0 * np.log(xm / 2.0)) / xm_sq
                - 1.0 / (xm_sq - 1.0)
                + (2.0 * atanh) / ((xm_sq - 1.0) * sqrt_term)
            )

        if np.any(m2):
            xm = x[m2]
            xm_sq = xm*xm

            sqrt_term = np.sqrt(xm_sq - 1.0)
            atan = np.arctan(np.sqrt((xm - 1.0) / (1.0 + xm)))

            jota[m2] = (
                (4.0 * atan) / (xm_sq * sqrt_term)
                + (2.0 * np.log(xm / 2.0)) / xm_sq
                - 1.0 / (xm_sq - 1.0)
                + (2.0 * atan) / ((xm_sq - 1.0) ** 1.5)
            )

        if np.any(m3):
            jota[m3] = 2.0 * np.log(0.5) + 5.0 / 3.0

        rs_m = r200/c200
        kapak = (2.0 * rs_m * deltac * self.rho_c) * (1e-12) # Msun/pc^2

        return kapak * jota

    def sigma_miss(self, R, M200):

        R = np.atleast_1d(R) # for compatibility with eli funcs

        Ntheta = 100
        NRs = 100

        # if c200 is None:
        c200 = self.c_200(M200)

        # if s_off is None:
        s_off = 0.4
            #s_off = tau * self.R_200(M200)

        # --- integration grids ---
        theta = np.linspace(0, 2*np.pi, Ntheta)
        Rs = np.linspace(0, 5.0 * s_off, NRs)

        Rg = R[:, None, None]
        Rsg = Rs[None, :, None]
        thetag = theta[None, None, :]
        Rtrue = np.sqrt(Rg**2 + Rsg**2 + 2.0 * Rg * Rsg * np.cos(thetag))

        Sigma_vals = self.sigma_1h(Rtrue, M200)

        #angular average
        Sigma_theta = simpson(Sigma_vals, theta, axis=2) / (2.0 * np.pi)
        
        #miscentering pdf
        P = self.miss_pdf(Rs, s_off)

        Sigma_miss = simpson(Sigma_theta * P[None, :], Rs, axis=1)

        return Sigma_miss


    def dsigma_miss(self, R, M200):

        num_x = 200
        x_grid = np.linspace(1e-5, R.max(), num_x)

        integrand = x_grid * self.sigma_miss(x_grid, M200=M200)
        cumulative = cumulative_trapezoid(integrand, x_grid, initial=0.0)
        interp = np.interp(R, x_grid, cumulative)

        Sigma_miss = self.sigma_miss(R, M200)
        
        Sigma_bar = (2.0/R**2) * interp
        return Sigma_bar - Sigma_miss

    def dsigma_2h(self, R, M200):
        '''
        NFW contrast density from colossus
        units Msun/pc2
        '''

        # if c200 is None:
        c200 = self.c_200(M200)

        b = bias.haloBias(M200, model='tinker10', z=self.redshift, mdef='200c')
        outer_term = profile_outer.OuterTermCorrelationFunction(z=self.redshift, bias=b)
        pNFW = profile_nfw.NFWProfile(M=M200, mdef='200c', z=self.redshift, c=c200, outer_terms=[outer_term])    
        
        # Outer term integrated up to 50Mpc (Luo et al. 2017, Niemic et al 2017)
        ds_out = pNFW.deltaSigmaOuter(R*1.e3, interpolate=False, interpolate_surface_density=False, accuracy=0.01, max_r_integrate=100e3)
        return ds_out/(1.e3**2)

    def delta_sigma(self, R, M200, pcc):

        ds_cen = self.dsigma_1h(R, M200=M200)
        ds_miss = self.dsigma_miss(R, M200=M200)
        #ds_2h = self.dsigma_2h(R, M200, c200)

        return pcc * ds_cen + (1.0 - pcc) * ds_miss #+ ds_2h 


models_dict = {
    'NFW':NFW,
}
default_limits = {
    'NFW':{
        'M200':(1e10, 1e16), 
        # 'c200':(1.0, 10.0), 
        'pcc':(0.1,1.0), 
        # 's_off':(0.01,1.0)
    }
}
default_guess = {
    'NFW':{
        'M200':1e14, 
        # 'c200':4.0, 
        'pcc':0.8, 
        # 's_off':0.4
    },
}

# ==========================
#  DENSITY MODELS FOR VOIDS
# ==========================

# class HSW(BaseModelFast):
#     def density_contrast(self, r, dc, rs, a, b):
#         return dc*(1-(r/rs)**a)/(1+r**b)

# class B15(BaseModelFast):
#     def density_contrast(self, r, dc, rs, rv, a, b):
#         return dc*(1-(r/rs)**a)/(1+(r/rv)**b)

# class ModifiedLW(BaseModelFast):
#     def density_contrast(self, r, dc, dw, rw):
#         rv = 1.0
#         return np.where(r<rv, (dc-dw)*(1.0-(r/rv)**3), 0.0) + np.where(r<rw, dw, 0.0)


# class TopHat(BaseModelFast):
#     def density_contrast(self, r, dc, dw, rw):
#         rv = 1.0
#         return np.where(r<rv, dc-dw, 0.0) + np.where(r<rw, dw, 0.0)

#     # easier to compute since is integrable
#     def sigma(self, R, dc, dw, rw, sigma0=0.0):
#         rv = 1.0
#         return np.where(R<rv, (dc-dw)*np.sqrt(rv**2-R**2), 0.0) + np.where(R<rw, dw*np.sqrt(rw**2-R**2), 0.0) + sigma0

#     def delta_sigma(self, R, dc, dw, rw):
#         rv = 1.0
#         I1 = np.where(R<rv, 1/3*(dc-dw)*(rv**3-(rv**2-R**2)**(3/2)), 1/3*(dc-dw)*rv**3)
#         I2 = np.where(R<rw, 1/3*dw*(rw**3-(rw**2-R**2)**(3/2)), 1/3*dw*rw**3)

#         return (2.0/R**2)*(I1+I2) - self.sigma(R, dc, dw, rw)

# class Paz13(BaseModelFast):
#     def density_contrast(self, r, S, Rs, P, W):
#         x = np.log10(r/Rs)
#         asym_gauss = np.where(r<Rs, np.exp(-S*x**2), np.exp(-W*x**2))

#         Delta = 0.5*(erf(S*x)-1) + P*asym_gauss

#         t1 = S*np.exp(-(S*x)**2)/(SQPI*r)
#         t2 = (-2.0*P*x/r) * asym_gauss
#         Delta_prime = t1+t2

#         return Delta+1/3*r*Delta_prime

# class THLogistic(BaseModelFast):
#     def density_contrast(self, r, dc, dw, rw):
#         k=15
#         return (dc-dw)*(1.0-logistic(r, x0=1, k=k)) + dw*(1.0-logistic(r, x0=rw, k=k))

# class ModLWLogistic(BaseModelFast):
#     # not tested! weird values at r=rv
#     def density_contrast(self, r, dc, dw, rw):
#         rv = 1.0
#         k=15
#         return (dc-dw)*(1.0-(r/rv)**3)*(1.0-logistic(r, x0=rv, k=k)) + dw*(1.0-logistic(r, x0=rw, k=k))

# models_dict = {
#     'HSW':HSW(),
#     'TH':TopHat(),
#     'mLW':ModifiedLW(),
#     'B15':B15(),
# }
# default_limits = {
#     'HSW':{'dc':(-1.0,0.0),'rs':(0.5,5.0),'a':(1.0,15.0),'b':(1.0,15.0),'sigma0':(-0.5,0.5)},
#     'B15':{'dc':(-1.0,0.0),'rs':(0.5,5.0),'rv':(0.5,5.0),'a':(1.0,15.0),'b':(1.0,15.0),'sigma0':(-0.5,0.5)},
#     'TH':{'dc':(-1.0,0.0),'dw':(-0.5,0.5),'rw':(1.0,5.0),'sigma0':(-0.5,0.5)},
#     'mLW':{'dc':(-1.0,0.0),'dw':(-0.5,0.5),'rw':(1.0,5.0),'sigma0':(-0.5,0.5)},
# }
# default_guess = {
#     'HSW':(-0.7,0.9,3.0,6.0,0.0),
#     'B15':(-0.7,0.9,1.0,3.0,6.0,0.0),
#     'TH':(-0.7,0.2,2.5,0.0),
#     'mLW':(-0.7,0.2,2.5,0.0),
# }
