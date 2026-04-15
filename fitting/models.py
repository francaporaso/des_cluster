import numpy as np
from scipy.integrate import simpson, quad, cumulative_trapezoid
from scipy.special import erf
import astropy.units as u

from fitting.constants import *

def logistic(x, x0=1, k=10):
    return (1.0+np.exp(-2.0*k*(x-x0)))**(-1)

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
    def __init__(self, redshift):
        self.redshift = redshift
        self.roc_mpc = cosmo.critical_density(redshift).to(u.kg/(u.Mpc)**3).value

    def density(self):
        pass

    def R_200(self, M200:float) -> float | list[float]:
        '''    
        Returns the R_200
        ------------------------------------------------------------------
        INPUT:
        M200         (float or array of floats) M_200 mass in solar masses
        roc_mpc      (float or array of floats) Critical density at the z 
                    of the halo in units of kg/Mpc**3
        ------------------------------------------------------------------
        OUTPUT:
        R_200         (float or array of floats) 
        '''

        return ((M200*(3.0*Msun))/(800.0*np.pi*self.roc_mpc))**(1./3.)

    def c_200(self, mass):
        '''
        Concentration value using Duffy+2008 relation.
        mass in solar masses
        '''
        return 5.71*((mass/2.e12)**-0.084)*((1.+self.redshift)**(-0.47))

    def delta_sigma(self, R, M200, c200=None):

        '''
        Projected density contrast of NFW density model.
        M200 in solar masses
        R in h^-1 Mpc
        '''
    
        r200 = self.R_200(M200)
        
        if not c200:
            c200 = self.c_200(M200*cosmo.h)
        
        ####################################################
        
        deltac = (200./3.)*( (c200**3) / (np.log(1.+c200) - c200/(1+c200)) )
        x = np.round((R*c200)/r200, 12)
        m1 = (x < 1.0)
        m2 = (x > 1.0) 
        m3 = (x == 1.0)
        
        try: 
            jota = np.zeros_like(x)
            
            atanh = np.arctanh( ( (1.0-x[m1])/(1.0+x[m1]) )**0.5 )
            
            jota[m1] = (4.0*atanh)/((x[m1]**2.0)*((1.0-x[m1]**2.0)**0.5)) \
                + (2.0*np.log(x[m1]/2.0))/(x[m1]**2.0) - 1.0/(x[m1]**2.0-1.0) \
                + (2.0*atanh)/((x[m1]**2.0-1.0)*((1.0-x[m1]**2.0)**0.5))    
            
            atan = np.arctan( ( (x[m2]-1.0)/(1.0+x[m2]) )**0.5 )
            
            jota[m2]=(4.0*atan)/((x[m2]**2.0)*((x[m2]**2.0-1.0)**0.5)) \
                + (2.0*np.log(x[m2]/2.0))/(x[m2]**2.0) - 1.0/(x[m2]**2.0-1.0) \
                + (2.0*atan)/((x[m2]**2.0-1.0)**1.5)
            
            jota[m3]=2.0*np.log(0.5)+5.0/3.0
        
        except:
        
            if m1:
                atanh = np.arctanh( ( (1.0-x[m1])/(1.0+x[m1]) )**0.5 )
        
                jota = (4.0*atanh)/((x[m1]**2.0)*((1.0-x[m1]**2.0)**0.5)) \
                    + (2.0*np.log(x[m1]/2.0))/(x[m1]**2.0) - 1.0/(x[m1]**2.0-1.0) \
                    + (2.0*atanh)/((x[m1]**2.0-1.0)*((1.0-x[m1]**2.0)**0.5))   
        
            if m2:
                atan = np.arctan( ( (x[m2]-1.0)/(1.0+x[m2]) )**0.5 )
        
                jota = (4.0*atan)/((x[m2]**2.0)*((x[m2]**2.0-1.0)**0.5)) \
                    + (2.0*np.log(x[m2]/2.0))/(x[m2]**2.0) - 1.0/(x[m2]**2.0-1.0) \
                    + (2.0*atan)/((x[m2]**2.0-1.0)**1.5)
        
            if m3:
                jota = 2.0*np.log(0.5)+5.0/3.0
            
        rs_m=(r200*1.e6*pc)/c200
        kapak=( (2.0*rs_m*deltac*self.roc_mpc)*(pc**2/Msun) )/( (pc*1.0e6)**3.0 )
        return kapak*jota

models_dict = {
    'NFW':NFW,
}
default_limits = {
    'NFW':{'M200':(1e10, 1e16), 'c200':(1.0, 10.0)}
}
default_guess = {
    'NFW':(1e14, 3.0),
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
