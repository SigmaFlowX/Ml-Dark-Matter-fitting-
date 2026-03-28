import numpy as np
from scipy.integrate import quad

def rho_nfw(r, rho_s, r_s):
    return rho_s/(r/r_s)/(1+r/r_s)**2

def osmirnov_anisotropy(r, beta_inf, r_beta):
    return beta_inf * r / (r + r_beta)

def nu_plummer_profile(r, a, m0):
    return 3 * m0 / (4 * 3.14 * a**3) * (1 + r**2 / a**2)**(-5/2)

def mass_nfw(r, rho_s, r_s): # M(<r)
    integral = lambda x: 4 *  np.pi * x**2 * rho_nfw(r, rho_s, r_s)
    return quad(integral, 0, r)