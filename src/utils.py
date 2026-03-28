def nsfw_profile(rho_s, r_s, r):
    return rho_s/(r/r_s)/(1+r/r_s)**2

def osmirnov_anisotropy(beta_inf, r_beta, r):
    return beta_inf * r / (r + r_beta)

def plummer_profile(a, m0, r):
    return 3 * m0 / (4 * 3.14 * a**3) * (1 + r**2 / a**2)**(-5/2)
