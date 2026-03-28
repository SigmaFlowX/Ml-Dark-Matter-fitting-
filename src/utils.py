

def nsfw_profile(rho_s, r_s, r):
    return rho_s/(r/r_s)/(1+r/r_s)**2

def osmirnov_anisotropy(beta_inf, r_beta, r):
    return beta_inf * r / (r + r_beta)
