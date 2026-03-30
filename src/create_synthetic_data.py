from solve_jeans_eq import solve_jeans_eq
import numpy as np
from scipy.interpolate import interp1d
from utils import osmirnov_anisotropy

def create_data(r_grid, rho_s, r_s, a, m0, beta_inf, r_beta, n_stars):
    sigma2 = solve_jeans_eq(r_grid, rho_s, r_s, a, m0, beta_inf, r_beta)
    sigma2_funct =  interp1d(r_grid, sigma2, kind="cubic", fill_value="extrapolate")

    radii = generate_star_radii_analytic(n_stars, a)
    sigmas = sigma2_funct(radii)

    R = radii * np.sqrt(np.random.uniform(0, 1, len(radii)))

    beta_vals = osmirnov_anisotropy(radii, beta_inf, r_beta)

    sigma_los2 = sigmas * (1 - beta_vals * R ** 2 / radii ** 2)

    sigma_los2 = np.clip(sigma_los2, 1e-10, None)

    v_los = np.random.normal(0, np.sqrt(sigma_los2))

    return radii, v_los


def generate_star_radii_analytic(n_stars, a):
    u = np.random.uniform(0, 1, n_stars)
    r_samples = a * np.sqrt((1 - u)**(-2/3) - 1)
    return r_samples

if __name__ == "__main__":

    create_data(
        r_grid = np.logspace(-2, 1, 120),
        rho_s = 0.1,
        r_s = 5,
        a = 1,
        m0 = 1e7,
        beta_inf = 0.3,
        r_beta = 2,
        n_stars =5000
    )



