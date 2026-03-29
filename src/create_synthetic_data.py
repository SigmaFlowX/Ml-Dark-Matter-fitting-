from solve_jeans_eq import solve_jeans_eq
import numpy as np
import matplotlib.pyplot as plt

def create_data(r_grid, rho_s, r_s, a, m0, beta_inf, r_beta, n_stars):
    sigma2 = solve_jeans_eq(r_grid, rho_s, r_s, a, m0, beta_inf, r_beta)

    radii = generate_star_radii_analytic(n_stars, a)
     #.....

def generate_star_radii_analytic(n_stars, a):
    u = np.random.uniform(0, 1, n_stars)
    r_samples = a * np.sqrt((1 - u)**(-2/3) - 1)
    return r_samples

if __name__ == "__main__":
    samples = generate_star_radii_analytic(5000, 1)

    plt.hist(samples, bins=100, density=True)
    plt.show()



