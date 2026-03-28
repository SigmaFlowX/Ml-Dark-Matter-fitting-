from utils import osmirnov_anisotropy, nu_plummer_profile, mass_nfw
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d
import numpy as np
import matplotlib.pyplot as plt

G = 4.302e-6  # kpc * (km/s)^2 / Msun


def dnu_plummer(r, a, m0):
    return -15 * m0 / (4 * np.pi * a**5) * r * (1 + r**2 / a**2)**(-7/2)

def mass_profile(r_grid, rho_s, r_s):
    M = np.array([mass_nfw(r, rho_s, r_s) for r in r_grid])

    return interp1d(
        r_grid,
        M,
        kind="cubic",
        fill_value="extrapolate"
    )


def jeans_eq(r, sigma2, M_interp, a, m0, beta_inf, r_beta):
    nu = nu_plummer_profile(r, a, m0)
    dnu = dnu_plummer(r, a, m0)
    beta = osmirnov_anisotropy(r, beta_inf, r_beta)

    return  -G*M_interp(r)/r**2 - (2*beta/r + dnu/nu)*sigma2

def solve_jeans_eq(r_grid, rho_s, r_s, a, mo, beta_inf, r_beta):
    M_interp = mass_profile(r_grid, rho_s, r_s)
    sol = solve_ivp(
        fun=lambda r, y: jeans_eq(r, y, M_interp, a, m0, beta_inf, r_beta),
        t_span=(r_grid[-1], r_grid[0]),
        y0=[0.0],
        t_eval=r_grid[::-1],
        method="RK45"
    )

    sigma2 = sol.y[0][::-1]
    return sigma2

if __name__ == "__main__":
    rho_s = 0.1
    r_s = 5
    a = 1
    m0 = 1e7
    beta_inf = 0.3
    r_beta = 2
    r_grid = np.logspace(-2, 1, 120)

    sigma2 = solve_jeans_eq(r_grid, rho_s, r_s, a, m0, beta_inf, r_beta)


    plt.loglog(r_grid, sigma2)
    plt.xlabel("r [kpc]")
    plt.ylabel("sigma_r^2")
    plt.grid()
    plt.show()