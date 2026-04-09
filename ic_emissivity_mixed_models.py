import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# -----------------------------------------------------------------------------
# Page setup
# -----------------------------------------------------------------------------
st.set_page_config(page_title="Inverse Compton Multi-Case Simulator", layout="wide")
st.title("Inverse Compton Up-scattering for various electron populations")

st.caption(
    "This proof-of-concept uses one shared inverse-Compton kernel for all five cases. "
    "Seed spectra are shown in relative/controlled units, while electron spectra are "
    "normalized numerically so the thermal branches remain visible."
)

# -----------------------------------------------------------------------------
# Constants
# -----------------------------------------------------------------------------
C = 2.99792458e8                 # m/s
H = 6.62607015e-34               # J s
KB_SI = 1.380649e-23             # J/K
SIGMA_T = 6.6524587158e-29       # m^2
ME_C2_KEV = 510.99895            # keV
KB_KEV_PER_K = 8.617333262145e-8  # keV/K
KEV_J = 1.602176634e-16          # J per keV
PI = np.pi

try:
    from scipy.integrate import simpson as scipy_simpson
except Exception:
    scipy_simpson = None


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def integrate(y, x, axis=-1):
    """SciPy Simpson if available, otherwise NumPy trapezoid."""
    if scipy_simpson is not None:
        return scipy_simpson(y, x=x, axis=axis)
    return np.trapezoid(y, x=x, axis=axis)


def normalize_to_area(x, y, target_area):
    area = integrate(y, x)
    if not np.isfinite(area) or area <= 0:
        return y
    return y * (target_area / area)


def nu_to_eps_keV(nu_hz):
    return (H * np.asarray(nu_hz, dtype=float)) / KEV_J


def peak_nu_from_T(T_K):
    return 2.821439 * KB_SI * T_K / H


def positive_log_grid(lo, hi, npts):
    lo = float(lo)
    hi = float(hi)
    npts = int(npts)
    if lo <= 0 or hi <= 0:
        raise ValueError("Log grids require positive limits.")
    if lo >= hi:
        raise ValueError("Lower limit must be smaller than upper limit.")
    return np.logspace(np.log10(lo), np.log10(hi), npts)


def plot_spectrum(x, y, title, x_label, y_label):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y) & (x > 0) & (y > 0)

    fig, ax = plt.subplots(figsize=(6.5, 4.8))
    if np.any(mask):
        ax.loglog(x[mask], y[mask], linewidth=1.6)
    else:
        ax.text(0.5, 0.5, "No positive data to plot", ha="center", va="center", transform=ax.transAxes)

    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.grid(True, which="both", alpha=0.3)
    st.pyplot(fig)
    plt.close(fig)


# -----------------------------------------------------------------------------
# Seed photon spectra
# -----------------------------------------------------------------------------
def seed_powerlaw_nu(nu, alpha, norm, nu_min, nu_max, nu0):
    nu = np.asarray(nu, dtype=float)
    Fnu = norm * (nu / nu0) ** (-alpha)
    mask = (nu >= nu_min) & (nu <= nu_max)
    return np.where(mask, Fnu, 0.0)


def seed_blackbody_nu(nu, T_seed_K, norm):
    nu = np.asarray(nu, dtype=float)
    x = (H * nu) / (KB_SI * T_seed_K)
    x = np.clip(x, 1e-12, 700.0)
    return norm * (2.0 * H * nu**3 / C**2) / np.expm1(x)


def seed_multicolor_bb_nu(nu, Tin_K, norm, rout_over_rin=1e3, n_r=200):
    """
    Multicolor blackbody disk shape:
    F_nu ∝ ∫ 2π r B_nu[T(r)] dr, with T(r) = Tin * (r/rin)^(-3/4)
    Here r is dimensionless in units of rin.
    """
    nu = np.asarray(nu, dtype=float)
    r = np.logspace(0.0, np.log10(rout_over_rin), int(n_r))
    T_r = Tin_K * r ** (-0.75)

    nu2d = nu[None, :]
    Tr2d = T_r[:, None]
    x = (H * nu2d) / (KB_SI * Tr2d)
    x = np.clip(x, 1e-12, 700.0)

    Bnu = (2.0 * H * nu2d**3 / C**2) / np.expm1(x)
    integrand = 2.0 * PI * r[:, None] * Bnu
    Fnu = integrate(integrand, x=r, axis=0)
    return norm * Fnu


def flux_to_seed_number_density(nu, Fnu):
    """
    We only need a photon-spectrum shape for the scattering integral.
    Converting F_nu to an energy-spectrum shape by dividing by photon energy
    is sufficient for the proof-of-concept.
    """
    eps_keV = np.maximum(nu_to_eps_keV(nu), 1e-300)
    return np.asarray(Fnu, dtype=float) / eps_keV


# -----------------------------------------------------------------------------
# Electron spectra as N(epsilon) vs epsilon
# -----------------------------------------------------------------------------
def electron_powerlaw_E(E_keV, p, nth, Emin_keV, Emax_keV):
    E_keV = np.asarray(E_keV, dtype=float)
    gamma = 1.0 + E_keV / ME_C2_KEV
    shape = gamma ** (-p)
    mask = (E_keV >= Emin_keV) & (E_keV <= Emax_keV)
    shape = np.where(mask, shape, 0.0)
    shape = E_keV ** (-p)
    return normalize_to_area(E_keV, shape, nth)


def electron_mb_E(E_keV, nth, T_e_K):
    E_keV = np.asarray(E_keV, dtype=float)
    kT_keV = KB_KEV_PER_K * T_e_K
    E_keV = np.maximum(E_keV, 0.0)
    shape = (2.0 / np.sqrt(PI)) * np.sqrt(E_keV) * np.exp(-E_keV / kT_keV) / (kT_keV ** 1.5)
    return normalize_to_area(E_keV, shape, nth)


def electron_mj_E(E_keV, nth, T_e_K):
    """
    Numerically normalized Maxwell-Jüttner shape.
    We use exp(-(gamma-1)/theta) for stability; the missing constant factor is
    absorbed by the numerical normalization.
    """
    E_keV = np.asarray(E_keV, dtype=float)
    theta = (KB_KEV_PER_K * T_e_K) / ME_C2_KEV
    theta = max(theta, 1e-12)

    gamma = 1.0 + E_keV / ME_C2_KEV
    beta = np.sqrt(np.clip(1.0 - 1.0 / gamma**2, 0.0, None))
    shape = gamma**2 * beta * np.exp(-(gamma - 1.0) / theta) / ME_C2_KEV
    return normalize_to_area(E_keV, shape, nth)


# -----------------------------------------------------------------------------
# Inverse Compton kernel
# -----------------------------------------------------------------------------
def kn_dsigma_d_es(eps_s_keV, seed_eps_keV, gamma_grid):
    """
    Differential KN cross-section used in the isotropic IC kernel.
    Energies are in keV.
    """
    gamma = np.asarray(gamma_grid, dtype=float)[:, None]
    eps = np.asarray(seed_eps_keV, dtype=float)[None, :]
    eps_s = float(eps_s_keV)

    if eps_s <= 0:
        return np.zeros((gamma.shape[0], eps.shape[1]))

    Gamma_e = 4.0 * gamma * eps / ME_C2_KEV
    denom = Gamma_e * (gamma * ME_C2_KEV - eps_s)

    valid = (Gamma_e > 0) & (denom > 0) & (eps > 0) & (eps_s < gamma * ME_C2_KEV)
    q = np.where(valid, eps_s / denom, 0.0)
    valid = valid & (q >= 1.0 / (4.0 * gamma**2)) & (q <= 1.0)

    q_safe = np.clip(q, 1e-300, 1.0)
    F = (
        2.0 * q_safe * np.log(q_safe)
        + (1.0 + 2.0 * q_safe) * (1.0 - q_safe)
        + 0.5 * (Gamma_e * q_safe) ** 2 * (1.0 - q_safe) / (1.0 + Gamma_e * q_safe)
    )

    eps_safe = np.where(eps > 0, eps, 1.0)
    dsdE = (3.0 * SIGMA_T / (4.0 * gamma**2 * eps_safe)) * F
    return np.where(valid, dsdE, 0.0)


def ic_emissivity(eps_s_grid, seed_eps, seed_n, e_grid_keV, ne_e):
    """
    j_IC(eps_s) = c * eps_s * ∫ dE_e N_e(E_e) ∫ dε n_ph(ε) [dσ_IC/dε_s]
    """
    seed_eps = np.asarray(seed_eps, dtype=float)
    seed_n = np.asarray(seed_n, dtype=float)
    e_grid_keV = np.asarray(e_grid_keV, dtype=float)
    ne_e = np.asarray(ne_e, dtype=float)

    gamma_grid = 1.0 + e_grid_keV / ME_C2_KEV

    output = []
    for eps_s in np.asarray(eps_s_grid, dtype=float):
        dsdE = kn_dsigma_d_es(eps_s, seed_eps, gamma_grid)
        inner_seed = integrate(seed_n[None, :] * dsdE * seed_eps[None, :], x=seed_eps, axis=1)
        total = integrate(ne_e * inner_seed, x=e_grid_keV)
        output.append(C * eps_s * total)
    return np.asarray(output)


# -----------------------------------------------------------------------------
# UI controls
# -----------------------------------------------------------------------------
st.sidebar.header("Common plotting / sampling controls")
seed_amp = st.sidebar.number_input("Seed amplitude / normalization", value=1e6, format="%.2e")
n_seed = st.sidebar.number_input("Seed frequency points", value=120, min_value=40, max_value=1000)
n_e = st.sidebar.number_input("Electron energy points", value=120, min_value=40, max_value=1000)
n_out = st.sidebar.number_input("Scattered photon points", value=120, min_value=40, max_value=1000)

out_lo = st.sidebar.number_input("Lower scattered photon energy ε₁ (keV)", value=1e-2, format="%.2e", min_value=1e-12)
out_hi = st.sidebar.number_input("Upper scattered photon energy ε₁ (keV)", value=1e3, format="%.2e", min_value=1e-12)

if out_lo <= 0 or out_hi <= 0 or out_lo >= out_hi:
    st.error("Scattered photon energy limits must be positive and satisfy lower < upper.")
    st.stop()

eps_s_grid = positive_log_grid(out_lo, out_hi, int(n_out))

# -----------------------------------------------------------------------------
# Case parameters
# -----------------------------------------------------------------------------
st.sidebar.header("Power-law seed photons")
pl_seed_alpha = st.sidebar.number_input("Seed photon index α", value=1.5)
pl_seed_nu_min = st.sidebar.number_input("Seed ν min (Hz)", value=1e8, format="%.2e", min_value=1e-30)
pl_seed_nu_max = st.sidebar.number_input("Seed ν max (Hz)", value=1e20, format="%.2e", min_value=1e-30)
pl_seed_nu0 = st.sidebar.number_input("Reference ν0 (Hz)", value=1e15, format="%.2e", min_value=1e-30)

if pl_seed_nu_min >= pl_seed_nu_max:
    st.error("Power-law seed ν min must be smaller than ν max.")
    st.stop()

st.sidebar.header("Power-law electrons")
pl_e_p = st.sidebar.number_input("Electron index p", value=2.5)
pl_e_Emin = st.sidebar.number_input("Electron Emin (keV)", value=1e-3, format="%.2e", min_value=1e-30)
pl_e_Emax = st.sidebar.number_input("Electron Emax (keV)", value=1e6, format="%.2e", min_value=1e-30)

if pl_e_Emin >= pl_e_Emax:
    st.error("Power-law electron Emin must be smaller than Emax.")
    st.stop()

st.sidebar.header("Blackbody seed photons")
bb_T = st.sidebar.number_input("Blackbody temperature T (K)", value=1e6, min_value=1.0)
st.sidebar.header("Multicolor blackbody seed photons")
mcd_Tin = st.sidebar.number_input("Inner disk temperature T_in (K)", value=1e7, min_value=1.0)
mcd_rout_over_rin = st.sidebar.number_input("Outer/inner radius ratio", value=1e3, min_value=2.0, format="%.2e")

st.sidebar.header("Thermal electrons")
Te = st.sidebar.number_input("Thermal electron temperature T_e (K)", value=1e9, min_value=1.0)
nth = st.sidebar.number_input("Thermal electron density n_th", value=1e6, min_value=0.0)

# -----------------------------------------------------------------------------
# Case builders
# -----------------------------------------------------------------------------
def make_powerlaw_powerlaw_case():
    nu = positive_log_grid(pl_seed_nu_min, pl_seed_nu_max, int(n_seed))
    seed_Fnu = seed_powerlaw_nu(nu, pl_seed_alpha, seed_amp, pl_seed_nu_min, pl_seed_nu_max, pl_seed_nu0)
    seed_eps = nu_to_eps_keV(nu)
    seed_n = flux_to_seed_number_density(nu, seed_Fnu)

    e_grid = positive_log_grid(pl_e_Emin, pl_e_Emax, int(n_e))
    ne = electron_powerlaw_E(e_grid, pl_e_p, nth, pl_e_Emin, pl_e_Emax)

    emiss = ic_emissivity(eps_s_grid, seed_eps, seed_n, e_grid, ne)
    return nu, seed_Fnu, e_grid, ne, emiss


def make_blackbody_powerlaw_case():
    nu_peak = max(peak_nu_from_T(bb_T), 1e8)
    nu = positive_log_grid(nu_peak * 1e-3, nu_peak * 1e2, int(n_seed))
    seed_Fnu = seed_blackbody_nu(nu, bb_T, seed_amp)
    seed_eps = nu_to_eps_keV(nu)
    seed_n = flux_to_seed_number_density(nu, seed_Fnu)

    e_grid = positive_log_grid(pl_e_Emin, pl_e_Emax, int(n_e))
    ne = electron_powerlaw_E(e_grid, pl_e_p, nth, pl_e_Emin, pl_e_Emax)

    emiss = ic_emissivity(eps_s_grid, seed_eps, seed_n, e_grid, ne)
    return nu, seed_Fnu, e_grid, ne, emiss


def make_mcd_powerlaw_case():
    nu_peak = max(peak_nu_from_T(mcd_Tin), 1e8)
    nu = positive_log_grid(nu_peak * 1e-3, nu_peak * 1e2, int(n_seed))
    seed_Fnu = seed_multicolor_bb_nu(
        nu,
        Tin_K=mcd_Tin,
        norm=seed_amp,
        rout_over_rin=mcd_rout_over_rin,
        n_r=200,
    )
    seed_eps = nu_to_eps_keV(nu)
    seed_n = flux_to_seed_number_density(nu, seed_Fnu)

    e_grid = positive_log_grid(pl_e_Emin, pl_e_Emax, int(n_e))
    ne = electron_powerlaw_E(e_grid, pl_e_p, nth, pl_e_Emin, pl_e_Emax)

    emiss = ic_emissivity(eps_s_grid, seed_eps, seed_n, e_grid, ne)
    return nu, seed_Fnu, e_grid, ne, emiss


def thermal_energy_grid(T_K, npts):
    kT_keV = KB_KEV_PER_K * T_K
    Emin = max(1e-4 * kT_keV, 1e-8)
    Emax = max(25.0 * kT_keV, Emin * 10.0)
    return positive_log_grid(Emin, Emax, int(npts))


def make_mcd_mj_case():
    nu_peak = max(peak_nu_from_T(mcd_Tin), 1e8)
    nu = positive_log_grid(nu_peak * 1e-3, nu_peak * 1e2, int(n_seed))
    seed_Fnu = seed_multicolor_bb_nu(
        nu,
        Tin_K=mcd_Tin,
        norm=seed_amp,
        rout_over_rin=mcd_rout_over_rin,
        n_r=200,
    )
    seed_eps = nu_to_eps_keV(nu)
    seed_n = flux_to_seed_number_density(nu, seed_Fnu)

    e_grid = thermal_energy_grid(Te, n_e)
    ne = electron_mj_E(e_grid, nth, Te)

    emiss = ic_emissivity(eps_s_grid, seed_eps, seed_n, e_grid, ne)
    return nu, seed_Fnu, e_grid, ne, emiss


def make_mcd_mb_case():
    nu_peak = max(peak_nu_from_T(mcd_Tin), 1e8)
    nu = positive_log_grid(nu_peak * 1e-3, nu_peak * 1e2, int(n_seed))
    seed_Fnu = seed_multicolor_bb_nu(
        nu,
        Tin_K=mcd_Tin,
        norm=seed_amp,
        rout_over_rin=mcd_rout_over_rin,
        n_r=200,
    )
    seed_eps = nu_to_eps_keV(nu)
    seed_n = flux_to_seed_number_density(nu, seed_Fnu)

    e_grid = thermal_energy_grid(Te, n_e)
    ne = electron_mb_E(e_grid, nth, Te)

    emiss = ic_emissivity(eps_s_grid, seed_eps, seed_n, e_grid, ne)
    return nu, seed_Fnu, e_grid, ne, emiss


# -----------------------------------------------------------------------------
# Case display helper
# -----------------------------------------------------------------------------
def display_case(case_title, nu, seed_Fnu, e_grid, ne, emiss):
    st.subheader(case_title)

    c1, c2, c3 = st.columns(3)

    with c1:
        plot_spectrum(
            nu,
            seed_Fnu,
            "Raw seed photon spectrum",
            "Frequency ν (Hz)",
            "Flux density (W m^-2 Hz^-1)",
        )

    with c2:
        plot_spectrum(
            e_grid,
            ne,
            "Electron spectrum",
            "Electron energy ε (keV)",
            "N(ε)",
        )

    with c3:
        plot_spectrum(
            eps_s_grid,
            emiss,
            "Scattered volume emissivity",
            "Scattered photon energy ε₁ (keV)",
            "Volume emissivity (J s^-1 keV^-1 m^-3)",
        )

    out_df = pd.DataFrame(
        {
            "Epsilon_1_keV": eps_s_grid,
            "Volume_Emissivity": emiss,
        }
    )

    st.dataframe(out_df, use_container_width=True)
    st.download_button(
        label=f"Download {case_title} emissivity CSV",
        data=out_df.to_csv(index=False).encode("utf-8"),
        file_name=f"{case_title.lower().replace(' ', '_')}_emissivity.csv",
        mime="text/csv",
    )


# -----------------------------------------------------------------------------
# Compute all five cases
# -----------------------------------------------------------------------------
tab1, tab2, tab3, tab4, tab5 = st.tabs(
    [
        "1) Power-law seed + Power-law electrons",
        "2) Blackbody seed + Power-law electrons",
        "3) Multicolor BB seed + Power-law electrons",
        "4) Multicolor BB seed + Maxwell-Jüttner electrons",
        "5) Multicolor BB seed + Maxwell-Boltzmann electrons",
    ]
)

with tab1:
    nu, seed_Fnu, e_grid, ne, emiss = make_powerlaw_powerlaw_case()
    display_case("Power-law seed + Power-law electrons", nu, seed_Fnu, e_grid, ne, emiss)

with tab2:
    nu, seed_Fnu, e_grid, ne, emiss = make_blackbody_powerlaw_case()
    display_case("Blackbody seed + Power-law electrons", nu, seed_Fnu, e_grid, ne, emiss)

with tab3:
    nu, seed_Fnu, e_grid, ne, emiss = make_mcd_powerlaw_case()
    display_case("Multicolor blackbody seed + Power-law electrons", nu, seed_Fnu, e_grid, ne, emiss)

with tab4:
    nu, seed_Fnu, e_grid, ne, emiss = make_mcd_mj_case()
    display_case("Multicolor blackbody seed + Maxwell-Jüttner electrons", nu, seed_Fnu, e_grid, ne, emiss)
    st.caption("If this curve is too small, increase Seed amplitude or T_e slightly.")

with tab5:
    nu, seed_Fnu, e_grid, ne, emiss = make_mcd_mb_case()
    display_case("Multicolor blackbody seed + Maxwell-Boltzmann electrons", nu, seed_Fnu, e_grid, ne, emiss)
    st.caption("If this curve is too small, increase Seed amplitude or T_e slightly.")

st.markdown(
    """
    <div style='text-align: right;'>
        <p><strong>By Garv Trivedi</strong></p>
        <p><strong>under guidance of Dr. C. Konar</strong></p>
    </div>
    """,
    unsafe_allow_html=True  
)

