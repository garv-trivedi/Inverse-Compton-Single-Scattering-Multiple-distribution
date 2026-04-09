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
    "Seed spectra for Multi-colour Blackbody cases are derived from the uploaded CSV."
)

# -----------------------------------------------------------------------------
# File Uploader - THIS FIXES THE "FILE NOT FOUND" ERROR
# -----------------------------------------------------------------------------
st.sidebar.header("Data Input")
uploaded_file = st.sidebar.file_uploader("Upload '2026-04-09T18-00_export.csv'", type="csv")

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
    return np.logspace(np.log10(float(lo)), np.log10(float(hi)), int(npts))

def plot_spectrum(x, y, title, x_label, y_label):
    x, y = np.asarray(x), np.asarray(y)
    mask = (x > 0) & (y > 0)
    fig, ax = plt.subplots(figsize=(6, 4))
    if np.any(mask):
        ax.loglog(x[mask], y[mask])
    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    st.pyplot(fig)
    plt.close(fig)

# -----------------------------------------------------------------------------
# Seed photon spectra
# -----------------------------------------------------------------------------
def seed_powerlaw_nu(nu, alpha, norm, nu_min, nu_max, nu0):
    Fnu = norm * (nu / nu0) ** (-alpha)
    return np.where((nu >= nu_min) & (nu <= nu_max), Fnu, 0.0)

def seed_blackbody_nu(nu, T_seed_K, norm):
    x = (H * nu) / (KB_SI * T_seed_K)
    x = np.clip(x, 1e-12, 700.0)
    return norm * (2.0 * H * nu**3 / C**2) / np.expm1(x)

def seed_multicolor_bb_nu(nu, norm):
    """Interpolates the seed spectrum from the uploaded CSV file."""
    if uploaded_file is None:
        st.warning("Please upload the CSV file in the sidebar to see Multicolor BB results.")
        return np.zeros_like(nu)
    
    df = pd.read_csv(uploaded_file)
    f_csv = df['frequencies (Hz)'].values
    y_csv = df['nuLnu watts'].values
    
    # Fnu = (nu*Lnu) / nu
    mask = (f_csv > 0) & (y_csv > 0)
    f_csv, fnu_csv = f_csv[mask], y_csv[mask] / f_csv[mask]
    
    # Log-log interpolation
    return norm * (10**np.interp(np.log10(nu), np.log10(f_csv), np.log10(fnu_csv), left=-100, right=-100))

def flux_to_seed_number_density(nu, Fnu):
    return Fnu / np.maximum(nu_to_eps_keV(nu), 1e-300)

# -----------------------------------------------------------------------------
# Electron spectra
# -----------------------------------------------------------------------------
def electron_powerlaw_E(E_keV, p, nth, Emin, Emax):
    shape = np.where((E_keV >= Emin) & (E_keV <= Emax), E_keV**(-p), 0.0)
    return normalize_to_area(E_keV, shape, nth)

def electron_mj_E(E_keV, nth, T_e_K):
    theta = (KB_KEV_PER_K * T_e_K) / ME_C2_KEV
    gamma = 1.0 + E_keV / ME_C2_KEV
    beta = np.sqrt(np.clip(1.0 - 1.0/gamma**2, 0.0, None))
    shape = gamma**2 * beta * np.exp(-(gamma-1)/theta)
    return normalize_to_area(E_keV, shape, nth)

def electron_mb_E(E_keV, nth, T_e_K):
    kT = KB_KEV_PER_K * T_e_K
    shape = np.sqrt(E_keV) * np.exp(-E_keV / kT)
    return normalize_to_area(E_keV, shape, nth)

# -----------------------------------------------------------------------------
# IC Kernel & Emissivity
# -----------------------------------------------------------------------------
def kn_dsigma_d_es(eps_s_keV, seed_eps_keV, gamma_grid):
    gamma = gamma_grid[:, None]
    eps = seed_eps_keV[None, :]
    Gamma_e = 4.0 * gamma * eps / ME_C2_KEV
    denom = Gamma_e * (gamma * ME_C2_KEV - eps_s_keV)
    valid = (Gamma_e > 0) & (denom > 0) & (eps_s_keV < gamma * ME_C2_KEV)
    q = np.where(valid, eps_s_keV / denom, 0.0)
    valid = valid & (q >= 1.0/(4.0*gamma**2)) & (q <= 1.0)
    q_safe = np.clip(q, 1e-300, 1.0)
    F = 2*q_safe*np.log(q_safe) + (1+2*q_safe)*(1-q_safe) + 0.5*(Gamma_e*q_safe)**2*(1-q_safe)/(1+Gamma_e*q_safe)
    return np.where(valid, (3.0 * SIGMA_T / (4.0 * gamma**2 * eps)) * F, 0.0)

def ic_emissivity(eps_s_grid, seed_eps, seed_n, e_grid, ne_e):
    gamma_grid = 1.0 + e_grid / ME_C2_KEV
    output = []
    for es in eps_s_grid:
        dsdE = kn_dsigma_d_es(es, seed_eps, gamma_grid)
        total = integrate(ne_e * integrate(seed_n[None,:] * dsdE * seed_eps[None,:], seed_eps, axis=1), e_grid)
        output.append(C * es * total)
    return np.asarray(output)

# -----------------------------------------------------------------------------
# UI Sidebar Controls
# -----------------------------------------------------------------------------
seed_amp = st.sidebar.number_input("Seed amplitude", value=1e6, format="%.2e")
n_pts = st.sidebar.slider("Resolution (Points)", 40, 300, 100)
out_lo = st.sidebar.number_input("Output min (keV)", value=1e-2, format="%.2e")
out_hi = st.sidebar.number_input("Output max (keV)", value=1e3, format="%.2e")
eps_s_grid = positive_log_grid(out_lo, out_hi, n_pts)

# Electron Params
st.sidebar.subheader("Electron Params")
pl_e_p = st.sidebar.number_input("PL Index p", value=2.5)
Te = st.sidebar.number_input("Temp T_e (K)", value=1e9)
nth = st.sidebar.number_input("Density n_th", value=1e6)

# -----------------------------------------------------------------------------
# Tabs and Display
# -----------------------------------------------------------------------------
t1, t2, t3, t4, t5 = st.tabs(["PL+PL", "BB+PL", "CSV+PL", "CSV+MJ", "CSV+MB"])

# Case 3: CSV + Power Law (Example of integration)
with t3:
    st.header("MCD (CSV) + Power-law Electrons")
    nu = positive_log_grid(1e10, 1e18, n_pts)
    seed_Fnu = seed_multicolor_bb_nu(nu, seed_amp)
    e_grid = positive_log_grid(1e-1, 1e5, n_pts)
    ne = electron_powerlaw_E(e_grid, pl_e_p, nth, 1e-1, 1e5)
    
    if uploaded_file:
        emiss = ic_emissivity(eps_s_grid, nu_to_eps_keV(nu), flux_to_seed_number_density(nu, seed_Fnu), e_grid, ne)
        display_case = True
    else:
        emiss = np.zeros_like(eps_s_grid)
        display_case = False

    c1, c2, c3 = st.columns(3)
    with c1: plot_spectrum(nu, seed_Fnu, "Seed (CSV)", "Hz", "Flux")
    with c2: plot_spectrum(e_grid, ne, "Electrons (PL)", "keV", "N(E)")
    with c3: plot_spectrum(eps_s_grid, emiss, "IC Emissivity", "keV", "J/s/keV/m3")

st.markdown("<div style='text-align: right;'><p><strong>By Garv Trivedi</strong></p></div>", unsafe_allow_html=True)
