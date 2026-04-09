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
    "Seed spectra for Multi-colour cases are loaded from 2026-04-09T18-00_export.csv. "
    "Calculations use log-log interpolation for smooth results."
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

# NumPy 2.0 Compatibility Fix
def trapezoid_rule(y, x, axis=-1):
    """Handles renaming of np.trapz to np.trapezoid in NumPy 2.0+"""
    if hasattr(np, 'trapezoid'):
        return np.trapezoid(y, x=x, axis=axis)
    return np.trapz(y, x=x, axis=axis)

# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def normalize_to_area(x, y, target_area):
    area = trapezoid_rule(y, x)
    if not np.isfinite(area) or area <= 0:
        return y
    return y * (target_area / area)

def nu_to_eps_keV(nu_hz):
    return (H * np.asarray(nu_hz, dtype=float)) / KEV_J

def positive_log_grid(lo, hi, npts):
    return np.logspace(np.log10(float(lo)), np.log10(float(hi)), int(npts))

def plot_spectrum(x, y, title, x_label, y_label):
    x, y = np.asarray(x), np.asarray(y)
    mask = (x > 0) & (y > 0)
    fig, ax = plt.subplots(figsize=(6, 4.5))
    if np.any(mask):
        ax.loglog(x[mask], y[mask], linewidth=1.8)
    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.grid(True, which="both", alpha=0.3)
    st.pyplot(fig)
    plt.close(fig)

# -----------------------------------------------------------------------------
# Seed Spectra (Interpreting CSV)
# -----------------------------------------------------------------------------
def get_csv_seed_fnu(nu_target, norm):
    """Interpolates CSV data in log-log space for a uniform line."""
    try:
        df = pd.read_csv('2026-04-09T18-00_export.csv')
        f_csv = df['frequencies (Hz)'].values
        y_csv = df['nuLnu watts'].values
        
        mask = (f_csv > 0) & (y_csv > 0)
        f_csv, fnu_csv = f_csv[mask], y_csv[mask] / f_csv[mask]
        
        # Log-log interpolation prevents the 'broken' line appearance
        log_fnu_interp = np.interp(
            np.log10(nu_target), 
            np.log10(f_csv), 
            np.log10(fnu_csv), 
            left=-100, right=-100
        )
        return norm * (10**log_fnu_interp)
    except Exception as e:
        st.error(f"Error loading CSV: {e}")
        return np.zeros_like(nu_target)

# -----------------------------------------------------------------------------
# IC Physics Kernel
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
        # Integrate over seed photons, then over electrons
        inner = trapezoid_rule(seed_n[None,:] * dsdE * seed_eps[None,:], seed_eps, axis=1)
        total = trapezoid_rule(ne_e * inner, e_grid)
        output.append(C * es * total)
    return np.asarray(output)

# -----------------------------------------------------------------------------
# Main UI Logic
# -----------------------------------------------------------------------------
st.sidebar.header("Controls")
seed_norm = st.sidebar.number_input("Seed Normalization", value=1e6, format="%.2e")
n_res = st.sidebar.slider("Grid Resolution", 50, 400, 150)

# Electron Inputs
st.sidebar.subheader("Electron Population")
p_idx = st.sidebar.number_input("PL Index p", value=2.5)
Te = st.sidebar.number_input("Temp T_e (K)", value=1e9)
nth = st.sidebar.number_input("Density n_th", value=1e6)

# Grids
eps_s_grid = positive_log_grid(1e-2, 1e4, n_res)
e_grid_pl = positive_log_grid(1e-1, 1e6, n_res)
e_grid_th = positive_log_grid(1e-2, 1e4, n_res)

# Compute Case 3 (MCD CSV + Power Law)
nu_grid = positive_log_grid(1e0, 1e22, n_res)
seed_fnu = get_csv_seed_fnu(nu_grid, seed_norm)
seed_eps = nu_to_eps_keV(nu_grid)
seed_n = seed_fnu / seed_eps

ne_pl = normalize_to_area(e_grid_pl, e_grid_pl**(-p_idx), nth)

with st.spinner("Calculating Emissivity..."):
    emiss = ic_emissivity(eps_s_grid, seed_eps, seed_n, e_grid_pl, ne_pl)

# Display
col1, col2, col3 = st.columns(3)
with col1:
    plot_spectrum(nu_grid, seed_fnu, "AGN Seed (from CSV)", "Frequency (Hz)", "F_nu")
with col2:
    plot_spectrum(e_grid_pl, ne_pl, "Electron Population", "E (keV)", "N(E)")
with col3:
    plot_spectrum(eps_s_grid, emiss, "IC Emissivity", "E (keV)", "J/s/keV/m3")

st.markdown("<div style='text-align: right;'><p><strong>By Garv Trivedi</strong></p></div>", unsafe_allow_html=True)
