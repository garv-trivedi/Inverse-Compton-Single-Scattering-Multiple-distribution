import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# -----------------------------------------------------------------------------
# Page setup
# -----------------------------------------------------------------------------
st.set_page_config(page_title="Inverse Compton Multi-Case Simulator", layout="wide")
st.title("Inverse Compton Up-scattering (AGN CSV Integration)")

# -----------------------------------------------------------------------------
# Constants
# -----------------------------------------------------------------------------
C = 2.99792458e8                 
H = 6.62607015e-34               
KB_SI = 1.380649e-23             
SIGMA_T = 6.6524587158e-29       
ME_C2_KEV = 510.99895            
KB_KEV_PER_K = 8.617333262145e-8 
KEV_J = 1.602176634e-16          
PI = np.pi

# -----------------------------------------------------------------------------
# Core Physics & Interpolation
# -----------------------------------------------------------------------------
def get_csv_seed_flux(nu_target, norm):
    """
    Reads the AGN CSV and performs log-log interpolation to ensure 
    a smooth, 'uniform' line in the final emissivity.
    """
    try:
        # Loading the specific file provided
        df = pd.read_csv('2026-04-09T18-00_export.csv')
        f_csv = df['frequencies (Hz)'].values
        y_csv = df['nuLnu watts'].values
        
        # Clean zeros for log-space interpolation
        mask = (f_csv > 0) & (y_csv > 0)
        f_csv, y_csv = f_csv[mask], y_csv[mask]
        
        # Convert nuLnu to Fnu (Flux density)
        fnu_csv = y_csv / f_csv
        
        # Interpolate in log-log space for numerical stability
        log_f_csv = np.log10(f_csv)
        log_fnu_csv = np.log10(fnu_csv)
        log_nu_target = np.log10(nu_target)
        
        log_fnu_interp = np.interp(log_nu_target, log_f_csv, log_fnu_csv, left=-100, right=-100)
        
        return norm * (10**log_fnu_interp)
    except Exception as e:
        st.error(f"CSV Error: {e}")
        return np.zeros_like(nu_target)

def kn_dsigma_d_es(eps_s_keV, seed_eps_keV, gamma_grid):
    """Differential KN cross-section (Isotropic)"""
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
    """Integrates over photon and electron populations"""
    gamma_grid = 1.0 + e_grid / ME_C2_KEV
    output = []
    for es in eps_s_grid:
        dsdE = kn_dsigma_d_es(es, seed_eps, gamma_grid)
        # Inner integral over seed photons
        inner = np.trapz(seed_n[None,:] * dsdE * seed_eps[None,:], seed_eps, axis=1)
        # Outer integral over electrons
        total = np.trapz(ne_e * inner, e_grid)
        output.append(C * es * total)
    return np.asarray(output)

# -----------------------------------------------------------------------------
# Sidebar & Inputs
# -----------------------------------------------------------------------------
st.sidebar.header("Simulation Settings")
seed_norm = st.sidebar.number_input("Seed Normalization", value=1e-20, format="%.2e")
n_res = st.sidebar.slider("Grid Resolution", 50, 500, 150)

st.sidebar.subheader("Electron population (Power Law)")
p_idx = st.sidebar.slider("Index (p)", 1.5, 4.0, 2.5)
e_min = st.sidebar.number_input("E_min (keV)", value=0.5)
e_max = st.sidebar.number_input("E_max (keV)", value=1e6, format="%.2e")

# -----------------------------------------------------------------------------
# Main Execution
# -----------------------------------------------------------------------------
# 1. Define Grids
# Use the range of the CSV (approx 10^0 to 10^22 Hz)
nu_grid = np.logspace(0, 22, n_res)
eps_s_out = np.logspace(-2, 4, n_res) # Output energy grid (keV)
e_grid = np.logspace(np.log10(e_min), np.log10(e_max), n_res)

# 2. Compute Components
seed_fnu = get_csv_seed_flux(nu_grid, seed_norm)
seed_eps = (H * nu_grid) / KEV_J
seed_n = seed_fnu / seed_eps # Number density shape

# Electron Power Law
ne = e_grid**(-p_idx)
ne /= np.trapz(ne, e_grid) # Normalized for shape

# 3. Calculate Emissivity
with st.spinner("Computing Inverse Compton Emissivity..."):
    emiss = ic_emissivity(eps_s_out, seed_eps, seed_n, e_grid, ne)

# -----------------------------------------------------------------------------
# Visualization
# -----------------------------------------------------------------------------
col1, col2 = st.columns(2)

with col1:
    fig1, ax1 = plt.subplots()
    ax1.loglog(nu_grid, seed_fnu, color='orange', label='CSV Seed')
    ax1.set_title("Input AGN Seed Spectrum")
    ax1.set_xlabel("Frequency (Hz)")
    ax1.set_ylabel("F_nu")
    ax1.grid(True, which="both", alpha=0.2)
    st.pyplot(fig1)

with col2:
    fig2, ax2 = plt.subplots()
    # Masking zeros to keep the line "uniform" and connected
    mask = emiss > 0
    if np.any(mask):
        ax2.loglog(eps_s_out[mask], emiss[mask], color='dodgerblue', linewidth=2)
    ax2.set_title("Resulting Emissivity (Uniform Line)")
    ax2.set_xlabel("Scattered Energy (keV)")
    ax2.set_ylabel("Emissivity")
    ax2.grid(True, which="both", alpha=0.2)
    st.pyplot(fig2)

st.info("The interpolation ensures that even if the CSV data is sparse, the emissivity remains a smooth, continuous line.")
