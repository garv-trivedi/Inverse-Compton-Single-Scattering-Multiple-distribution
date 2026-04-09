import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# -----------------------------------------------------------------------------
# Page setup
# -----------------------------------------------------------------------------
st.set_page_config(page_title="Inverse Compton Multi-Case Simulator", layout="wide")
st.title("Inverse Compton Up-scattering (AGN CSV Data)")

st.caption(
    "Upload your AGN spectrum CSV to generate multi-colour blackbody seed cases. "
    "Calculations use log-log interpolation for smooth, uniform results."
)

# -----------------------------------------------------------------------------
# Constants & Compatibility
# -----------------------------------------------------------------------------
C = 2.99792458e8                 
H = 6.62607015e-34               
KB_SI = 1.380649e-23             
SIGMA_T = 6.6524587158e-29       
ME_C2_KEV = 510.99895            
KB_KEV_PER_K = 8.617333262145e-8 
KEV_J = 1.602176634e-16          
PI = np.pi

def integrate_safe(y, x, axis=-1):
    """Handles NumPy 2.0 rename of trapz to trapezoid."""
    if hasattr(np, 'trapezoid'):
        return np.trapezoid(y, x=x, axis=axis)
    return np.trapz(y, x=x, axis=axis)

# -----------------------------------------------------------------------------
# File Upload Logic
# -----------------------------------------------------------------------------
st.sidebar.header("Data Input")
uploaded_file = st.sidebar.file_uploader("Upload '2026-04-09T18-00_export.csv'", type="csv")

# -----------------------------------------------------------------------------
# Seed Spectrum & Interpolation
# -----------------------------------------------------------------------------
def get_seed_from_csv(nu_grid, norm, file_obj):
    if file_obj is None:
        return np.zeros_like(nu_grid)
    
    df = pd.read_csv(file_obj)
    f_csv = df['frequencies (Hz)'].values
    y_csv = df['nuLnu watts'].values
    
    # Clean and convert nuLnu to Fnu
    mask = (f_csv > 0) & (y_csv > 0)
    f_csv, fnu_csv = f_csv[mask], y_csv[mask] / f_csv[mask]
    
    # Log-log interpolation ensures a smooth, non-broken line
    log_fnu = np.interp(np.log10(nu_grid), np.log10(f_csv), np.log10(fnu_csv), left=-100, right=-100)
    return norm * (10**log_fnu)

# -----------------------------------------------------------------------------
# Physics Kernel
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
        inner = integrate_safe(seed_n[None,:] * dsdE * seed_eps[None,:], seed_eps, axis=1)
        total = integrate_safe(ne_e * inner, e_grid)
        output.append(C * es * total)
    return np.asarray(output)

# -----------------------------------------------------------------------------
# Main Dashboard
# -----------------------------------------------------------------------------
st.sidebar.header("Parameters")
seed_amp = st.sidebar.number_input("Normalization", value=1e6, format="%.2e")
n_res = st.sidebar.slider("Resolution", 50, 400, 120)
p_idx = st.sidebar.slider("Electron Index p", 1.5, 4.0, 2.5)

if uploaded_file is not None:
    # 1. Grids
    nu_grid = np.logspace(0, 22, n_res)
    e_grid = np.logspace(-1, 6, n_res)
    out_eps = np.logspace(-2, 4, n_res)
    
    # 2. Populations
    seed_fnu = get_seed_from_csv(nu_grid, seed_amp, uploaded_file)
    seed_eps = (H * nu_grid) / KEV_J
    seed_n = seed_fnu / seed_eps
    ne = e_grid**(-p_idx)
    ne /= integrate_safe(ne, e_grid) # Normalized
    
    # 3. Calculation
    with st.spinner("Processing..."):
        emiss = ic_emissivity(out_eps, seed_eps, seed_n, e_grid, ne)
    
    # 4. Plotting
    c1, c2, c3 = st.columns(3)
    
    def plot_log(x, y, title, xl, yl, col):
        mask = (x > 0) & (y > 0)
        fig, ax = plt.subplots()
        ax.loglog(x[mask], y[mask], linewidth=2)
        ax.set_title(title); ax.set_xlabel(xl); ax.set_ylabel(yl)
        col.pyplot(fig)

    plot_log(nu_grid, seed_fnu, "Seed (CSV)", "Hz", "F_nu", c1)
    plot_log(e_grid, ne, "Electrons", "keV", "N(E)", c2)
    plot_log(out_eps, emiss, "Emissivity (Uniform)", "keV", "J/s/keV/m3", c3)
else:
    st.warning("Please upload '2026-04-09T18-00_export.csv' in the sidebar to start.")

st.markdown("<div style='text-align: right;'><p><strong>By Garv Trivedi</strong></p></div>", unsafe_allow_html=True)
