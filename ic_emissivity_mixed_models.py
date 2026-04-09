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

#
