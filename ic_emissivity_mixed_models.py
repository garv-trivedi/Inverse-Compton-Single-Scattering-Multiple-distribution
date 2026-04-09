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
def kn_dsigma_d_es(eps_s_keV, seed_eps_keV,
