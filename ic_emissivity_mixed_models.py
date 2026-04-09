import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# -----------------------------------------------------------------------------
# 1. Setup and Compatibility Wrapper
# -----------------------------------------------------------------------------
st.set_page_config(page_title="Inverse Compton Multi-Case Simulator", layout="wide")
st.title("Inverse Compton Up-scattering Dashboard")

# Fix for NumPy 2.0 removing np.trapz
def trap_int(y, x, axis=-1):
    if hasattr(np, 'trapezoid'):
        return np.trapezoid(y, x=x, axis=axis)
    return np.trapz(y, x=x, axis=axis)

# -----------------------------------------------------------------------------
# 2. Constants
# -----------------------------------------------------------------------------
C = 2.99792458e8                 
H = 6.62607015e-34               
KB_SI = 1.380649e-23             
SIGMA_T = 6.6524587158e-29       
ME_C2_KEV = 510.99895            
KEV_J = 1.602176634e-16          

# -----------------------------------------------------------------------------
# 3. Sidebar Inputs
# -----------------------------------------------------------------------------
st.sidebar.header("Data Input")
uploaded_file = st.sidebar.file_uploader("Upload '2026-04-09T18-00_export.csv'", type="csv")

st.sidebar.header("Simulation Parameters")
seed_norm = st.sidebar.number_input("Seed Amplitude (Scale)", value=1.0, format="%.2e")
n_pts = st.sidebar.slider("Resolution (Points)", 50, 300, 150)

st.sidebar.subheader("Electron population")
p_idx = st.sidebar.number_input("PL Index p", value=2.5, step=0.1)
nth = st.sidebar.number_input("Density n_th", value=1e6, format="%.2e")

# -----------------------------------------------------------------------------
# 4. Physics Functions
# -----------------------------------------------------------------------------
def get_csv_shape(nu_grid, file):
    """Interpolates CSV in log-log space for a uniform, accurate seed."""
    df = pd.read_csv(file)
    f_csv = df['frequencies (Hz)'].values
    y_csv = df['nuLnu watts'].values
    
    # Clean data
    mask = (f_csv > 0) & (y_csv > 0)
    f_csv, y_csv = f_csv[mask], y_csv[mask]
    
    # We interpolate nuLnu directly for better shape preservation
    log_y = np.interp(np.log10(nu_grid), np.log10(f_csv), np.log10(y_csv), 
                      left=-50, right=-50) # Use floor of -50 to avoid 0 errors
    nuLnu = 10**log_y
    return nuLnu / nu_grid # Returns F_nu shape

def kn_kernel(eps_s, eps, gamma):
    """Full Klein-Nishina cross-section kernel."""
    Gamma_e = 4.0 * gamma * eps / ME_C2_KEV
    denom = Gamma_e * (gamma * ME_C2_KEV - eps_s)
    
    # Safety masks for valid scattering regions
    valid = (Gamma_e > 0) & (denom > 0) & (eps_s < gamma * ME_C2_KEV)
    q = np.where(valid, eps_s / denom, 0.0)
    valid = valid & (q >= 1.0/(4.0*gamma**2)) & (q <= 1.0)
    
    q_safe = np.clip(q, 1e-30, 1.0)
    F = 2*q_safe*np.log(q_safe) + (1+2*q_safe)*(1-q_safe) + 0.5*(Gamma_e*q_safe)**2*(1-q_safe)/(1+Gamma_e*q_safe)
    return np.where(valid, (3.0 * SIGMA_T / (4.0 * gamma**2 * eps)) * F, 0.0)

# -----------------------------------------------------------------------------
# 5. Execution Logic
# -----------------------------------------------------------------------------
if uploaded_file is not None:
    # 1. Establish Grids based on CSV data
    # Broad frequency grid to catch the full AGN spectrum
    nu_grid = np.logspace(0, 22, n_pts) 
    seed_fnu = get_csv_shape(nu_grid, uploaded_file) * seed_norm
    seed_eps = (H * nu_grid) / KEV_J
    seed_n = seed_fnu / seed_eps # Number density
    
    # Electron Grid (Power Law)
    e_grid = np.logspace(-1, 7, n_pts)
    ne = e_grid**(-p_idx)
    # Normalize electron density
    area = trap_int(ne, e_grid)
    ne = (ne / area) * nth
    
    # Scattered Grid (Output)
    eps_out = np.logspace(-3, 6, n_pts)
    
    # 2. Calculate Emissivity
    emiss = []
    gamma_grid = 1.0 + e_grid / ME_C2_KEV
    
    with st.spinner("Calculating smooth IC emissivity..."):
        for es in eps_out:
            # Kernel for this output energy vs all seed photons and all electrons
            dsdE = kn_kernel(es, seed_eps, gamma_grid[:, None])
            # Integrate over seed photons (inner)
            inner = trap_int(seed_n * dsdE * seed_eps, seed_eps, axis=1)
            # Integrate over electrons (outer)
            total = trap_int(ne * inner, e_grid)
            emiss.append(C * es * total)
    
    emiss = np.array(emiss)

    # 3. Plotting
    c1, c2, c3 = st.columns(3)
    
    def plot_spec(x, y, title, xl, yl, col, color):
        mask = (x > 0) & (y > 1e-40) # Mask deep zeros for log plot
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.loglog(x[mask], y[mask], color=color, linewidth=2)
        ax.set_title(title); ax.set_xlabel(xl); ax.set_ylabel(yl)
        ax.grid(True, which="both", alpha=0.2)
        col.pyplot(fig)

    plot_spec(nu_grid, seed_fnu * nu_grid, "Seed (nuLnu)", "Hz", "Watts", c1, "red")
    plot_spec(e_grid, ne, "Electrons N(E)", "keV", "Density", c2, "green")
    plot_spec(eps_out, emiss, "IC Emissivity", "keV", "J/s/keV/m3", c3, "blue")

else:
    st.info("Please upload your CSV file in the sidebar to generate the accurate spectrum.")

st.markdown("<div style='text-align: right;'><p><strong>By Garv Trivedi</strong></p></div>", unsafe_allow_html=True)
