import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

try:
    from scipy.integrate import simpson as scipy_simpson
    from scipy.special import kv as besselk
    from scipy.special import gamma as sp_gamma
    from scipy.special import zeta as sp_zeta
except Exception:
    scipy_simpson = None
    besselk = None
    sp_gamma = None
    sp_zeta = None

# -----------------------------------------------------------------------------
# Constants
# -----------------------------------------------------------------------------
C = 2.99792458e8                 # speed of light (m/s)
SIGMA_T = 6.6524587158e-29       # Thomson cross-section (m^2)
ME_C2_KEV = 510.99895000         # electron rest energy (keV)
KB_KEV_PER_K = 8.617333262145e-8 # Boltzmann constant in keV/K

# Original constants used in your power-law branch
R = 10
Q = 10**5
m = 9e-31
s = 2.3676e12
h = 6.626e-34
kB_SI = 1.38e-23
PI = np.pi

# -----------------------------------------------------------------------------
# Functions 
# -----------------------------------------------------------------------------
def integrate(y, x, axis=-1):
    """SciPy Simpson if available, otherwise fallback to trapezoidal."""
    if scipy_simpson is not None:
        return scipy_simpson(y, x=x, axis=axis)
    return np.trapz(y, x=x, axis=axis)

@st.cache_data
def load_seed_spectrum(uploaded_file):
    df = pd.read_csv(uploaded_file, sep=r"\s+|,", engine="python", comment="#")
    if "Epsilon" not in df.columns or "V_Epsilon" not in df.columns:
        raise ValueError("Uploaded file must contain columns named 'Epsilon' and 'V_Epsilon'.")
    df = df[["Epsilon", "V_Epsilon"]].dropna().sort_values("Epsilon")
    df["Epsilon"] = pd.to_numeric(df["Epsilon"], errors="coerce")
    df["V_Epsilon"] = pd.to_numeric(df["V_Epsilon"], errors="coerce")
    df = df.dropna().sort_values("Epsilon")
    return df

def log_or_linear_grid(lo, hi, npts):
    lo = float(lo)
    hi = float(hi)
    npts = int(npts)
    if lo > 0 and hi > 0:
        return np.logspace(np.log10(lo), np.log10(hi), npts)
    return np.linspace(lo, hi, npts)

def findc(q_pl, B, p):
    y = 3 - p
    v = m * (C**2)
    N = (q_pl / (s * (B**2)) * (v**y)) * (y / ((Q**y) - (R**y)))
    c = N * (v**(-p))
    return c

def Const1T(p, T, q_pl, B):
    a = p + 3
    b = (p + 5) / 2
    c = findc(q_pl, B, p)
    A = ((p**2) + 4*p + 11) / ((a**2) * (2*b) * (p + 1))
    f = sp_gamma(b) * sp_zeta(b) * A
    return (8 * (PI**2) * (2.817e-15**2) * c * ((kB_SI * T)**b) * f) / ((h**3) * (C**2))

def Const1(p, q_pl, B):
    a = p + 3
    b = (p + 5) / 2
    c = findc(q_pl, B, p)
    A = ((p**2) + 4*p + 11) / ((a**2) * (2*b) * (p + 1))
    return PI * C * (2.817e-15**2) * c * A

def simpsons_one_third(E, V, p):
    n = len(E) - 1
    if n < 2:
        raise ValueError("Need at least 3 points for Simpson's rule.")
    if n % 2 != 0:
        raise ValueError("Simpson's rule requires an even number of intervals (odd number of points).")

    E = np.asarray(E, dtype=float)
    V = np.asarray(V, dtype=float)
    hstep = E[1] - E[0]
    if not np.allclose(np.diff(E), hstep, rtol=1e-5, atol=1e-8):
        raise ValueError("Epsilon values must be equally spaced for Simpson's rule.")

    transformed = V * E**((p - 1) / 2)
    result = transformed[0] + transformed[-1] + 4 * np.sum(transformed[1:n:2]) + 2 * np.sum(transformed[2:n-1:2])
    return (hstep / 3) * result

def int2(E, p, alpha):
    Ei = float(E[0])
    Ef = float(E[-1])
    expo = (p - 1) / 2
    if np.isclose(alpha, expo):
        if Ei <= 0:
            raise ValueError("Logarithm undefined for non-positive values.")
        return np.log(Ef) - np.log(Ei)
    return (Ef**(expo - alpha) / (expo - alpha)) - (Ei**(expo - alpha) / (expo - alpha))

def powerlaw_emissivity(eps_grid, p, seed_E, seed_V, q_pl, B, alpha, F, vref, theta_len_arcsec, theta_brd_arcsec, T=None):
    P1 = Const1(p, q_pl, B)
    P2 = simpsons_one_third(seed_E, seed_V, p)
    diff = int2(seed_E, p, alpha)
    o = ((F * (vref**alpha)) / (C * (h**(1 - alpha)))) * ((theta_len_arcsec * theta_brd_arcsec) / (4.254520225 * (10**10)))

    if T is None:
        constt1 = P1 * P2
    else:
        constt1 = Const1T(p, T, q_pl, B)

    constt2 = P1 * o * diff

    final = constt1 * eps_grid**(-((p - 1) / 2)) * 1.6e-16
    final2 = constt2 * eps_grid**(-((p - 1) / 2)) * 1.6e-16
    return final, final2

def maxwell_juttner_ne_gamma(gamma, nth, T):
    """
    N_MJ(gamma) = nth * gamma^2 * beta / (Theta * K2(1/Theta)) * exp(-gamma/Theta)
    """
    if besselk is None:
        raise RuntimeError("SciPy is required for the Maxwell-Jüttner branch (besselk).")
    theta = (KB_KEV_PER_K * T) / ME_C2_KEV
    if theta <= 0:
        raise ValueError("Temperature must be positive.")
    gamma = np.asarray(gamma, dtype=float)
    beta = np.sqrt(np.clip(1.0 - 1.0 / np.maximum(gamma, 1.0)**2, 0.0, None))
    norm = nth / (theta * besselk(2, 1.0 / theta))
    return norm * gamma**2 * beta * np.exp(-gamma / theta)

def maxwell_boltzmann_ne_gamma(gamma, nth, T):
    gamma = np.asarray(gamma, dtype=float)
    kT_keV = KB_KEV_PER_K * T
    if kT_keV <= 0:
        raise ValueError("Temperature must be positive.")
    # E = (gamma - 1) * m_e * c^2 [cite: 3, 21]
    E_keV = np.maximum((gamma - 1.0) * ME_C2_KEV, 0.0)
    # N_MB(E) distribution [cite: 3, 19]
    n_E = nth * (2.0 / (np.sqrt(np.pi) * (kT_keV**1.5))) * np.sqrt(E_keV) * np.exp(-E_keV / kT_keV)
    # Return N(gamma) = N(E) * (dE/dgamma)
    return n_E * ME_C2_KEV

def kn_dsigma_d_es(eps_s, seed_eps, gamma_grid):
    """
    Klein-Nishina differential cross section used in the isotropic IC kernel.
    All energies are treated consistently in keV.
    """
    gamma = np.asarray(gamma_grid, dtype=float)[:, None]
    eps = np.asarray(seed_eps, dtype=float)[None, :]

    eps_s = float(eps_s)
    if eps_s <= 0:
        return np.zeros((gamma_grid.size, seed_eps.size))

    Gamma_e = 4.0 * gamma * eps / ME_C2_KEV
    denom = Gamma_e * (gamma * ME_C2_KEV - eps_s)

    valid = (Gamma_e > 0) & (denom > 0) & (eps > 0) & (eps_s < gamma * ME_C2_KEV)
    q = np.where(valid, eps_s / denom, 0.0)
    valid = valid & (q >= 1.0 / (4.0 * gamma**2)) & (q <= 1.0)

    q_safe = np.clip(q, 1e-300, 1.0)
    F = (
        2.0 * q_safe * np.log(q_safe)
        + (1.0 + 2.0 * q_safe) * (1.0 - q_safe)
        + 0.5 * (Gamma_e * q_safe)**2 * (1.0 - q_safe) / (1.0 + Gamma_e * q_safe)
    )

    eps_safe = np.where(eps > 0, eps, 1.0)
    dsdE = (3.0 * SIGMA_T / (4.0 * gamma**2 * eps_safe)) * F
    return np.where(valid, dsdE, 0.0)

def thermal_ic_emissivity(eps_s_grid, seed_eps, seed_n, gamma_grid, ne_gamma):
    """
    j_IC(eps_s) = c * eps_s * ∫ dγ Ne(γ) ∫ dε n_ph(ε) [dσ_IC/dε_s]
    """
    seed_eps = np.asarray(seed_eps, dtype=float)
    seed_n = np.asarray(seed_n, dtype=float)
    gamma_grid = np.asarray(gamma_grid, dtype=float)
    ne_gamma = np.asarray(ne_gamma, dtype=float)

    output = []
    for eps_s in eps_s_grid:
        dsdE = kn_dsigma_d_es(eps_s, seed_eps, gamma_grid)
        inner_seed = integrate(seed_n[None, :] * dsdE * seed_eps[None, :], x=seed_eps, axis=1)
        total = integrate(ne_gamma * inner_seed, x=gamma_grid)
        output.append(C * eps_s * total)
    return np.asarray(output)

# -----------------------------------------------------------------------------
# Plotting
# -----------------------------------------------------------------------------
def plot_spectrum(x, y, title, x_label="Epsilon", y_label="Volume Emissivity"):
    fig, ax = plt.subplots(figsize=(10, 6))
    
    x = np.asarray(x)
    y = np.asarray(y)

    # Filter out non-positive values for log plotting
    mask = (x > 0) & (y > 0)
    if not np.any(mask):
        st.warning(f"No positive data to plot for {title}")
        return

    ax.plot(x[mask], y[mask], marker='o', markersize=2, linestyle='-')
    
    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)

    ax.set_xscale("log")
    ax.set_yscale("log")
    
    # Dynamic limits: set bottom at 8 orders of magnitude below peak
    y_max = np.max(y[mask])
    ax.set_ylim(y_max * 1e-8, y_max * 2)

    ax.grid(True, which="both", alpha=0.3)
    st.pyplot(fig)
# -----------------------------------------------------------------------------
# Streamlit UI
# -----------------------------------------------------------------------------
st.title("Inverse Compton Spectra for Single Scattering")

# -----------------------------
# Sidebar inputs
# -----------------------------
st.sidebar.header("Seed photon spectrum")

uploaded_file = st.sidebar.file_uploader(
    "Upload a whitespace/comma separated txt file",
    type=["txt", "csv"]
)

st.sidebar.write("File must contain columns named `Epsilon` and `V_Epsilon`.")

# -----------------------------
# Data source selection
# -----------------------------
use_sample = st.sidebar.checkbox("Use Sample.txt (default)", value=True)

if use_sample:
    try:
        seed_df = load_seed_spectrum("Sample.txt")
        st.success("Using Sample.txt as default seed photon spectrum")
    except Exception as e:
        st.error(f"Failed to load Sample.txt: {e}")
        st.stop()

else:
    if uploaded_file is None:
        st.warning("Please upload a file or enable 'Use Sample.txt'")
        st.stop()
    
    seed_df = load_seed_spectrum(uploaded_file)
    st.success(f"Using uploaded file: {uploaded_file.name}")
seed_E = seed_df["Epsilon"].to_numpy(dtype=float)
seed_V = seed_df["V_Epsilon"].to_numpy(dtype=float)

st.sidebar.header("Output grid")
lower_E = st.sidebar.number_input("Lower limit of epsilon", value=1e-2, min_value=1e-12, format="%.2e")
upper_E = st.sidebar.number_input("Upper limit of epsilon", value=1e3, min_value=1e-12, format="%.2e")
n_out = st.sidebar.number_input("Number of output points", value=378, min_value=3, max_value=2000)

eps_grid = np.logspace(np.log10(lower_E), np.log10(upper_E), n_out)

st.sidebar.header("General thermal-grid settings")
gamma_min = st.sidebar.number_input("Gamma min", value=1.0, min_value=1.0)
gamma_max = st.sidebar.number_input("Gamma max", value=5.0, min_value=1.0)
n_gamma = st.sidebar.number_input("Gamma grid points", value=300, min_value=20, max_value=5000)

# -----------------------------
# Power-law parameters
# -----------------------------
st.sidebar.header("Power-law inputs")
p = st.sidebar.number_input("Electron index p", value=2.5)
q_pl = st.sidebar.number_input("Normalization parameter q", value=2.72)
B = st.sidebar.number_input("Magnetic field B", value=2.72)
alpha = st.sidebar.number_input("Seed photon index alpha", value=2.72)
F = st.sidebar.number_input("Flux density F", value=2.72)
vref = st.sidebar.number_input("Reference frequency v", value=2.72)
theta_len = st.sidebar.number_input("Theta length (arcsec)", value=1.0)
theta_brd = st.sidebar.number_input("Theta breadth (arcsec)", value=1.0)
T_blackbody = st.sidebar.number_input(
    "Temperature T for blackbody shortcut (set 0 to ignore)",
    value=0.0,
    min_value=0.0
)

# -----------------------------
# Thermal parameters
# -----------------------------
st.sidebar.header("Thermal electron inputs")
nth = st.sidebar.number_input("Thermal electron density n_th", value=1.0e6, min_value=0.0)
T_thermal = st.sidebar.number_input("Thermal temperature T (K)", value=1.0e8, min_value=0.0)

# Helper for display tables
def make_display_df(x, y):
    return pd.DataFrame({
        "Epsilon": [f"{val:.6e}" for val in np.asarray(x)],
        "Volume Emissivity": [f"{val:.6e}" for val in np.asarray(y)],
    })

# -----------------------------
# Tabs
# -----------------------------
tab_powerlaw, tab_mj, tab_mb = st.tabs(
    ["Power law", "Maxwell-Jüttner", "Maxwell-Boltzmann"]
)

# =============================================================================
# Power-law 
# =============================================================================
with tab_powerlaw:
    st.header("Power Law Distribution")

    final, final2 = powerlaw_emissivity(
        eps_grid, p, seed_E, seed_V,
        q_pl, B, alpha, F, vref,
        theta_len, theta_brd,
        T=T_blackbody if T_blackbody > 0 else None
    )

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Power-law seed integral")
        data2 = make_display_df(eps_grid, final2)
        st.dataframe(data2, use_container_width=True)
        plot_spectrum(eps_grid, final2, "Inverse Compton Result (Power Law)")

    with col2:
        if T_blackbody > 0:
            st.subheader("Blackbody shortcut")
            data = make_display_df(eps_grid, final)
            st.dataframe(data, use_container_width=True)
            plot_spectrum(eps_grid, final, "Inverse Compton Result (Blackbody shortcut)")
        else:
            st.subheader("Blackbody shortcut")
            st.write("Set a positive temperature in the sidebar to display this branch.")

    st.latex(r"\frac{dE}{dVdt d\epsilon_1} = \pi c r_0^2 C A(p)\epsilon_1^{-(p-1)/2}\int d\epsilon\, \epsilon^{(p-1)/2} v(\epsilon)")
    st.latex(r"A(p) = 2^{p+3}\frac{p^2+4p+11}{(p+3)^2(p+5)(p+1)}")
    st.latex(r"C = N_0 (m_e c^2)^{-p}")

# =============================================================================
# Maxwell-Juttner 
# =============================================================================
with tab_mj:
    st.header("Maxwell-Jüttner Distribution")
    if besselk is None:
        st.error("SciPy required.")
    else:
        # Calculate scale
        theta = (KB_KEV_PER_K * T_thermal) / ME_C2_KEV
        
        # Zoom grid: MJ peaks near 1 + theta. We cover the core of the distribution.
        # For 10^8 K, theta is ~0.016, so grid is ~1.0001 to 1.4
        g_min = 1.0 + (1e-4 * theta) 
        g_max_thermal = 1.0 + (25.0 * theta)
        
        gamma_grid = np.logspace(np.log10(g_min), np.log10(g_max_thermal), int(n_gamma))

        ne_gamma = maxwell_juttner_ne_gamma(gamma_grid, nth, T_thermal)
        emissivity = thermal_ic_emissivity(eps_grid, seed_E, seed_V, gamma_grid, ne_gamma)

        st.dataframe(make_display_df(eps_grid, emissivity), use_container_width=True)
        plot_spectrum(eps_grid, emissivity, "Inverse Compton Result (Maxwell-Jüttner)")
# =============================================================================
# Maxwell-Boltzmann 
# =============================================================================
with tab_mb:
    st.header("Maxwell-Boltzmann Distribution")
    
    kT_keV = KB_KEV_PER_K * T_thermal
    
    # Peak is at 0.5 * kT. We sample up to 15 * kT to capture the tail.
    E_min = 1e-3 * kT_keV
    E_max = 15.0 * kT_keV
    
    energy_grid = np.linspace(E_min, E_max, int(n_gamma))
    # gamma = 1 + E / (mc^2)
    gamma_grid = 1.0 + (energy_grid / ME_C2_KEV)

    ne_gamma = maxwell_boltzmann_ne_gamma(gamma_grid, nth, T_thermal)
    emissivity = thermal_ic_emissivity(eps_grid, seed_E, seed_V, gamma_grid, ne_gamma)

    st.dataframe(make_display_df(eps_grid, emissivity), use_container_width=True)
    plot_spectrum(eps_grid, emissivity, "Inverse Compton Result (Maxwell-Boltzmann)")
# -----------------------------------------------------------------------------
# Footer
# -----------------------------------------------------------------------------
st.markdown(
    """
    <div style='text-align: right;'>
        <p><strong>By Garv Trivedi</strong></p>
        <p><strong>under guidance of Dr. C. Konar</strong></p>
    </div>
    """,
    unsafe_allow_html=True
)


