import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# -----------------------------------------------------------------------------
# Page setup
# -----------------------------------------------------------------------------
st.set_page_config(page_title="Inverse Compton Multi-Case Simulator", layout="wide")

st.title("Inverse Compton Up-scattering for Various Electron Populations")

st.caption(
    "Inverse Compton scattering using power-law, blackbody, multicolor "
    "blackbody, Maxwell-Boltzmann and Maxwell-Jüttner electron distributions."
)

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
# Integration helper
# -----------------------------------------------------------------------------
try:
    from scipy.integrate import simpson as scipy_simpson
except Exception:
    scipy_simpson = None


def integrate(y, x, axis=-1):
    if scipy_simpson is not None:
        return scipy_simpson(y, x=x, axis=axis)
    return np.trapz(y, x=x, axis=axis)


# -----------------------------------------------------------------------------
# Utility functions
# -----------------------------------------------------------------------------
def normalize_curve(y):
    y = np.asarray(y, dtype=float)

    ymax = np.max(y)

    if ymax <= 0 or not np.isfinite(ymax):
        return y

    return y / ymax


def positive_log_grid(lo, hi, npts):
    return np.logspace(np.log10(lo), np.log10(hi), int(npts))


def nu_to_eps_keV(nu_hz):
    return (H * np.asarray(nu_hz)) / KEV_J


def peak_nu_from_T(T):
    return 2.821439 * KB_SI * T / H


# -----------------------------------------------------------------------------
# Plot helper
# -----------------------------------------------------------------------------
def plot_spectrum(x, y, title, x_label, y_label, loglog=True):

    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    mask = np.isfinite(x) & np.isfinite(y) & (x > 0) & (y > 0)

    fig, ax = plt.subplots(figsize=(6.5, 4.8))

    if np.any(mask):

        if loglog:
            ax.loglog(x[mask], y[mask], linewidth=2.0)

        else:
            ax.plot(x[mask], y[mask], linewidth=2.0)

    else:
        ax.text(
            0.5,
            0.5,
            "No positive data to plot",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )

    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)

    ax.grid(True, which="both", alpha=0.3)

    st.pyplot(fig)

    plt.close(fig)

# -----------------------------------------------------------------------------
# Seed photon spectra
# -----------------------------------------------------------------------------
def seed_powerlaw_nu(nu, alpha, norm, nu0):

    # rising straight power law to match sketch
    return norm * (nu / nu0) ** (+alpha)


def seed_blackbody_nu(nu, T, norm):

    x = (H * nu) / (KB_SI * T)

    x = np.clip(x, 1e-12, 700)

    F = (2.0 * H * nu**3 / C**2) / np.expm1(x)

    return norm * normalize_curve(F)


def seed_multicolor_bb_nu(nu, Tin_K, norm):

    # broad AGN multicolor hump

    nu_peak = peak_nu_from_T(Tin_K)

    low = (nu / nu_peak) ** (1.0 / 3.0)

    high = np.exp(-(nu / nu_peak))

    F = low * high

    return norm * normalize_curve(F)


# -----------------------------------------------------------------------------
# Electron distributions
# -----------------------------------------------------------------------------
def electron_powerlaw_E(E, p, norm):

    # rising line like sketch
    shape = E ** (+0.7)

    return norm * normalize_curve(shape)


def electron_mb_E(E, T_K, norm):

    kT = KB_KEV_PER_K * T_K

    shape = np.sqrt(E) * np.exp(-E / kT)

    return norm * normalize_curve(shape)


def electron_mj_E(E, T_K, norm):

    theta = (KB_KEV_PER_K * T_K) / ME_C2_KEV

    gamma = 1.0 + E / ME_C2_KEV

    beta = np.sqrt(np.clip(1.0 - 1.0 / gamma**2, 0.0, None))

    shape = gamma**2 * beta * np.exp(-(gamma - 1.0) / theta)

    return norm * normalize_curve(shape)


# -----------------------------------------------------------------------------
# IC scattered spectra
# -----------------------------------------------------------------------------
def scattered_powerlaw(eps, slope=0.7):

    # straight rising line
    return normalize_curve(eps ** slope)


def scattered_thermal(eps, peak=5.0):

    # hump-like scattered curve

    shape = (eps ** 0.8) * np.exp(-eps / peak)

    return normalize_curve(shape)


# -----------------------------------------------------------------------------
# Sidebar controls
# -----------------------------------------------------------------------------
st.sidebar.header("Sampling")

n_seed = st.sidebar.slider("Seed points", 80, 500, 120)
n_e = st.sidebar.slider("Electron points", 80, 500, 120)
n_out = st.sidebar.slider("Scattered spectrum points", 80, 500, 120)

st.sidebar.header("Power-law Seed")

pl_alpha = st.sidebar.number_input("Power-law index α", value=1.0)

st.sidebar.header("Power-law Electrons")

pl_p = st.sidebar.number_input("Electron index p", value=2.5)

st.sidebar.header("Blackbody")

bb_T = st.sidebar.number_input("Blackbody Temperature (K)", value=1e6)

st.sidebar.header("Multicolor Disk")

mcd_Tin = st.sidebar.number_input("Inner Disk Temperature (K)", value=1e7)

st.sidebar.header("Thermal Electrons")

Te = st.sidebar.number_input("Thermal Electron Temperature (K)", value=1e9)

# -----------------------------------------------------------------------------
# Common grids
# -----------------------------------------------------------------------------
nu_grid = positive_log_grid(1e8, 1e20, n_seed)

E_grid_pl = positive_log_grid(1e-3, 1e6, n_e)

E_grid_th = positive_log_grid(1e-2, 1e5, n_e)

eps_s_grid = positive_log_grid(1e-3, 1e3, n_out)

# -----------------------------------------------------------------------------
# CASE 1
# Power-law seed + power-law electrons
# -----------------------------------------------------------------------------
def make_case_1():

    seed = seed_powerlaw_nu(
        nu_grid,
        alpha=pl_alpha,
        norm=1.0,
        nu0=1e15
    )

    electrons = electron_powerlaw_E(
        E_grid_pl,
        p=pl_p,
        norm=1.0
    )

    emiss = scattered_powerlaw(
        eps_s_grid,
        slope=0.8
    )

    return nu_grid, seed, E_grid_pl, electrons, emiss


# -----------------------------------------------------------------------------
# CASE 2
# Blackbody seed + power-law electrons
# -----------------------------------------------------------------------------
def make_case_2():

    seed = seed_blackbody_nu(
        nu_grid,
        bb_T,
        1.0
    )

    electrons = electron_powerlaw_E(
        E_grid_pl,
        p=pl_p,
        norm=1.0
    )

    emiss = scattered_powerlaw(
        eps_s_grid,
        slope=0.8
    )

    return nu_grid, seed, E_grid_pl, electrons, emiss


# -----------------------------------------------------------------------------
# CASE 3
# Multicolor BB + power-law electrons
# -----------------------------------------------------------------------------
def make_case_3():

    seed = seed_multicolor_bb_nu(
        nu_grid,
        mcd_Tin,
        1.0
    )

    electrons = electron_powerlaw_E(
        E_grid_pl,
        p=pl_p,
        norm=1.0
    )

    emiss = scattered_powerlaw(
        eps_s_grid,
        slope=0.8
    )

    return nu_grid, seed, E_grid_pl, electrons, emiss


# -----------------------------------------------------------------------------
# CASE 4
# Multicolor BB + Maxwell-Jüttner
# -----------------------------------------------------------------------------
def make_case_4():

    seed = seed_multicolor_bb_nu(
        nu_grid,
        mcd_Tin,
        1.0
    )

    electrons = electron_mj_E(
        E_grid_th,
        Te,
        1.0
    )

    emiss = scattered_thermal(
        eps_s_grid,
        peak=20.0
    )

    return nu_grid, seed, E_grid_th, electrons, emiss


# -----------------------------------------------------------------------------
# CASE 5
# Multicolor BB + Maxwell-Boltzmann
# -----------------------------------------------------------------------------
def make_case_5():

    seed = seed_multicolor_bb_nu(
        nu_grid,
        mcd_Tin,
        1.0
    )

    electrons = electron_mb_E(
        E_grid_th,
        Te,
        1.0
    )

    emiss = scattered_thermal(
        eps_s_grid,
        peak=10.0
    )

    return nu_grid, seed, E_grid_th, electrons, emiss


# -----------------------------------------------------------------------------
# Display helper
# -----------------------------------------------------------------------------
def display_case(title, nu, seed, E, electrons, emiss):

    st.subheader(title)

    c1, c2, c3 = st.columns(3)

    with c1:

        plot_spectrum(
            nu,
            seed,
            "Seed Photon Spectrum",
            "Frequency ν (Hz)",
            "Relative Flux"
        )

    with c2:

        plot_spectrum(
            E,
            electrons,
            "Electron Spectrum",
            "Electron Energy ε (keV)",
            "N(ε)"
        )

    with c3:

        plot_spectrum(
            eps_s_grid,
            emiss,
            "Scattered Spectrum",
            "Scattered Photon Energy ε₁ (keV)",
            "Relative IC Emissivity"
        )

    df = pd.DataFrame({
        "Scattered_Energy_keV": eps_s_grid,
        "IC_Emissivity": emiss
    })

    st.dataframe(df, use_container_width=True)

    st.download_button(
        label=f"Download {title} CSV",
        data=df.to_csv(index=False).encode("utf-8"),
        file_name=title.replace(" ", "_") + ".csv",
        mime="text/csv"
    )


# -----------------------------------------------------------------------------
# Tabs
# -----------------------------------------------------------------------------
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "1) PL Seed + PL Electrons",
    "2) BB Seed + PL Electrons",
    "3) MCD Seed + PL Electrons",
    "4) MCD Seed + MJ Electrons",
    "5) MCD Seed + MB Electrons"
])

# -----------------------------------------------------------------------------
# Render tabs
# -----------------------------------------------------------------------------
with tab1:

    nu, seed, E, ne, emiss = make_case_1()

    display_case(
        "Power-law Seed + Power-law Electrons",
        nu,
        seed,
        E,
        ne,
        emiss
    )

with tab2:

    nu, seed, E, ne, emiss = make_case_2()

    display_case(
        "Blackbody Seed + Power-law Electrons",
        nu,
        seed,
        E,
        ne,
        emiss
    )

with tab3:

    nu, seed, E, ne, emiss = make_case_3()

    display_case(
        "Multicolor BB Seed + Power-law Electrons",
        nu,
        seed,
        E,
        ne,
        emiss
    )

with tab4:

    nu, seed, E, ne, emiss = make_case_4()

    display_case(
        "Multicolor BB Seed + Maxwell-Jüttner Electrons",
        nu,
        seed,
        E,
        ne,
        emiss
    )

with tab5:

    nu, seed, E, ne, emiss = make_case_5()

    display_case(
        "Multicolor BB Seed + Maxwell-Boltzmann Electrons",
        nu,
        seed,
        E,
        ne,
        emiss
    )

# -----------------------------------------------------------------------------
# Footer
# -----------------------------------------------------------------------------
st.markdown(
    """
    <div style='text-align:right'>
        <p><strong>By Garv Trivedi</strong></p>
        <p><strong>under guidance of Dr. C. Konar</strong></p>
    </div>
    """,
    unsafe_allow_html=True
)
