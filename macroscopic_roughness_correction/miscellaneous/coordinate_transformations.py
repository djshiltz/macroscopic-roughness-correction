import numpy as np


def psi_i_e_g(i, e, g):
    """
    Computes the azimuth angle psi as a function of i, e, g
    """
    argument = (np.cos(g) - np.cos(i) * np.cos(e)) / (np.sin(i) * np.sin(e))
    rounded_argument = arccos_domain(argument)
    return np.arccos(rounded_argument)


def g_i_e_psi(i, e, psi):
    """
    Computes the phase angle g as a function of i, e, psi
    """
    argument = np.cos(i) * np.cos(e) + np.sin(i) * np.sin(e) * np.cos(psi)
    rounded_argument = arccos_domain(argument)
    return np.arccos(rounded_argument)


def psi_phi_CW(phi_CW):
    """
    Computes the azimuth angle psi from i, e, and GRIT's phi_CW
    """
    phi = phi_CW * (phi_CW < np.pi) - (2 * np.pi - phi_CW) * (phi_CW >= np.pi)
    psi = np.abs(phi)
    return psi


def arccos_domain(x):
    x = np.array(x)
    x[x < -1] = -1
    x[x > 1] = 1
    return x
