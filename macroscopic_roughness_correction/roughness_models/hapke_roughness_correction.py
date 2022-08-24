import numpy as np
from macroscopic_roughness_correction.data.define_smooth_reflectance import smooth_reflectance
from macroscopic_roughness_correction.miscellaneous.coordinate_transformations import g_i_e_psi
from alive_progress import alive_bar
from define_root_directory import ROOT_DIR


albedo = np.load(f'{ROOT_DIR}/macroscopic_roughness_correction/data/albedo.npy', allow_pickle=True).item()

def r_hapke(i, e, psi, M, mineral, multifacet_correction=False):
    """
    Computes Hapke's roughness correction

    Parameters
    ----------
    i : 1d numpy array
        zenith angle of incident light [rad]
    e : 1d numpy array
        zenith angle of exiting light [rad]
    psi : 1d numpy array
        azimuth angle between incident and exiting light [rad]
    M : float
        RMS slope of the surface (unitless)
    mineral : str
        'olivine' or 'quartz'
    multifacet_correction: bool
        whether to use Hapke's proposed interfacet scattering correction

    Returns
    -------
    r : 1d numpy array
        rough surface bidirectional reflectance
    """

    M = np.expand_dims(np.atleast_1d(M), -1)
    if multifacet_correction:
        print('Computing Hapke Roughness Correction (single and multi facet)')
        r0 = albedo[mineral]['r0']
        theta_bar = np.arctan(np.sqrt(2/np.pi) * M)
        theta_bar_p = (1 - r0) * theta_bar
        M_p = np.sqrt(np.pi/2) * np.tan(theta_bar_p)

        # set M equal to M_p for multifacet scattering approximation
        M = M_p
    else:
        print('Computing Hapke Roughness Correction (single facet)')

    # make 1D arrays the same length
    n_points = (i * e * psi).shape[0]
    i = np.ones(n_points) * i
    e = np.ones(n_points) * e
    psi = np.ones(n_points) * psi
    M = (M.T * np.ones(n_points)).T

    # compute correction 1 point at a time
    result = []
    with alive_bar(n_points) as bar:
        for n in range(n_points):
            # effective angles and shadowing term
            i_e, e_e, S = hapke_roughness(i=i[n], e=e[n], psi=psi[n], M=M[n, :])

            # phase angle
            g = g_i_e_psi(i=i[n], e=e[n], psi=psi[n])

            # bidirectional reflectance at effective angles
            r_eff = smooth_reflectance(i=i_e, e=e_e, g=g, mineral=mineral)

            # reshape to length 2151 array
            if multifacet_correction:
                r_eff = np.diagonal(r_eff)
                rough_reflectance = r_eff * S
            else:
                rough_reflectance = r_eff * S
                rough_reflectance = np.squeeze(rough_reflectance)

            result.append(rough_reflectance)
            bar()

    return np.array(result)


def hapke_roughness(i, e, psi, M):
    """
    Computes the effective incident and exit directions, as well as the shadowing term,
    for Hapke's macroscopic roughness correction

    Parameters
    ----------
    i : 1d numpy array
        zenith angle of incident light [rad]
    e : 1d numpy array
        zenith angle of exiting light [rad]
    psi : 1d numpy array
        azimuth angle between incident and exiting light [rad]
    M : float
        RMS slope of the surface (unitless)
    w : float or None
        single scattering albedo, between 0 and 1
        only used by this function if use_empirical_correction is True
    use_empirical_correction : bool
        whether or not to use Hapke's empirical correction to the roughness model,
        in which the roughness parameter is modified by w to account
        for interfacet scattering

    Returns
    -------
    i_e : 1d numpy array
        effective incidence angle
    e_e : 1d numpy array
        effective exit angle
    S : 1d numpy array
        shadowing factor
    """

    # Determine theta_bar from RMS slope m
    theta_bar = np.arctan(M * np.sqrt(2 / np.pi))

    mu_e = mu_e__i_lessthan_e(i, e, psi, theta_bar) * (i < e) \
         + mu_e__e_lessthan_i(i, e, psi, theta_bar) * (e <= i)

    mu_0e = mu_0e__i_lessthan_e(i, e, psi, theta_bar) * (i < e) \
          + mu_0e__e_lessthan_i(i, e, psi, theta_bar) * (e <= i)

    # 12.12a
    e_e = np.arccos(mu_e)

    # 12.12b
    i_e = np.arccos(mu_0e)

    S = S__i_lessthan_e(i, e, psi, theta_bar) * (i < e) \
      + S__e_lessthan_i(i, e, psi, theta_bar) * (e <= i)

    return i_e, e_e, S


# 12.45a
def chi(theta_bar):
    return 1 / np.sqrt(1 + np.pi * np.tan(theta_bar)**2)


# 12.45b
def E1(x, theta_bar):
    return np.exp(-2 / np.pi * cot(theta_bar) * cot(x))


# 12.45c
def E2(x, theta_bar):
    return np.exp(-1/np.pi * cot(theta_bar)**2 * cot(x)**2)


# 12.46
def mu_0e__i_lessthan_e(i, e, psi, theta_bar):
    return chi(theta_bar) * (np.cos(i) + np.sin(i) * np.tan(theta_bar) \
               * (np.cos(psi) * E2(e, theta_bar) + np.sin(psi/2)**2 * E2(i, theta_bar)) \
               / (2 - E1(e, theta_bar) - (psi/np.pi) * E1(i, theta_bar)))

# 12.47
def mu_e__i_lessthan_e(i, e, psi, theta_bar):
    return chi(theta_bar) * (np.cos(e) + np.sin(e) * np.tan(theta_bar) \
               * (E2(e, theta_bar) - np.sin(psi/2)**2 * E2(i, theta_bar)) \
               / (2 - E1(e, theta_bar) - (psi/np.pi) * E1(i, theta_bar)))


# 12.48
def eta_e(e, theta_bar):
    return chi(theta_bar) * (np.cos(e) + np.sin(e) * np.tan(theta_bar) \
                                 * E2(e, theta_bar) / (2 - E1(e, theta_bar)))

# 12.49
def eta_0e(i, theta_bar):
    return chi(theta_bar) * (np.cos(i) + np.sin(i) * np.tan(theta_bar) \
                                 * E2(i, theta_bar) / (2 - E1(i, theta_bar)))


# 12.50
def S__i_lessthan_e(i, e, psi, theta_bar):
    mu_e = mu_e__i_lessthan_e(i, e, psi, theta_bar)
    mu_0 = np.cos(i)
    return mu_e/eta_e(e, theta_bar) * mu_0/eta_0e(i, theta_bar) * chi(theta_bar) \
               / (1 - f(psi) + f(psi) * chi(theta_bar) * mu_0 / eta_0e(i, theta_bar))


# 12.51
def f(psi):
    return np.exp(-2 * np.tan(psi/2))


# 12.52
def mu_0e__e_lessthan_i(i, e, psi, theta_bar):
    return chi(theta_bar) * (np.cos(i) + np.sin(i) * np.tan(theta_bar) \
               * (E2(i, theta_bar) - np.sin(psi/2)**2 * E2(e, theta_bar)) \
               / (2 - E1(i, theta_bar) - (psi/np.pi) * E1(e, theta_bar)))


# 12.53
def mu_e__e_lessthan_i(i, e, psi, theta_bar):
    return chi(theta_bar) * (np.cos(e) + np.sin(e) * np.tan(theta_bar) \
               * (np.cos(psi) * E2(i, theta_bar) + np.sin(psi/2)**2 * E2(e, theta_bar)) \
               / (2 - E1(i, theta_bar) - (psi/np.pi) * E1(e, theta_bar)))

# 12.54
def S__e_lessthan_i(i, e, psi, theta_bar):
    mu_e = mu_e__e_lessthan_i(i, e, psi, theta_bar)
    mu_0 = np.cos(i)
    mu = np.cos(e)
    return mu_e/eta_e(e, theta_bar) * mu_0/eta_0e(i, theta_bar) * chi(theta_bar) \
               / (1 - f(psi) + f(psi) * chi(theta_bar) * mu / eta_e(e, theta_bar))


def cot(x):
    valid_angles = np.abs(x) > 0
    result = np.zeros_like(x)
    result[valid_angles] = 1/np.tan(x[valid_angles])
    result[np.logical_not(valid_angles)] = np.nan
    return result
