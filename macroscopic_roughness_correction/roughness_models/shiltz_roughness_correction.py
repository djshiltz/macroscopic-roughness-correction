import numpy as np
from scipy.integrate import trapz
from macroscopic_roughness_correction.roughness_models.heitz_2013_corrected import nu_AB_i_e_M, p_not_proj
from macroscopic_roughness_correction.miscellaneous.coordinate_transformations import g_i_e_psi, psi_i_e_g
from macroscopic_roughness_correction.data.define_smooth_reflectance import smooth_reflectance
from alive_progress import alive_bar
from macroscopic_roughness_correction.processing.fitting_parameters import c_L, c_NL
from define_root_directory import ROOT_DIR


# Load diffusive reflectance
albedo = np.load(f'{ROOT_DIR}/macroscopic_roughness_correction/data/albedo.npy', allow_pickle=True).item()
r0_olivine = albedo['olivine']['r0']
r0_quartz = albedo['quartz']['r0']


def r_shiltz(i, e, psi, M, mineral, c_L=c_L, c_NL=c_NL, N=100, stdevs=5):
    '''
    Computes the roughness correction using corrected slope distribution function

    Parameters
    ----------
    i : 1d numpy array
        zenith angle of incident light [rad]
    e : 1d numpy array
        zenith angle of exiting light [rad]
    psi : 1d numpy array
        azimuth angle between incident and exiting light [rad]
    M : 1d numpy array
        RMS slope of the surface (unitless)
    mineral : str
        'quartz' or 'olivine'
    c_L : float
        empirical constant for Lambertian multi-facet scattering
    c_NL : float
        empirical constant for non-Lambertian multi-facet scattering
    N : int
        number of points in each dimension (m_x, m_y) for integration
    stdevs : int
        number of standard deviations (in each direction) over which to integrate normal distribution

    Returns:
    --------
    r_single : 2d numpy array
        single facet reflectance only
    r_total_Lambertian : 2d numpy array
        single plus multiple facet reflectance, Lambertian multi-facet scattering approximation
    r_total_nonLambertian : 2d numpy array
        single plus multiple facet reflectance, non-Lambertian multi-facet scattering approximation
    '''

    # Make all arrays the same length
    num_points = int(np.max([np.size(i), np.size(e), np.size(psi), np.size(M)]).astype(np.int32))
    i_list = np.ones(num_points) * i
    e_list = np.ones(num_points) * e
    psi_list = np.ones(num_points) * psi
    M_list = np.ones(num_points) * M
    g_list = g_i_e_psi(i=i, e=e, psi=psi)
    del i, e, psi, M  # delete the original inputs to avoid confusion

    # Check for opposition direction
    if np.any((i_list == e_list) * (psi_list == 0)):
        raise ValueError("Cannot compute reflectance in the exact opposition direction")

    # Compute diffusive reflectance
    r0 = r0_olivine * (mineral == 'olivine') + r0_quartz * (mineral == 'quartz')

    # Compute rough reflectance at each point
    r_s = []
    r_m_Lambertian = []
    r_m_nonLambertian = []
    print('Computing Shiltz Roughness Correction')
    with alive_bar(num_points) as bar:
        for point_index in range(num_points):
            # Parameters at this particular point
            i = i_list[point_index]
            e = e_list[point_index]
            g = g_list[point_index]
            psi = psi_list[point_index]
            M = M_list[point_index]

            # Compute probability density function
            m_x_array = np.linspace(-stdevs * M, stdevs * M, N + 1)
            m_y_array = np.linspace(-stdevs * M, stdevs * M, N)
            m_x, m_y = np.meshgrid(m_x_array, m_y_array, indexing='ij')
            f_mm = 1 / (2 * np.pi * M ** 2) * np.exp(-1 / (2 * M ** 2) * (m_x ** 2 + m_y ** 2))

            # Check area under probability density function
            area = int2d(f=f_mm, u=m_x, v=m_y, u_axis=0, v_axis=1)
            if area < 0.999:
                raise ValueError('Area under pdf is significantly less than 1')

            # Compute angles
            phi_i = 0
            phi_e = psi
            m_i = np.cos(phi_i) * m_x + np.sin(phi_i) * m_y
            m_e = np.cos(phi_e) * m_x + np.sin(phi_e) * m_y
            cos_theta = 1 / np.sqrt(m_x**2 + m_y**2 + 1)
            iota = np.arccos((np.cos(i) - m_i * np.sin(i)) * cos_theta)
            epsilon = np.arccos((np.cos(e) - m_e * np.sin(e)) * cos_theta)

            # relative azimuth in tilted coordinate frame (reference only)
            Psi = psi_i_e_g(i=iota, e=epsilon, g=g)

            # Smooth surface reflectance
            r = smooth_reflectance(i=iota, e=epsilon, g=g, mineral=mineral)

            # Compute integral
            integrand = ((step(cot(i) - m_i) * step(cot(e) - m_e) * (1 - m_e * np.tan(e)) * f_mm).T * r.T).T
            m_x = np.expand_dims(m_x, axis=-1)
            m_y = np.expand_dims(m_y, axis=-1)
            integral = int2d(f=integrand, u=m_x, v=m_y, u_axis=0, v_axis=1)

            # Projected shadows
            nu_A, nu_B = nu_AB_i_e_M(i=i, e=e, M=M)
            p_not_proj_given_not_tilt = p_not_proj(nu_A=nu_A, nu_B=nu_B, phi=psi)
            r_s.append(p_not_proj_given_not_tilt * integral)

            # Multi-facet reflectance
            r_m_Lambertian.append(c_L * M * r0 * np.cos(i) / np.pi)
            r_m_nonLambertian.append(c_L * M * r0 * np.cos(i) / np.pi \
                * (1 + c_NL * np.exp(-4/np.pi * (np.pi - g)**2)))
            bar()
    r_s = np.array(r_s)
    r_m_Lambertian = np.array(r_m_Lambertian)
    r_m_nonLambertian = np.array(r_m_nonLambertian)

    r_single = r_s
    r_total_Lambertian = r_s + r_m_Lambertian
    r_total_nonLambertian = r_s + r_m_nonLambertian

    return r_single, r_total_Lambertian, r_total_nonLambertian


def step(x):
    output = np.zeros_like(x)
    output[x >= 0] = 1
    return output


def cot(x):
    return 1 / np.tan(x)


def int2d(f, u, v, u_axis, v_axis):
    """
    Computes the 2D integral of a function f along axes u and v

    Parameters
    ----------
    f : numpy array
        the function to be integrated
    u : numpy array
        2D array describing the u
    v : 1d numpy array
        2D array describing the v
    u_axis : int
        index corresponding to the u axis in f
    v_axis : int
        index corresponding to the v axis in f
    """
    N_axes = len(f.shape)
    axes = list(range(N_axes))
    int_u = trapz(f, u, axis=u_axis)
    # move u axis to the beginning, then choose the first value (v is the same for every u)
    v = np.moveaxis(v, source=u_axis, destination=0)[0]
    axes.remove(u_axis)
    v_axis = np.where(np.array(axes) == v_axis)[0][0]
    int_uv = trapz(int_u, v, axis=v_axis)
    return int_uv
