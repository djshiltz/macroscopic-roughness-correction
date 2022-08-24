import numpy as np
from scipy import linalg
from macroscopic_roughness_correction.miscellaneous.coordinate_transformations import g_i_e_psi
from macroscopic_roughness_correction.data.define_smooth_reflectance import smooth_reflectance
from alive_progress import alive_bar


def d(r1, r2, psi):
    '''
    Returns the distance between the tips of two rays r1 and r2,
    separated by angle psi
    '''
    return np.sqrt(r1 ** 2 + r2 ** 2 - 2 * r1 * r2 * np.cos(psi))


def R(N, dr, psi):
    '''
    Builds distance matrix for single scattering

    Parameters:
    -----------
    N : int
        number of points in each line
    dr : float
        spacing between points
    psi : float
        angular spacing between lines

    Outputs:
    --------
    distances : (2N+2) x (2N+2) array
        matrix of distances between each of the points
    '''
    # Start by building the upper triangular part of the matrix
    R_upper = np.zeros((2 * N + 2, 2 * N + 2))
    # top row
    R_upper[0, 1:(N + 1)] = np.arange(start=1, stop=(N + 1), step=1) * dr
    R_upper[0, (N + 1):(2 * N + 1)] = np.arange(start=1, stop=(N + 1), step=1) * dr
    R_upper[0, -1] = dr
    # toeplitz blocks
    column = np.arange(start=0, stop=N, step=1) * dr
    R_upper[1:(N + 1), 1:(N + 1)] = np.triu(linalg.toeplitz(c=column))
    R_upper[(N + 1):(2 * N + 1), (N + 1):(2 * N + 1)] = np.triu(linalg.toeplitz(c=column))
    # distance block
    a = np.expand_dims(np.arange(start=1, stop=(N + 1), step=1) * dr, axis=0)
    r1 = np.repeat(a=a, repeats=N, axis=0)
    r2 = r1.T
    d_upper = d(r1=np.triu(r1), r2=np.triu(r2), psi=psi)
    d_full = d_upper.T + np.triu(d_upper, k=1)
    R_upper[1:(N + 1), (N + 1):(2 * N + 1)] = d_full
    # last column
    R_upper[1:(N + 1), -1] = np.squeeze(d(dr, a.T, psi=np.pi/3))
    R_upper[(N + 1):(2 * N + 1), -1] = np.squeeze(d(dr, a.T, psi=np.pi/3 - psi))
    return R_upper + R_upper.T


def random_surface(num_surfaces, M, L, dr, psi):
    '''
    Generates randomly distributed heights along two transects intersecting at azimuth angle psi

    Parameters:
    -----------
    num_surfaces : int
        number of random surfaces to generate
    M : float
        RMS slope of the surface
    L : float
        length of transects in units of correlation length
    dr : float
        distance between points in units of correlation length
    psi : float
        relative azimuth angle between transects [radians]

    Outputs:
    --------
    z : ndarray
        set of M randomly generated surfaces
    '''

    # Correlation length
    l = 1

    # RMS height
    s = l * M / np.sqrt(2)

    # Number of points per transect
    N = int(L * l / dr)

    # Generate transects
    cov = s ** 2 * np.exp(-(R(N=N, dr=dr, psi=psi) / l) ** 2)
    mean = np.zeros(2 * N + 2)
    z = np.random.multivariate_normal(mean=mean, cov=cov, size=num_surfaces)
    z_A = z[:, 0:(N + 1)]
    z_B = np.hstack((np.expand_dims(z[:, 0], 1), z[:, (N + 1):(2 * N + 1)]))
    z_facet = np.vstack((z[:, 0], z[:, 1], z[:, 2 * N + 1])).T
    r = np.arange(start=0, stop=N+1, step=1) * dr
    return z_A, z_B, z_facet, r, dr


def compute_angles(z_facet, i, e, psi, dr):

    n_points = z_facet.shape[0]
    p = np.vstack((np.ones(n_points) * dr, np.zeros(n_points), z_facet[:, 1] - z_facet[:, 0])).T
    q = np.vstack((np.ones(n_points) * dr/2, np.ones(n_points) * np.sqrt(3)/2 * dr, z_facet[:, 2] - z_facet[:, 0])).T
    n = np.cross(p, q)
    n_mag = np.linalg.norm(n, axis=1)
    n_hat = (n.T / n_mag).T
    facet_area = n_mag * 1/2

    z_hat = np.array([0, 0, 1])
    cos_theta = np.dot(n_hat, z_hat)
    theta = np.arccos(cos_theta)

    i_hat = np.array([np.sin(i), 0, np.cos(i)])
    cos_iota = np.dot(n_hat, i_hat)
    iota = np.arccos(cos_iota)

    e_hat = np.array([np.cos(psi)*np.sin(e), np.sin(psi)*np.sin(e), np.cos(e)])
    cos_epsilon = np.dot(n_hat, e_hat)
    epsilon = np.arccos(cos_epsilon)

    return theta, iota, epsilon, facet_area


def p_monte_carlo(i, e, psi, M, num_surfaces, L, dr):
    """
    Computes various roughness probabilities using Monte Carlo simulation

    Parameters
    ----------
    i : float
        zenith angle of incident light [rad]
    e : float
        zenith angle of exiting light [rad]
    psi : float
        relative azimuth angle between transects [radians]
    M : float
        RMS slope of the surface
    num_surfaces : int
        number of random surfaces to generate
    L : float
        length of transects in units of correlation length
    dr : float
        distance between points in units of correlation length

    Returns
    -------
    p_not_proj_given_not_tilt : float
        probability that a facet is not in a projected shadow, given that it is in tilt shadow
    """
    print('Running Monte Carlo')
    g = g_i_e_psi(i=i, e=e, psi=psi)
    z_A, z_B, z_facet, r, dr = random_surface(num_points=num_surfaces, M=M, L=L, dr=dr, psi=psi)
    theta, iota, epsilon, facet_area = compute_angles(z_facet, i, e, psi, dr)

    tilt_shadow = iota > np.pi/2
    tilt_mask = epsilon > np.pi/2
    tilted = np.logical_or(tilt_shadow, tilt_mask)

    ray_A = (z_A[:, 0] + np.expand_dims(r / np.tan(i), axis=-1) * np.ones(num_surfaces)).T
    ray_B = (z_B[:, 0] + np.expand_dims(r / np.tan(e), axis=-1) * np.ones(num_surfaces)).T

    projected_shadow = np.any(ray_A < z_A, axis=-1)
    projected_mask = np.any(ray_B < z_B, axis=-1)
    projected = np.logical_or(projected_shadow, projected_mask)

    p_not_proj_given_not_tilt = np.sum(np.logical_and(np.logical_not(tilted), np.logical_not(projected))) \
        / np.sum(np.logical_not(tilted))

    return p_not_proj_given_not_tilt


def hist_plot(x, bins=100, weights=1, range=None):
    x_hist, bin_edges = np.histogram(x, bins=bins, density=True, range=range, weights=weights * np.ones_like(x))
    x_plot = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    return x_plot, x_hist


def r_monte_carlo(i, e, psi, M, mineral, num_surfaces=100000, L=10, dr=0.05):
    """
    Computes the bidirectional reflectance of a rough surface using Monte Carlo simulation

    Parameters
    ----------
    i : float
        zenith angle of incident light [rad]
    e : float
        zenith angle of exiting light [rad]
    psi : float
        relative azimuth angle between transects [radians]
    M : float
        RMS slope of the surface
    mineral : str
        'quartz' or 'olivine
    num_surfaces : int
        number of random surfaces to generate
    L : float
        length of transects in units of correlation length
    dr : float
        distance between points in units of correlation length

    Returns
    -------
    r_rough : 1d numpy array
        bidirectional reflectance of rough surface
    """
    print('Running Monte Carlo')
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

    R_ROUGH = []
    with alive_bar(num_points) as bar:
        for point_index in range(num_points):
            i = i_list[point_index]
            e = e_list[point_index]
            psi = psi_list[point_index]
            M = M_list[point_index]
            g = g_list[point_index]
            z_A, z_B, z_facet, r, dr = random_surface(num_surfaces=num_surfaces, M=M, L=L, dr=dr, psi=psi)
            theta, iota, epsilon, facet_area = compute_angles(z_facet, i, e, psi, dr)

            tilt_shadow = iota > np.pi/2
            tilt_mask = epsilon > np.pi/2
            tilted = np.logical_or(tilt_shadow, tilt_mask)

            ray_A = (z_A[:, 0] + np.expand_dims(r / np.tan(i), axis=-1) * np.ones(num_surfaces)).T
            ray_B = (z_B[:, 0] + np.expand_dims(r / np.tan(e), axis=-1) * np.ones(num_surfaces)).T

            projected_shadow = np.any(ray_A < z_A, axis=-1)
            projected_mask = np.any(ray_B < z_B, axis=-1)
            projected = np.logical_or(projected_shadow, projected_mask)

            # Smooth surface reflectance
            reflectance = smooth_reflectance(i=iota, e=epsilon, g=g, mineral=mineral)
            reflectance[tilted] = 0
            reflectance[projected] = 0
            r_rough = np.mean((reflectance.T * (1/np.cos(theta)) * np.cos(epsilon)/np.cos(e)).T, axis=0)

            R_ROUGH.append(r_rough)
            bar()

    return np.array(R_ROUGH)
