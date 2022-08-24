import numpy as np
from scipy.special import erfc, erf
from scipy.integrate import trapz
import matplotlib.pyplot as plt
from define_root_directory import ROOT_DIR
plt.style.use(f'{ROOT_DIR}/macroscopic_roughness_correction/plotters/custom.mplstyle')


validation = True

# Eq. 4, corrected and put in terms of nu, see Bourlier & Berginc 2003
def Lambda(nu):
    return 1 / (2 * np.sqrt(np.pi) * nu) * np.exp(-nu ** 2) - 1/2 * erfc(nu)


# Heaviside Function
def Y(x):
    return x > 0


# Eq. 12
def p_gamma(gamma_0x, sigma_gamma):
    return 1/(np.sqrt(2 * np.pi) * sigma_gamma) * np.exp(-1/2 * (gamma_0x / sigma_gamma) ** 2)


# Eq. 13, corrected and put in terms of nu
def S_gamma_bar(nu_A, nu_B, phi):
    # integrate out to 5 stdev for the Gaussian
    G_x = np.linspace(start=-5, stop=nu_A, num=10000, endpoint=True)
    integrand = 1 / (2 * np.sqrt(np.pi)) * np.exp(-G_x ** 2) * (1 + erf((nu_B - G_x * np.cos(phi)) / np.sin(phi)))
    integral = trapz(y=integrand, x=G_x, axis=0)
    return integral


# Eq. 28, using the author's code in the appendix, not the equations listed in the text
def r_0(nu_A, nu_B, phi):
    nu_A = np.atleast_1d(nu_A)
    nu_B = np.atleast_1d(nu_B)
    phi = np.atleast_1d(phi)
    assert len(nu_A) == len(nu_B) == len(phi)
    diff = np.atleast_1d(nu_A - nu_B)
    r0 = []
    for index, d in enumerate(diff):
        r0_single = r_0_single(d, phi[index])
        r0.append(r0_single)
    r0 = np.array(r0)
    r0[phi >= np.pi/2] = 1
    return r0


def r_0_single(d, phi):
    with np.errstate(divide='raise'):
        try:
            a = 0.17 / np.abs(d) ** 10.49
        except:
            # nu_A and nu_B are very close, return 1
            return 1
        b = 8.85
        numerator = np.log(1 + a * phi ** b)
        denominator = np.log(1 + a * (np.pi / 2) ** b)
        with np.errstate(all='raise'):
            try:
                return numerator / denominator
            except:
                # limit as diff -> infinity
                return 0.01837972835 * phi ** b


def S(nu_A, nu_B, phi):
    return S_gamma_bar(nu_A, nu_B, phi) * p_not_proj(nu_A, nu_B, phi)


def p_not_proj(nu_A, nu_B, phi):
    """
    Computes the probability that a facet is not in projected shadow, given that it is not in tilt shadow
    This just returns 1 / denominator from S, without computing the numerator.

    Parameters
    ----------
    nu_A : 1d numpy array
        normalized slope (lesser)
    nu_B : 1d numpy array
        normalized slope (greater)
    phi : float
        azimuth angle

    Returns
    -------
    P : 1d numpy array
        probability of not being in projected shadow, given not tilt shadowed
    """
    L_A = Lambda(nu_A)
    L_B = Lambda(nu_B)
    r0 = r_0(nu_A, nu_B, phi)
    return 1 / (1 + L_A + r0 * L_B)


def nu_AB_i_e_M(i, e, M):
    """
    Computes nu_A and nu_B from i, e, and M
    """
    nu_A = 1 / np.tan(np.max(np.vstack((i, e)), axis=0)) / (np.sqrt(2) * M)
    nu_B = 1 / np.tan(np.min(np.vstack((i, e)), axis=0)) / (np.sqrt(2) * M)
    return nu_A, nu_B


if validation:
    phi = np.deg2rad(np.linspace(start=0, stop=90, num=100))
    nu_list = [(0.10, 0.25), (0.25, 0.26)]
    for index, nu in enumerate(nu_list):
        nu_A = nu_list[index][0] * np.ones_like(phi)
        nu_B = nu_list[index][1] * np.ones_like(phi)
        r0 = r_0(nu_A=nu_A, nu_B=nu_B, phi=phi)
        plt.figure(index)
        plt.plot(np.rad2deg(phi), r0)
        plt.title(r'$\nu_A = $' + f' {nu_list[index][0]:.2f},  ' + r'$\nu_B = $' + f' {nu_list[index][1]:.2f}')
        plt.xticks([0, 10, 20, 30, 40, 50, 60, 70, 80, 90])
        plt.ylim([0, 1])
        plt.xlabel(r'$\phi$')
        plt.ylabel(r'$r_0$')
        plt.savefig(f'{ROOT_DIR}/macroscopic_roughness_correction/roughness_models/heitz_validation_figures/r_0_{index:.0f}.png')
        plt.close()

    phi = np.deg2rad(np.linspace(start=0.1, stop=179.9, num=200))
    nu_list = [(0.10, 0.11), (0.25, 0.26), (0.10, 0.25), (0.25, 0.50), (0.10, 0.50), (0.50, 0.51)]
    y_lim = [[0.0, 0.3], [0.1, 0.6], [0.0, 0.3], [0.1, 0.6], [0.0, 0.3], [0.3, 0.9]]
    yticks = [[0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3], [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
              [0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3], [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
              [0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3], [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]]
    fig = plt.figure(4, figsize=(20, 30))
    for index, nu in enumerate(nu_list):
        ax = fig.add_subplot(int('32' + str(index + 1)))
        nu_A = nu_list[index][0] * np.ones_like(phi)
        nu_B = nu_list[index][1] * np.ones_like(phi)
        illumination = S(nu_A=nu_A, nu_B=nu_B, phi=phi)
        ax.plot(np.rad2deg(phi), illumination, 'r')
        ax.set_xlabel(r'$\phi$')
        ax.set_xlim([0, 180])
        ax.set_ylim(y_lim[index])
        ax.set_title(r'$\nu_A = $' + f' {nu_list[index][0]:.2f}, ' + r'$\nu_B = $' + f'{nu_list[index][1]:.2f}')
        ax.set_yticks(yticks[index])
        ax.set_xticks([0, 20, 40, 60, 80, 100, 120, 140, 160, 180])
    plt.tight_layout()
    plt.savefig(f'{ROOT_DIR}/macroscopic_roughness_correction/roughness_models/heitz_validation_figures/bistatic_illumination.png')
    plt.close()
