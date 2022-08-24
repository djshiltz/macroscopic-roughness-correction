import numpy as np
from define_root_directory import ROOT_DIR
from macroscopic_roughness_correction.miscellaneous.coordinate_transformations import psi_i_e_g
import pickle


quartz_brf_interpolation = pickle.load(
    open(f'{ROOT_DIR}/macroscopic_roughness_correction/data/quartz/smooth_brf_interpolation.obj', 'rb'))
olivine_brf_interpolation = pickle.load(
    open(f'{ROOT_DIR}/macroscopic_roughness_correction/data/olivine/smooth_brf_interpolation.obj', 'rb'))

def smooth_reflectance(i, e, g, mineral):
    """
    Computes the bidirectional reflectance of smooth mineral.  Dimensions of i, e, and g must be broadcastable
    to one another

    Parameters
    ----------
    i : numpy array
        illumination angle [rad]
    e : numpy array
        emergence angle [rad]
    g : numpy array
        phase angle [rad]
    mineral : str
        name of mineral, either "quartz" or "olivine"

    Returns
    -------
    r : numpy array
        bidirectional reflectance
    """
    i_deg = np.rad2deg(i)
    e_deg = np.rad2deg(e)
    psi_deg = np.rad2deg(psi_i_e_g(i=i, e=e, g=g))

    if mineral == 'olivine':
        brf = olivine_brf_interpolation((i_deg, e_deg, psi_deg))
    elif mineral == 'quartz':
        brf = quartz_brf_interpolation((i_deg, e_deg, psi_deg))
    r = (brf.T * np.cos(i).T / np.pi).T
    return r
