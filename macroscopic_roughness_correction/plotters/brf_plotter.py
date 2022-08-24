import numpy as np
import matplotlib.pyplot as plt
from define_root_directory import ROOT_DIR
plt.style.use(f'{ROOT_DIR}/macroscopic_roughness_correction/plotters/custom.mplstyle')
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.interpolate import griddata


def plot_brf(fig, ax, i_deg, e_deg, phi_deg_CW, BRF, min_value=None, max_value=None, nadir_threshold_deg=2,
             self_shadow_radius_deg=5, mirror_hemisphere=False, carriage_position=None, cmap='jet', colorbar=True):
    '''
    Parameters:
    -----------
    ax : matplotlib axis
        The handle for the axis upon which to plot the reflectance
    i_deg : int or float
        Illumination angle in degrees
    e_deg : list or 1d numpy array
        Emergence angle in degrees
    phi_deg_CW : list or 1d numpy array
        Relative azimuth angle, clockwise from light source, in degrees
    BRF : list or 1d numpy array
        Reflectance at a given wavelength
    min_value : float or None
        Minimum value on colorbar
    max_value : float or None
        Maximum value on colorbar
    nadir_threshold_deg : float
        Points with emergence angles less than this are considered "nadir" and averaged together
    self_shadow_radius_deg : float
        Radius of self-shadowing region
    mirror_hemisphere : bool
        Only use data from one side of the hemisphere (carriage position <= 98.2) and force the other half
        to mirror it (used for Dylan's roughness scans)
    carriage_position : list or 1d numpy array
        GRIT's carriage position in degrees (used only if mirror_hemisphere == True)
    cmap : str
        color map
    '''
    assert np.size(np.array(i_deg)) == 1
    assert len(BRF.shape) == 1

    if mirror_hemisphere:
        # DOUBLE ARRAY LENGTH (for psi > 180)
        e_deg = np.hstack((e_deg, e_deg))
        phi_deg_CW = np.hstack((phi_deg_CW, 360 - phi_deg_CW))
        BRF = np.hstack((BRF, BRF))

    # average nadir
    nadir_indices = np.abs(e_deg) <= nadir_threshold_deg
    average_nadir_reflectance = np.mean(BRF[nadir_indices])
    BRF[nadir_indices] = average_nadir_reflectance

    resolution = 1001
    midpoint = (resolution - 1) / 2
    x = e_deg * np.sin(np.deg2rad(phi_deg_CW))
    y = e_deg * np.cos(np.deg2rad(phi_deg_CW))
    X, Y = np.meshgrid(np.linspace(-90, 90, resolution),
                       np.linspace(-90, 90, resolution))

    BRF = griddata(points=(x, y),
                   values=BRF,
                   xi=(X, Y),
                   method='linear')
    mappable = ax.imshow(BRF, cmap=cmap, vmin=min_value, vmax=max_value, origin='lower')
    light_source = plt.Circle(xy=(midpoint, midpoint + i_deg * resolution / 180),
                              radius=resolution * self_shadow_radius_deg / 180, color='black', fill=True)
    ax.add_patch(light_source)
    for zenith in [10, 20, 30, 40, 50, 60, 70, 80, 89]:
        ring = plt.Circle(xy=(midpoint, midpoint), radius=zenith / 180 * resolution, color='black', fill=False)
        ax.add_patch(ring)

    for azimuth in [0, 45, 90, 135]:
        line = plt.Line2D((midpoint + 0.495 * np.sin(np.deg2rad(azimuth)) * resolution,
                           midpoint - 0.495 * np.sin(np.deg2rad(azimuth)) * resolution),
                          (midpoint + 0.495 * np.cos(np.deg2rad(azimuth)) * resolution,
                           midpoint - 0.495 * np.cos(np.deg2rad(azimuth)) * resolution),
                          color='black', lw=1)
        ax.add_line(line)
    ax.set_axis_off()
    if colorbar:
        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size='5%', pad=0.05)
        cbar = fig.colorbar(mappable, cax=cax, orientation='vertical')
        cbar.ax.tick_params(labelsize=9)

    return mappable