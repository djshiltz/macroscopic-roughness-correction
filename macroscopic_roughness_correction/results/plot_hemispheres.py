import numpy as np
from macroscopic_roughness_correction.processing.fitting_parameters import c_L, c_NL
import matplotlib.pyplot as plt
from macroscopic_roughness_correction.plotters.brf_plotter import plot_brf
import pickle
from matplotlib.colors import LinearSegmentedColormap
from define_root_directory import ROOT_DIR


dpi = 300
plot_wavelength = 1100

#                       olivine    quartz
min_reflectance_value = [0.062,    0.200]
max_reflectance_value = [0.096,    0.247]

ticks = [[0.065, 0.07, 0.075, 0.08, 0.085, 0.09, 0.095], [0.20, 0.205, 0.210, 0.215, 0.220, 0.225, 0.230, 0.235, 0.240, 0.245]]
labels = [[r'$0.065$', r'$0.070$', r'$0.075$', r'$0.080$', r'$0.085$', r'$0.090$', r'$0.095$'],
          [r'$0.200$', r'$0.205$', r'$0.210$', r'$0.215$', r'$0.220$', r'$0.225$', r'$0.230$', r'$0.235$', r'$0.240$', r'$0.245$']]


cmap = LinearSegmentedColormap.from_list('custom', colors=['darkviolet', 'mediumpurple', 'deepskyblue', 'seagreen', 'lawngreen', 'orange', 'red', 'fuchsia', 'white'])
#cmap = 'gist_ncar'

for n, mineral in enumerate(['olivine', 'quartz']):
    rough_collect = np.load(
        f'{ROOT_DIR}/macroscopic_roughness_correction/data/{mineral}/processed_data/rough_hemisphere_processed.npy',
        allow_pickle=True).item()
    wavelength = rough_collect['wavelength']
    wavelength_index = np.where(wavelength == plot_wavelength)[0][0]

    albedo = np.load(f'{ROOT_DIR}/macroscopic_roughness_correction/data/albedo.npy', allow_pickle=True).item()
    r0 = albedo[mineral]['r0']

    # SMOOTH BRDF
    interpolation = pickle.load(
        open(f'{ROOT_DIR}/macroscopic_roughness_correction/data/{mineral}/smooth_brf_interpolation.obj', 'rb'))
    i = 30
    phase_angle_min = 5
    i_deg = []
    e_deg = []
    psi_deg = []
    BRF = []
    for e in np.linspace(0, 70, 10).astype(int):
        for psi in np.linspace(0, 180, 19).astype(int):
            i_deg.append(i)
            e_deg.append(e)
            psi_deg.append(psi)
            BRF.append(interpolation((i, e, psi))[wavelength_index])
    i_deg = np.array(i_deg)
    e_deg = np.array(e_deg)
    psi_deg = np.array(psi_deg)
    BRF = np.array(BRF)
    phi_deg_CW = psi_deg
    r = BRF * np.cos(np.deg2rad(i_deg)) / np.pi

    fig, ax = plt.subplots(4, 4)
    for column in [0, 1, 2, 3]:
        mappable = plot_brf(fig=fig, ax=ax[0, column], i_deg=i, e_deg=e_deg, phi_deg_CW=phi_deg_CW, BRF=r, mirror_hemisphere=True,
                 min_value=min_reflectance_value[n], max_value=max_reflectance_value[n], cmap=cmap, colorbar=False)
    for row, mold in enumerate(['green', 'pink', 'red']):
        row += 1
        indices = np.where(rough_collect['mold'] == mold)

        # measured
        plot_brf(fig=fig, ax=ax[row, 0], i_deg=i, e_deg=rough_collect['e_deg'][indices],
                 phi_deg_CW=rough_collect['phi_deg_CW'][indices],
                 BRF=rough_collect['r_mean'][indices, wavelength_index].flatten(), mirror_hemisphere=True,
                 min_value=min_reflectance_value[n], max_value=max_reflectance_value[n], cmap=cmap, colorbar=False)

        # proposed model
        M = rough_collect['M'][indices][0]
        g = np.deg2rad(rough_collect['g_deg'][indices])
        r_rough_shiltz_multi = c_L * r0[wavelength_index] * M * np.cos(np.deg2rad(i)) / np.pi \
            * (1 + c_NL * np.exp(-4/np.pi * (np.pi - g)**2))
        plot_brf(fig=fig, ax=ax[row, 1], i_deg=i, e_deg=rough_collect['e_deg'][indices],
                 phi_deg_CW=rough_collect['phi_deg_CW'][indices],
                 BRF=(rough_collect['r_rough_shiltz_single'][indices, wavelength_index] + r_rough_shiltz_multi).flatten(),
                 mirror_hemisphere=True, min_value=min_reflectance_value[n], max_value=max_reflectance_value[n], cmap=cmap,
                 colorbar=False)

        # Hapke's model (single)
        plot_brf(fig=fig, ax=ax[row, 2], i_deg=i, e_deg=rough_collect['e_deg'][indices],
                 phi_deg_CW=rough_collect['phi_deg_CW'][indices],
                 BRF=rough_collect['r_rough_hapke_single'][indices, wavelength_index].flatten(), mirror_hemisphere=True,
                 min_value=min_reflectance_value[n], max_value=max_reflectance_value[n], cmap=cmap, colorbar=False)

        # Hapke's model (modified)
        plot_brf(fig=fig, ax=ax[row, 3], i_deg=i, e_deg=rough_collect['e_deg'][indices],
                 phi_deg_CW=rough_collect['phi_deg_CW'][indices],
                 BRF=rough_collect['r_rough_hapke_modified'][indices, wavelength_index].flatten(), mirror_hemisphere=True,
                 min_value=min_reflectance_value[n], max_value=max_reflectance_value[n], cmap=cmap, colorbar=False)

    ax[0, 0].set_title(r'$\mathrm{Measured}$', fontsize=12)
    ax[0, 0].set_ylabel(r'$M = 0$', fontsize=12)
    ax[0, 1].set_title(r'$\mathrm{Proposed}$' + '\n' + r'$\mathrm{Correction}$', fontsize=12)
    ax[0, 2].set_title(r"$\mathrm{Hapke's}$" + '\n' + r'$\mathrm{Correction}$'+ '\n' + r'$\mathrm{(Single)}$', fontsize=12)
    ax[0, 3].set_title(r"$\mathrm{Hapke's}$" + '\n' + r'$\mathrm{Correction}$' + '\n' + r'$\mathrm{(Modified)}$', fontsize=12)
    plt.tight_layout()

    ax[0, 0].text(-0.2, 0.5, r'$M = 0$' + '\n' + r'$\mathrm{(smooth)}$',
            transform=ax[0, 0].transAxes, fontsize=12,
            va='center', ha='center')
    ax[1, 0].text(-0.25, 0.5, r'$M = 0.177$',
            transform=ax[1, 0].transAxes, fontsize=12,
            va='center', ha='center')
    ax[2, 0].text(-0.25, 0.5, r'$M = 0.265$',
            transform=ax[2, 0].transAxes, fontsize=12,
            va='center', ha='center')
    ax[3, 0].text(-0.25, 0.5, r'$M = 0.354$' + '\n' + r'$\mathrm{(roughest)}$',
            transform=ax[3, 0].transAxes, fontsize=12,
            va='center', ha='center')

    # add colorbar
    cbar = fig.colorbar(mappable, ax=ax.ravel().tolist(), ticks=ticks[n], aspect=50)
    cbar.ax.set_yticklabels(labels[n], fontsize=12)
    plt.savefig(f'{ROOT_DIR}/macroscopic_roughness_correction/results/figures/{mineral}_hemisphere_{dpi:.0f}dpi.png', dpi=dpi)
