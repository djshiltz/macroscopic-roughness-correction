import numpy as np
import matplotlib.pyplot as plt
from macroscopic_roughness_correction.miscellaneous.coordinate_transformations import g_i_e_psi
import pickle
from define_root_directory import ROOT_DIR
plt.style.use(f'{ROOT_DIR}/macroscopic_roughness_correction/plotters/custom.mplstyle')
from macroscopic_roughness_correction.plotters.brf_plotter import plot_brf


wavelength = 750
i = 30
phase_angle_min = 5
min_r_value = None
max_r_value = None
dpi = 300


wavelengths = np.linspace(350, 2500, 2151, endpoint=True).astype(int)
wavelength_index = np.where(wavelengths == wavelength)[0][0]
fig, axes = plt.subplots(1, 2, figsize=(6.5, 3))
for index, mineral in enumerate(['olivine', 'quartz']):
    ax = axes[index]
    interpolation = pickle.load(
        open(f'{ROOT_DIR}/macroscopic_roughness_correction/data/{mineral}/smooth_brf_interpolation.obj', 'rb'))

    i_deg = []
    e_deg = []
    psi_deg = []
    BRF = []
    for e in np.linspace(0, 90, 10).astype(int):
        for psi in np.linspace(0, 180, 19).astype(int):
            i_deg.append(i)
            e_deg.append(e)
            psi_deg.append(psi)
            g = g_i_e_psi(i=np.deg2rad(i), e=np.deg2rad(e), psi=np.deg2rad(psi))
            BRF.append(interpolation((i, e, psi))[wavelength_index])
    i_deg = np.array(i_deg)
    e_deg = np.array(e_deg)
    psi_deg = np.array(psi_deg)
    BRF = np.array(BRF)
    r = BRF * np.cos(np.deg2rad(i_deg)) / np.pi
    phi_deg_CW = psi_deg

    plot_brf(fig=fig, ax=ax, i_deg=i, e_deg=e_deg, phi_deg_CW=phi_deg_CW, BRF=r, mirror_hemisphere=True)
    if mineral == 'olivine':
        ax.set_title(r'$\mathrm{Olivine}$', fontsize=12)
    elif mineral == 'quartz':
        ax.set_title(r'$\mathrm{Quartz}$', fontsize=12)
plt.tight_layout()
plt.savefig(f'{ROOT_DIR}/macroscopic_roughness_correction/results/figures/interpolated_reflectance_{dpi:.0f}dpi.png', dpi=dpi)