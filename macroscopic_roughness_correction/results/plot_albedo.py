import numpy as np
import matplotlib.pyplot as plt
from define_root_directory import ROOT_DIR
plt.style.use(f'{ROOT_DIR}/macroscopic_roughness_correction/plotters/custom.mplstyle')


dpi=300

albedo = np.load(f'{ROOT_DIR}/macroscopic_roughness_correction/data/albedo.npy', allow_pickle=True).item()
wavelength = np.linspace(350, 2500, 2151)
good_wavelengths = np.logical_and(wavelength >= 500, wavelength <= 2250)

fig, ax = plt.subplots(3, 2, figsize=(8, 8))
ax[0, 0].plot(wavelength[good_wavelengths], albedo['olivine']['w'][good_wavelengths], 'g', label=r'$\mathrm{Olivine}$')
ax[0, 0].plot(wavelength[good_wavelengths], albedo['quartz']['w'][good_wavelengths], 'b', label=r'$\mathrm{Quartz}$')
ax[0, 0].set_xlabel(r'$\lambda \; \mathrm{[nm]}$', fontsize=12)
ax[0, 0].set_xticks([500, 750, 1000, 1250, 1500, 1750, 2000, 2250])
ax[0, 0].set_xticklabels([r'$500$', r'$750$', r'$1000$', r'$1250$', r'$1500$', r'$1750$', r'$2000$', r'$2250$'],
                   fontsize=12)
ax[0, 0].set_yticks([0.85, 0.90, 0.95, 1.0])
ax[0, 0].set_ylim([0.85, 1.01])
ax[0, 0].set_yticklabels([r'$0.85$', r'$0.90$', r'$0.95$', r'$1.00$'], fontsize=12)
ax[0, 0].set_ylabel(r'$w$', fontsize=12)

ax[1, 0].plot(wavelength[good_wavelengths], albedo['quartz']['b'][good_wavelengths], 'b')
ax[1, 0].plot(wavelength[good_wavelengths], albedo['olivine']['b'][good_wavelengths], 'g')
ax[1, 0].set_xlabel(r'$\lambda \; \mathrm{[nm]}$', fontsize=12)
ax[1, 0].set_xticks([500, 750, 1000, 1250, 1500, 1750, 2000, 2250])
ax[1, 0].set_xticklabels([r'$500$', r'$750$', r'$1000$', r'$1250$', r'$1500$', r'$1750$', r'$2000$', r'$2250$'],
                   fontsize=12)
ax[1, 0].set_yticks([0.2, 0.4, 0.6, 0.8])
ax[1, 0].set_ylim([0.2, 0.8])
ax[1, 0].set_yticklabels([r'$0.2$', r'$0.4$', r'$0.6$', r'$0.8$'], fontsize=12)
ax[1, 0].set_ylabel(r'$b$', fontsize=12)

ax[2, 0].plot(wavelength[good_wavelengths], albedo['quartz']['c'][good_wavelengths], 'b')
ax[2, 0].plot(wavelength[good_wavelengths], albedo['olivine']['c'][good_wavelengths], 'g')
ax[2, 0].set_xlabel(r'$\lambda \; \mathrm{[nm]}$', fontsize=12)
ax[2, 0].set_xticks([500, 750, 1000, 1250, 1500, 1750, 2000, 2250])
ax[2, 0].set_xticklabels([r'$500$', r'$750$', r'$1000$', r'$1250$', r'$1500$', r'$1750$', r'$2000$', r'$2250$'],
                   fontsize=12)
ax[2, 0].set_yticks([-1.2, -1.0, -0.8, -0.6])
ax[2, 0].set_ylim([-1.2, -0.6])
ax[2, 0].set_yticklabels([r'$-1.2$', r'$-1.0$', r'$-0.8$', r'$-0.6$'], fontsize=12)
ax[2, 0].set_ylabel(r'$c$', fontsize=12)

ax[0, 1].plot(wavelength[good_wavelengths], albedo['olivine']['w_star'][good_wavelengths], 'g')
ax[0, 1].plot(wavelength[good_wavelengths], albedo['quartz']['w_star'][good_wavelengths], 'b')
ax[0, 1].set_xlabel(r'$\lambda \; \mathrm{[nm]}$', fontsize=12)
ax[0, 1].set_xticks([500, 750, 1000, 1250, 1500, 1750, 2000, 2250])
ax[0, 1].set_xticklabels([r'$500$', r'$750$', r'$1000$', r'$1250$', r'$1500$', r'$1750$', r'$2000$', r'$2250$'],
                   fontsize=12)
ax[0, 1].set_yticks([0.6, 0.7, 0.8, 0.9, 1.0])
ax[0, 1].set_ylim([0.6, 1.02])
ax[0, 1].set_yticklabels([r'$0.60$', r'$0.70$', r'$0.80$', r'$0.90$', r'$1.00$'], fontsize=12)
ax[0, 1].set_ylabel(r'$w^*$', fontsize=12)

ax[1, 1].plot(wavelength[good_wavelengths], albedo['quartz']['beta'][good_wavelengths], 'b')
ax[1, 1].plot(wavelength[good_wavelengths], albedo['olivine']['beta'][good_wavelengths], 'g')
ax[1, 1].set_xlabel(r'$\lambda \; \mathrm{[nm]}$', fontsize=12)
ax[1, 1].set_xticks([500, 750, 1000, 1250, 1500, 1750, 2000, 2250])
ax[1, 1].set_xticklabels([r'$500$', r'$750$', r'$1000$', r'$1250$', r'$1500$', r'$1750$', r'$2000$', r'$2250$'],
                   fontsize=12)
ax[1, 1].set_yticks([0.2, 0.4, 0.6, 0.8])
ax[1, 1].set_ylim([0.2, 0.8])
ax[1, 1].set_yticklabels([r'$0.2$', r'$0.4$', r'$0.6$', r'$0.8$'], fontsize=12)
ax[1, 1].set_ylabel(r'$\beta$', fontsize=12)

ax[2, 1].plot(wavelength[good_wavelengths], albedo['quartz']['r0'][good_wavelengths], 'b')
ax[2, 1].plot(wavelength[good_wavelengths], albedo['olivine']['r0'][good_wavelengths], 'g')
ax[2, 1].set_xlabel(r'$\lambda \; \mathrm{[nm]}$', fontsize=12)
ax[2, 1].set_xticks([500, 750, 1000, 1250, 1500, 1750, 2000, 2250])
ax[2, 1].set_xticklabels([r'$500$', r'$750$', r'$1000$', r'$1250$', r'$1500$', r'$1750$', r'$2000$', r'$2250$'],
                   fontsize=12)
ax[2, 1].set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
ax[2, 1].set_ylim([0.2, 1.0])
ax[2, 1].set_yticklabels([r'$0.2$', r'$0.4$', r'$0.6$', r'$0.8$', r'$1.0$'], fontsize=12)
ax[2, 1].set_ylabel(r'$r_0$', fontsize=12)

fig.legend(fontsize=12, bbox_to_anchor=(0.70, 0.0), ncol=2)
plt.tight_layout()
plt.savefig(f'{ROOT_DIR}/macroscopic_roughness_correction/results/figures/albedo_{dpi}dpi.png', dpi=dpi)
