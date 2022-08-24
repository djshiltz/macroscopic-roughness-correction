import numpy as np
from macroscopic_roughness_correction.miscellaneous.coordinate_transformations import psi_phi_CW
from macroscopic_roughness_correction.miscellaneous.round_array_to_nearest_N import round_array_to_nearest_N
import matplotlib.pyplot as plt
from define_root_directory import ROOT_DIR
plt.style.use(f'{ROOT_DIR}/macroscopic_roughness_correction/plotters/custom.mplstyle')
from macroscopic_roughness_correction.roughness_models.shiltz_roughness_correction import r_shiltz
from macroscopic_roughness_correction.roughness_models.hapke_roughness_correction import r_hapke
from macroscopic_roughness_correction.data.define_smooth_reflectance import smooth_reflectance
from macroscopic_roughness_correction.miscellaneous.coordinate_transformations import g_i_e_psi


plot_wavelength = 1100
mold_name = 'red'    # red is the roughest mold
N_e = 100            # number of points in each transect along emergence direction
dpi = 300            # figure resolution

for mineral_index, mineral in enumerate(['quartz', 'olivine']):
    for i_index, I_deg in enumerate([10, 30, 60]):
        collect_combined = np.load(
            f'{ROOT_DIR}/macroscopic_roughness_correction/data/{mineral}/processed_data/rough_transect_processed.npy',
            allow_pickle=True).item()
        if mineral == 'olivine':
            title_string1 = r'$\mathrm{Olivine,\;\;}$'
        elif mineral == 'quartz':
            title_string1 = r'$\mathrm{Quartz,\;\;}$'
        title_string2 = f'$i=\mathrm{{ {I_deg} }}^\circ$'


        i_deg = collect_combined['i_deg'].astype(int)
        e_deg = collect_combined['e_deg']
        g_deg = collect_combined['g_deg']
        phi_deg_CW = collect_combined['phi_deg_CW']
        psi_deg = np.rad2deg(psi_phi_CW(np.deg2rad(phi_deg_CW))).astype(int)
        psi_deg = round_array_to_nearest_N(array=psi_deg, N=5)
        r_mean = collect_combined['r_mean']
        r_stdev = collect_combined['r_stdev']
        r_monte_carlo = collect_combined['r_monte_carlo']

        wavelength = collect_combined['wavelength']
        wavelength_index = np.where(wavelength == plot_wavelength)[0][0]
        M = collect_combined['M']
        mold = collect_combined['mold']

        fig_index = 0
        m = M[mold == mold_name][0]
        print(f'Plotting {mold_name} {mineral} at i = {I_deg} deg')

        fig, ax = plt.subplots(2, 2)
        ax = ax.flatten()
        plt.suptitle(title_string1 + title_string2, fontsize=18)
        ymin = []
        ymax = []

        for psi_index, Psi_deg in enumerate([0, 60, 120, 180]):
            plot_indices = np.where((i_deg == I_deg) * (psi_deg == Psi_deg) * (mold == mold_name))[0]
            nadir_index = np.where((e_deg < 5) * (i_deg == I_deg) * (mold == mold_name))[0][0]

            if np.min(e_deg[plot_indices]) > 5:
                # add a nadir index
                plot_indices = np.append(plot_indices, nadir_index)

            ax[psi_index].errorbar(x=e_deg[plot_indices],
                                   y=r_mean[plot_indices, wavelength_index],
                                   yerr=r_stdev[plot_indices, wavelength_index],
                                   markerfacecolor='black',
                                   markeredgecolor='black',
                                   marker='o',
                                   linestyle='None',
                                   ecolor='black',
                                   capsize=2,
                                   label=r'$\tilde{r}_\mathrm{Measured}$')
            ax[psi_index].set_xlim([-5, 75])


            e_deg_plot = np.linspace(0.1, 70, N_e)
            g_plot = g_i_e_psi(i=np.deg2rad(I_deg), e=np.deg2rad(e_deg_plot), psi=np.deg2rad(Psi_deg))

            R_HAPKE = r_hapke(i=np.deg2rad(I_deg), e=np.deg2rad(e_deg_plot), psi=np.deg2rad(Psi_deg), M=m, mineral=mineral)
            R_HAPKE_modified = r_hapke(i=np.deg2rad(I_deg), e=np.deg2rad(e_deg_plot), psi=np.deg2rad(Psi_deg), M=m, mineral=mineral,
                                       multifacet_correction=True)
            R_SHILTZ_SINGLE, R_SHILTZ_TOTAL_LAMBERTIAN, R_SHILTZ_TOTAL_NONLAMBERTIAN = r_shiltz(i=np.deg2rad(I_deg),
                                                                                                e=np.deg2rad(e_deg_plot),
                                                                                                psi=np.deg2rad(Psi_deg),
                                                                                                M=m, mineral=mineral)
            R_SMOOTH = smooth_reflectance(i=np.deg2rad(I_deg), e=np.deg2rad(e_deg_plot), g=g_plot, mineral=mineral)

            ax[psi_index].plot(e_deg_plot, R_HAPKE[:, wavelength_index], color='red', label=r'$\tilde{r}_\mathrm{Hapke,\;single}$')
            ax[psi_index].plot(e_deg_plot, R_HAPKE_modified[:, wavelength_index], color='blue',
                               label=r'$\tilde{r}_\mathrm{Hapke,\;modified}$')
            ax[psi_index].plot(e_deg_plot, R_SHILTZ_SINGLE[:, wavelength_index], color='purple', linestyle='solid', label=r'$\tilde{r}_\mathrm{Proposed,\;single}$')
            ax[psi_index].plot(e_deg_plot, R_SHILTZ_TOTAL_LAMBERTIAN[:, wavelength_index], color='green', linestyle='dashed',
                               label=r'$\tilde{r}_\mathrm{Proposed}\;\;(c_\mathrm{\,NL}=0)$')
            ax[psi_index].plot(e_deg_plot, R_SHILTZ_TOTAL_NONLAMBERTIAN[:, wavelength_index], color='green', linestyle='solid',
                               label=r'$\tilde{r}_\mathrm{Proposed}$')
            ax[psi_index].plot(e_deg_plot, R_SMOOTH[:, wavelength_index], color='black', linewidth=0, linestyle='dotted', label=' ')
            ax[psi_index].plot(e_deg[plot_indices],
                               r_monte_carlo[plot_indices, wavelength_index],
                               color='purple',
                               marker='v',
                               linestyle='None',
                               label=r'$\tilde{r}_\mathrm{Monte\;Carlo,\;single}$')
            '''
            ax[psi_index].plot(e_deg_plot,
                               smooth_reflectance(i=np.deg2rad(I_deg), e=np.deg2rad(e_deg_plot), g=g_plot, mineral=mineral)[:, wavelength_index],
                               color='black',
                               linestyle='dashed',
                               label=r'$r$')
            '''

            ax[psi_index].set_xticks([0, 15, 30, 45, 60, 75])
            ax[psi_index].set_xticklabels([r'$0^\circ$', r'$15^\circ$', r'$30^\circ$', r'$45^\circ$', r'$60^\circ$', r'$75^\circ$'], fontsize=12)
            ax[psi_index].set_xlabel(r'$e$', fontsize=12)

            # CUSTOM AXIS LIMITS / LABELS
            if mineral == 'quartz' and I_deg == 10:
                ax[psi_index].set_ylim([0.22, 0.26])
                ax[psi_index].set_yticks([0.22, 0.24, 0.26, 0.28])
                ax[psi_index].set_yticklabels([r'$0.22$', r'$0.24$', r'$0.26$', r'$0.28$'], fontsize=12)
            if mineral == 'quartz' and I_deg == 30:
                ax[psi_index].set_ylim([0.18, 0.26])
                ax[psi_index].set_yticks([0.18, 0.20, 0.22, 0.24, 0.26])
                ax[psi_index].set_yticklabels([r'$0.18$', r'$0.20$', r'$0.22$', r'$0.24$', r'$0.26$'], fontsize=12)
            if mineral == 'quartz' and I_deg == 60:
                ax[psi_index].set_ylim([0.08, 0.2])
                ax[psi_index].set_yticks([0.08, 0.10, 0.12, 0.14, 0.16, 0.18, 0.20])
                ax[psi_index].set_yticklabels([r'$0.08$', r'$0.10$', r'$0.12$', r'$0.14$', r'$0.16$', r'$0.18$', r'$0.20$'], fontsize=12)
            if mineral == 'olivine' and I_deg == 10:
                ax[psi_index].set_ylim([0.07, 0.09])
                ax[psi_index].set_yticks([0.07, 0.075, 0.08, 0.085, 0.09])
                ax[psi_index].set_yticklabels([r'$0.070$', r'$0.075$', r'$0.080$', r'$0.085$', r'$0.090$'], fontsize=12)
            if mineral == 'olivine' and I_deg == 30:
                ax[psi_index].set_ylim([0.06, 0.09])
                ax[psi_index].set_yticks([0.06, 0.07, 0.08, 0.09])
                ax[psi_index].set_yticklabels([r'$0.06$', r'$0.07$', r'$0.08$', r'$0.09$'], fontsize=12)
            if mineral == 'olivine' and I_deg == 60:
                ax[psi_index].set_ylim([0.03, 0.09])
                ax[psi_index].set_yticks([0.03, 0.05, 0.07, 0.09])
                ax[psi_index].set_yticklabels([r'$0.03$', r'$0.05$', r'$0.07$', r'$0.09$'], fontsize=12)

            ax[psi_index].set_ylabel(r'$\tilde{r}$', fontsize=12)
        # specify order of items in legend
        order = [7, 4, 3, 2, 5, 0, 1, 6]

        # get handles and labels
        handles, labels = plt.gca().get_legend_handles_labels()

        # add legend to plot
        #fig.legend(bbox_to_anchor=(0.8, 0.0))
        fig.legend([handles[idx] for idx in order], [labels[idx] for idx in order], bbox_to_anchor=(0.72, 0.0), ncol=2, fontsize=12)
        #ax[0].legend([handles[idx] for idx in order], [labels[idx] for idx in order], loc=4, fontsize=12)
        ax[0].set_title(r'$\psi = 0^\circ\;\;\mathrm{(backward\;direction)}$', fontsize=12)
        ax[1].set_title(r'$\psi = 60^\circ$', fontsize=12)
        ax[2].set_title(r'$\psi = 120^\circ$', fontsize=12)
        ax[3].set_title(r'$\psi = 180^\circ\;\;\mathrm{(forward\;direction)}$', fontsize=12)
        plt.tight_layout()
        plt.savefig(f'{ROOT_DIR}/macroscopic_roughness_correction/results/figures/{mold_name}_{mineral}_i{str(I_deg)}_{str(plot_wavelength)}nm_{dpi:.0f}dpi.png', dpi=dpi)
