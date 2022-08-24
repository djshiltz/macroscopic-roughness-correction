import numpy as np
from macroscopic_roughness_correction.miscellaneous.coordinate_transformations import psi_phi_CW
from macroscopic_roughness_correction.roughness_models.hapke_roughness_correction import r_hapke
from macroscopic_roughness_correction.roughness_models.shiltz_roughness_correction import r_shiltz
from macroscopic_roughness_correction.roughness_models.monte_carlo import r_monte_carlo
from define_root_directory import ROOT_DIR


for mineral in ['olivine', 'quartz']:
    print(f'Processing {mineral}:')
    collect = np.load(f'{ROOT_DIR}/macroscopic_roughness_correction/data/{mineral}/rough_transect_measurements.npy',
                      allow_pickle=True).item()
    i = np.deg2rad(collect['i_deg'])
    e = np.deg2rad(collect['e_deg'])
    phi_CW = np.deg2rad(collect['phi_deg_CW'])
    psi = psi_phi_CW(phi_CW)
    M = collect['M']

    r_rough_hapke_single = r_hapke(i=i, e=e, psi=psi, M=M, mineral=mineral, multifacet_correction=False)
    r_rough_hapke_modified = r_hapke(i=i, e=e, psi=psi, M=M, mineral=mineral, multifacet_correction=True)
    r_rough_shiltz_single, r_total_Lambertian, r_total_nonLambertian = r_shiltz(i=i, e=e, psi=psi, M=M, mineral=mineral)
    r_monte_carlo_single = r_monte_carlo(i=i, e=e, psi=psi, M=M, mineral=mineral)

    # Processed Scan
    collect_processed = collect
    collect_processed['r_rough_hapke_single'] = r_rough_hapke_single
    collect_processed['r_rough_hapke_modified'] = r_rough_hapke_modified
    collect_processed['r_rough_shiltz_single'] = r_rough_shiltz_single
    collect_processed['r_monte_carlo'] = r_monte_carlo_single
    np.save(f'{ROOT_DIR}/macroscopic_roughness_correction/data/{mineral}/processed_data/rough_transect_processed.npy', collect_processed, allow_pickle=True)
