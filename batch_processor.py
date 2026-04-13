import taufactor as tau
import tifffile as tiff
import numpy as np
import os
import pandas as pd


input_folder = 'nrel_data'
output_folder = 'results'
output_file = os.path.join(output_folder, 'battery_analysis_results.csv')

if not os.path.exists(output_folder):
    os.makedirs(output_folder)

results_list = []

files = [f for f in os.listdir(input_folder) if f.endswith('.tif')]
print(f"Found {len(files)} files in {input_folder}. Starting GPU processing...")

for filename in files:
    print(f"\n--- Current File: {filename} ---")
    path = os.path.join(input_folder, filename)

    try:
        voxel_array = tiff.imread(path)

        # Phase Fractions
        total = voxel_array.size
        eps_pore = np.sum(voxel_array == 0) / total
        eps_active = np.sum(voxel_array == 1) / total
        eps_cbd = np.sum(voxel_array == 2) / total

        # Calculation A: Pores Only (Realistic)
        print("Running: Pores-Only calculation...")
        net_pores = np.where(voxel_array == 0, 1, 0)
        solver_p = tau.Solver(net_pores, device='cuda')
        solver_p.solve()
        tau_pores = solver_p.tau.item()

        # Calculation B: Pores + CBD (Ideal/Effective)
        print("Running: Pores + CBD calculation...")
        net_combined = np.where((voxel_array == 0) | (voxel_array == 2), 1, 0)
        solver_c = tau.Solver(net_combined, device='cuda')
        solver_c.solve()
        tau_combined = solver_c.tau.item()


        results_list.append({
            'Filename': filename,
            'Porosity': eps_pore,
            'Active_Material': eps_active,
            'CBD_Fraction': eps_cbd,
            'Tau_Pores_Only': tau_pores,
            'Tau_Combined': tau_combined,
            'CBD_Penalty': tau_pores - tau_combined
        })

    except Exception as e:
        print(f"Error processing {filename}: {e}")

if results_list:
    df = pd.DataFrame(results_list)
    df.to_csv(output_file, index=False)
    print(f"\n✅ SUCCESS!")
    print(f"Processed {len(results_list)} files.")
    print(f"Results saved to: {output_file}")
else:
    print("\n❌ No files were processed.")