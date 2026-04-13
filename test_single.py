import taufactor as tau
import tifffile as tiff
import numpy as np

# --- 0. Load the Data ---
filename = 'nmc-1-uncal-withcbd-w001-binarized.tif'
print(f"Loading {filename}...")
voxel_array = tiff.imread(filename)

print(f"Success! Array shape is: {voxel_array.shape}")

# --- 1. Check the 3 Phase Fractions ---
pore_voxels = np.sum(voxel_array == 0)
active_voxels = np.sum(voxel_array == 1)
cbd_voxels = np.sum(voxel_array == 2)
total_voxels = voxel_array.size

print(f"\nPorosity (Empty Space): {(pore_voxels / total_voxels) * 100:.2f}%")
print(f"Active Material (Solid): {(active_voxels / total_voxels) * 100:.2f}%")
print(f"Carbon Binder (Glue): {(cbd_voxels / total_voxels) * 100:.2f}%")

# --- 2. Prepare the data for TauFactor ---
# Here we treat BOTH Pores (0) and CBD (2) as the conductive pathway (1)
print("\nIsolating Pore + CBD network for TauFactor...")
pore_network = np.where((voxel_array == 0) | (voxel_array == 2), 1, 0)

# --- 3. Calculate Tortuosity Factor USING CUDA ---
print("Running TauFactor calculation on GPU (CUDA)...")
solver = tau.Solver(pore_network, device='cuda')
solver.solve()

# We use .item() here to ensure the NumPy array prints as a single number
print(f"\nTortuosity Factor (Z axis): {solver.tau.item():.4f}")
print("Calculation complete using GPU!")