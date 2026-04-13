import taufactor as tau
import tifffile as tiff
import numpy as np

filename = 'nmc-1-uncal-withcbd-w001-binarized.tif'
print(f"Loading {filename}...")
voxel_array = tiff.imread(filename)

print(f"Success! Array shape is: {voxel_array.shape}")

pore_voxels = np.sum(voxel_array == 0)
active_voxels = np.sum(voxel_array == 1)
cbd_voxels = np.sum(voxel_array == 2)
total_voxels = voxel_array.size

print(f"\nPorosity (Empty Space): {(pore_voxels / total_voxels) * 100:.2f}%")
print(f"Active Material (Solid): {(active_voxels / total_voxels) * 100:.2f}%")
print(f"Carbon Binder (Glue): {(cbd_voxels / total_voxels) * 100:.2f}%")

print("\nIsolating Pore + CBD network for TauFactor...")
pore_network = np.where((voxel_array == 0) | (voxel_array == 2), 1, 0)

print("Running TauFactor calculation on GPU (CUDA)...")
solver = tau.Solver(pore_network, device='cuda')
solver.solve()

print(f"\nTortuosity Factor (Z axis): {solver.tau.item():.4f}")
print("Calculation complete using GPU!")
