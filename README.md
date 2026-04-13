# Electrode Tortuosity Analysis (GPU-Accelerated)

This repository contains the workflow for quantifying the **geometric tortuosity factor ($\tau$)** and **CBD transport penalties** in battery electrode microstructures.

##  Project Overview
Developed for the **Politecnico di Torino Battery Project**, this tool uses the **TauFactor** solver to analyze segmented 3D TIFF volumes from NREL. 

### Key Features:
* **GPU Acceleration:** Optimized for NVIDIA CUDA (tested on RTX 3050).
* **Comparative Analysis:** Calculates tortuosity for both "Pores-Only" and "Pores + CBD" scenarios.
* **Batch Processing:** Automatically processes multiple samples and exports data to CSV.

##  Requirements
To run these scripts, you need a Python environment with:
* `taufactor`
* `pytorch` (with CUDA support)
* `tifffile`
* `pandas`
* `numpy`

##  Usage
1. **Data:** Create a folder named `nrel_data/` in the root directory.
2. **Download:** Place your segmented `.tif` files (NMC or Graphite) in that folder.
3. **Run:** Execute `scripts/batch_processor.py`.
4. **Output:** Results will be saved in the `results/` folder as a CSV.

---
*Note: The raw .tif datasets (e.g., Toda NMC532 and Conoco Phillips A12) are not included in this repository due to file size limits.*
