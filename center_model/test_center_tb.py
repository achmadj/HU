"""
Test script for Center Model Tight Binding - using iteration 4 (smaller dataset)
Generates center model from vertex model and runs tight binding analysis
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import pickle
from center_penrose_tiling_fast import CenterPenroseLattice, print_separator
from center_penrose_tight_binding import CenterPenroseTightBinding

# Step 1: Generate center model dari vertex model iteration 4
print_separator()
print("STEP 1: GENERATING CENTER MODEL FROM VERTEX MODEL (Iteration 4)")
print_separator()

center_gen = CenterPenroseLattice()
center_gen.load_vertex_model('vertex_model/data/penrose_lattice_iter4.npz')
center_gen.detect_rhombi()
center_gen.compute_centers()
center_gen.build_dual_edges()

# Prepare data for tight binding
center_ids = np.arange(center_gen.N_center, dtype=np.int32)
center_coords = center_gen.centers
edge_list = np.array(center_gen.dual_edges, dtype=np.int32)

# Create data dictionary for tight binding
data_dict = {
    'vertices': {i: coord for i, coord in enumerate(center_coords)},
    'edges': {tuple(e): 1 for e in edge_list},
    'N': center_gen.N_center,
    'E': center_gen.E_center,
    'iteration': center_gen.iteration,
    'phi': (1 + np.sqrt(5)) / 2,
}

print(f"\n✓ Center model generated:")
print(f"  N_centers = {center_gen.N_center}")
print(f"  E_dual = {center_gen.E_center}")

# Step 2: Run tight binding analysis
print("\n")
print_separator()
print("STEP 2: TIGHT BINDING ANALYSIS")
print_separator()

tb_model = CenterPenroseTightBinding(epsilon_0=0.0, t=1.0)

# Load data directly from dict
print(f"\n[LOADING] Using generated center model data...")
tb_model.vertices = data_dict['vertices']
tb_model.edges = data_dict['edges']
tb_model.N = data_dict['N']
tb_model.E = data_dict['E']
tb_model.phi = data_dict['phi']
tb_model.iteration = data_dict['iteration']

print(f"  ✓ Loaded N={tb_model.N} vertices (center sites)")
print(f"  ✓ Loaded E={tb_model.E} edges (dual bonds)")
print(f"  ✓ Iteration: {tb_model.iteration}")

# Build Hamiltonian
tb_model.build_hamiltonian()

# Diagonalize
tb_model.diagonalize()

# Statistics
print("\n")
print_separator()
print("ENERGY STATISTICS")
print_separator()

stats = tb_model.get_statistics()
print(f"Mean energy:          {stats['mean_energy']:12.6f}")
print(f"Std deviation:        {stats['std_energy']:12.6f}")
print(f"Min energy:           {stats['min_energy']:12.6f}")
print(f"Max energy:           {stats['max_energy']:12.6f}")
print(f"Bandwidth:            {stats['bandwidth']:12.6f}")
print(f"Zero-energy states:   {stats['zero_energy_count']:12d}")

# Plot spectrum
print("\n")
print_separator()
print("GENERATING PLOTS")
print_separator()

tb_model.plot_energy_spectrum(save_fig=True, filename='center_model/imgs/center_energy_spectrum_iter4.png')
tb_model.plot_density_of_states(bins=min(tb_model.N, 100), save_fig=True, filename='center_model/imgs/center_dos_iter4.png')
tb_model.plot_integrated_dos(save_fig=True, filename='center_model/imgs/center_idos_iter4.png')

# Find state closest to E = 2
print("\n")
print_separator()
print("WAVEFUNCTION ANALYSIS")
print_separator()

target_energy = 2.0
energy_diff = np.abs(tb_model.eigenvalues - target_energy)
closest_idx = np.argmin(energy_diff)
closest_energy = tb_model.eigenvalues[closest_idx]

print(f"\nFinding state closest to E = {target_energy}:")
print(f"  State {closest_idx}: E = {closest_energy:.6f}")

# Plot wavefunction untuk state di E≈2
print("\n")
print_separator()
print("WAVEFUNCTION VISUALIZATION")
print_separator()

import matplotlib.pyplot as plt
fig, ax = plt.subplots(1, 1, figsize=(10, 10))

tb_model.plot_wavefunction(closest_idx, ax)

plt.tight_layout()
wavefunction_filename = 'center_model/imgs/center_wavefunctions_E2_iter4.png'
plt.savefig(wavefunction_filename, dpi=500, bbox_inches='tight')
print(f"\n  ✓ Saved wavefunction plot: {wavefunction_filename}")
plt.close()

print("\n")
print_separator()
print("✅ Center model tight binding analysis completed!")
print("📊 Energy spectrum, DOS, IDOS, and wavefunction plots generated")
print("   Files saved in center_model/imgs/")
print_separator()
print()
