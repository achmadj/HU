"""
Test script untuk vertex model tight binding dengan iteration kecil
"""
import sys
sys.path.insert(0, '.')

from vertex_model.penrose_tight_binding import PenroseTightBinding, print_separator
import numpy as np
from collections import Counter

# Inisialisasi model
tb_model = PenroseTightBinding(epsilon_0=0.0, t=1.0)

# Load data iteration 3 (lebih kecil)
print_separator()
print("VERTEX MODEL - TIGHT BINDING (Iteration 3)")
print_separator()

tb_model.load_from_numpy('vertex_model/data/penrose_lattice_iter3.npz')

# Build Hamiltonian
tb_model.build_hamiltonian()

# Diagonalisasi
tb_model.diagonalize()

# Statistik
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

# State fraction analysis
target_energy = 0.0
energy_threshold = 0.01

states_near_E0 = np.sum(np.abs(tb_model.eigenvalues - target_energy) < energy_threshold)
total_states = tb_model.N
fraction_E0 = states_near_E0 / total_states

print("\n")
print_separator()
print("STATE FRACTION ANALYSIS AT E ≈ 0")
print_separator()

print(f"\nState Fraction Analysis (E = {target_energy} ± {energy_threshold}):")
print(f"  N (states near E=0):  {states_near_E0}")
print(f"  N₀ (total states):    {total_states}")
print(f"  f = N/N₀:             {fraction_E0:.6f} ({fraction_E0*100:.4f}%)")

print("\n")
print_separator()
print("✅ Analysis completed!")
print_separator()
