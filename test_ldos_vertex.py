"""
Test script untuk LDOS dengan iteration kecil
"""
import sys
sys.path.insert(0, '.')

from vertex_model.penrose_tight_binding import PenroseTightBinding, print_separator
import numpy as np

# Inisialisasi model
tb_model = PenroseTightBinding(epsilon_0=0.0, t=1.0)

# Load data iteration 4
print_separator()
print("VERTEX MODEL - TIGHT BINDING LDOS TEST (Iteration 4)")
print_separator()

tb_model.load_from_numpy('vertex_model/data/penrose_lattice_iter4.npz')

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

# Plot LDOS
print("\n")
print_separator()
print("GENERATING LDOS PLOT")
print_separator()

tb_model.plot_ldos(n_points=200, delta=0.05, save_fig=True, 
                  filename='vertex_model/imgs/penrose_ldos_iter4.png')

# Calculate and display LDOS at E=0
print("\n")
print_separator()
print("LDOS ANALYSIS AT E=0")
print_separator()

# Create energy array centered at E=0
energies = np.array([0.0])
ldos_dict = tb_model.calculate_ldos(energies=energies, delta=0.05)
z_values = sorted(ldos_dict.keys())

print(f"\nLDOS values at E = 0 (δ = 0.05):")

for z in z_values:
    ldos_at_E0 = ldos_dict[z][0]
    print(f"  ρ_{z}(ω=0) = {ldos_at_E0:.6f}")

total_ldos_at_E0 = np.sum([ldos_dict[z][0] for z in z_values])
print(f"\n  Total ρ(ω=0) = {total_ldos_at_E0:.6f}")

print("\n")
print_separator()
print("✅ LDOS test completed!")
print_separator()
