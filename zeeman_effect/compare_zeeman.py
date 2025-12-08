"""
Compare Penrose Lattice DOS with Different Zeeman Splitting

This script generates a comparison plot showing how the DOS changes
with different Zeeman field strengths.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix
import argparse
import os
import sys

# Import from penrose_zeeman.py
sys.path.append(os.path.dirname(__file__))
from penrose_zeeman import (
    load_penrose_data, 
    build_hamiltonian_with_zeeman,
    calculate_dos
)


def compare_zeeman_effects(iteration, zeeman_values, data_dir='../vertex_model/data', 
                          output_dir='results'):
    """
    Compare DOS for different Zeeman values
    
    Parameters:
        iteration: Penrose lattice iteration
        zeeman_values: list of Zeeman splitting values
        data_dir: directory containing Penrose lattice data
        output_dir: directory to save plots
    """
    print("=" * 70)
    print("ZEEMAN EFFECT COMPARISON")
    print("=" * 70)
    print(f"Iteration: {iteration}")
    print(f"Zeeman values: {zeeman_values}")
    print("=" * 70)
    
    # Load data
    filename = os.path.join(data_dir, f'penrose_lattice_iter{iteration}.npz')
    vertex_coords, edge_list, N, E = load_penrose_data(filename)
    print(f"\n✓ Loaded: {filename}")
    print(f"  Vertices: {N}, Edges: {E}")
    
    # Create figure
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()
    
    colors = plt.cm.viridis(np.linspace(0, 0.9, len(zeeman_values)))
    
    # Calculate and plot for each Zeeman value
    all_dos_data = []
    
    for idx, zeeman in enumerate(zeeman_values):
        print(f"\n[{idx+1}/{len(zeeman_values)}] Calculating Z = {zeeman:.3f}...")
        
        # Build Hamiltonian
        H = build_hamiltonian_with_zeeman(edge_list, N, t=-1.0, zeeman=zeeman)
        
        # Diagonalize
        H_dense = H.toarray()
        eigenvalues = np.linalg.eigvalsh(H_dense)
        
        # Calculate DOS
        energies, dos = calculate_dos(eigenvalues, energy_range=(-5, 5), num_bins=500)
        
        all_dos_data.append({
            'zeeman': zeeman,
            'energies': energies,
            'dos': dos,
            'eigenvalues': eigenvalues
        })
        
        # Plot in subplot
        ax = axes[min(idx, 3)]
        ax.plot(energies, dos, linewidth=2, color=colors[idx], 
                label=f'Z={zeeman:.2f}')
        ax.fill_between(energies, dos, alpha=0.3, color=colors[idx])
        ax.axvline(x=0, color='red', linestyle='--', linewidth=1, alpha=0.7)
        ax.set_xlabel('Energy', fontsize=11)
        ax.set_ylabel('DOS', fontsize=11)
        ax.set_title(f'DOS with Z={zeeman:.3f}', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend()
    
    plt.tight_layout()
    
    # Save individual comparison
    os.makedirs(output_dir, exist_ok=True)
    filename1 = os.path.join(output_dir, f'zeeman_comparison_iter{iteration}.png')
    plt.savefig(filename1, dpi=300, bbox_inches='tight')
    print(f"\n✓ Saved: {filename1}")
    plt.show()
    
    # Create overlay plot
    fig2, ax2 = plt.subplots(1, 1, figsize=(12, 8))
    
    for idx, data in enumerate(all_dos_data):
        ax2.plot(data['energies'], data['dos'], linewidth=2.5, 
                color=colors[idx], label=f"Z={data['zeeman']:.2f}", alpha=0.8)
    
    ax2.axvline(x=0, color='red', linestyle='--', linewidth=2, alpha=0.7, label='E=0')
    ax2.set_xlabel('Energy', fontsize=13)
    ax2.set_ylabel('Density of States', fontsize=13)
    ax2.set_title(f'DOS Comparison: Zeeman Effect (Iteration {iteration})', 
                  fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=11, loc='best')
    
    plt.tight_layout()
    
    filename2 = os.path.join(output_dir, f'zeeman_overlay_iter{iteration}.png')
    plt.savefig(filename2, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {filename2}")
    plt.show()
    
    # Analysis summary
    print("\n" + "=" * 70)
    print("ANALYSIS SUMMARY")
    print("=" * 70)
    
    for data in all_dos_data:
        zeeman = data['zeeman']
        eigenvalues = data['eigenvalues']
        
        # Count zero energy states
        zero_count = np.sum(np.abs(eigenvalues) < 1e-10)
        zero_pct = 100 * zero_count / len(eigenvalues)
        
        print(f"\nZ = {zeeman:.3f}:")
        print(f"  Energy range: [{eigenvalues.min():.4f}, {eigenvalues.max():.4f}]")
        print(f"  Zero energy states: {zero_count} ({zero_pct:.2f}%)")
        print(f"  Expected splitting: ±{zeeman/2:.4f}")
    
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(
        description='Compare Penrose lattice DOS with different Zeeman fields'
    )
    parser.add_argument('--iteration', '-i', type=int, default=4,
                        help='Penrose lattice iteration (default: 4)')
    parser.add_argument('--zeeman-values', '-z', type=float, nargs='+',
                        default=[0.0, 0.2, 0.5, 1.0],
                        help='List of Zeeman values to compare (default: 0.0 0.2 0.5 1.0)')
    parser.add_argument('--data-dir', type=str, default='../vertex_model/data',
                        help='Directory containing Penrose lattice data')
    parser.add_argument('--output-dir', type=str, default='results',
                        help='Output directory for plots (default: results)')
    
    args = parser.parse_args()
    
    compare_zeeman_effects(
        iteration=args.iteration,
        zeeman_values=args.zeeman_values,
        data_dir=args.data_dir,
        output_dir=args.output_dir
    )
    
    print("\n✅ Comparison completed!\n")


if __name__ == "__main__":
    main()
