"""
Comprehensive Zeeman Effect Analysis on Penrose Lattice

This script performs detailed analysis of magnetic properties including:
1. Total magnetization vs Zeeman field
2. Local spin density mapping
3. Edge state analysis
4. Probability density plots near degenerate energies
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from scipy.sparse import csr_matrix
import argparse
import time
import os
from typing import Tuple, List, Dict


def load_penrose_data(filename):
    """Load Penrose lattice data from .npz file"""
    data = np.load(filename)
    vertex_coords = data['vertex_coords']
    edge_list = data['edge_list']
    N = int(data['N'])
    E = int(data['E'])
    return vertex_coords, edge_list, N, E


def build_hamiltonian_with_zeeman(edge_list, N, t=-1.0, zeeman=0.0):
    """
    Build tight-binding Hamiltonian with Zeeman effect
    
    Basis ordering: [up_0, up_1, ..., up_{N-1}, down_0, down_1, ..., down_{N-1}]
    """
    dim = 2 * N
    row_indices = []
    col_indices = []
    data = []
    
    # Hopping terms (spin-conserving)
    for i, j in edge_list:
        # Spin-up sector
        row_indices.extend([i, j])
        col_indices.extend([j, i])
        data.extend([t, t])
        
        # Spin-down sector
        row_indices.extend([i + N, j + N])
        col_indices.extend([j + N, i + N])
        data.extend([t, t])
    
    # Zeeman term: E_up = -zeeman/2, E_down = +zeeman/2
    for i in range(N):
        row_indices.append(i)
        col_indices.append(i)
        data.append(-zeeman / 2.0)
        
        row_indices.append(i + N)
        col_indices.append(i + N)
        data.append(+zeeman / 2.0)
    
    H = csr_matrix((data, (row_indices, col_indices)), shape=(dim, dim))
    return H


def calculate_dos(eigenvalues, num_bins=500):
    """Calculate Density of States"""
    E_min = eigenvalues.min() - 0.5
    E_max = eigenvalues.max() + 0.5
    dos, bin_edges = np.histogram(eigenvalues, bins=num_bins, 
                                   range=(E_min, E_max), density=True)
    energies = (bin_edges[:-1] + bin_edges[1:]) / 2
    return energies, dos


def find_degenerate_energy_peaks(eigenvalues, dos_energies, dos, threshold_percentile=90):
    """
    Find energy values where degeneracy occurs (peaks in DOS)
    
    Returns:
        peak_energies: List of energies where DOS peaks occur
    """
    # Find peaks in DOS
    dos_threshold = np.percentile(dos, threshold_percentile)
    peak_indices = np.where(dos > dos_threshold)[0]
    
    if len(peak_indices) == 0:
        return []
    
    # Group consecutive peaks
    peak_groups = []
    current_group = [peak_indices[0]]
    
    for idx in peak_indices[1:]:
        if idx == current_group[-1] + 1:
            current_group.append(idx)
        else:
            peak_groups.append(current_group)
            current_group = [idx]
    peak_groups.append(current_group)
    
    # Get representative energy for each peak group
    peak_energies = []
    for group in peak_groups:
        peak_dos_max_idx = group[np.argmax(dos[group])]
        peak_energies.append(dos_energies[peak_dos_max_idx])
    
    return peak_energies


def calculate_total_magnetization(eigenvectors, N, num_filled_states=None):
    """
    Calculate total magnetization M_tot = <S_z^tot> = sum_i <S_z_i>
    
    For ground state (T=0), fill states from lowest energy up to Fermi level.
    If num_filled_states is None, use half-filling (N states filled).
    
    Returns:
        M_tot: Total magnetization
    """
    if num_filled_states is None:
        num_filled_states = N  # Half-filling
    
    M_tot = 0.0
    
    # Sum over all filled states
    for state_idx in range(num_filled_states):
        psi = eigenvectors[:, state_idx]
        
        # Spin-up contribution: +1/2 * probability
        spin_up_prob = np.sum(np.abs(psi[:N])**2)
        
        # Spin-down contribution: -1/2 * probability
        spin_down_prob = np.sum(np.abs(psi[N:])**2)
        
        M_tot += 0.5 * (spin_up_prob - spin_down_prob)
    
    return M_tot


def calculate_local_spin_density(eigenvectors, N, num_filled_states=None):
    """
    Calculate local spin density <S_z_i> at each site i
    
    Returns:
        local_Sz: Array of length N with <S_z_i> for each site
    """
    if num_filled_states is None:
        num_filled_states = N  # Half-filling
    
    local_Sz = np.zeros(N)
    
    for state_idx in range(num_filled_states):
        psi = eigenvectors[:, state_idx]
        
        # Local contribution from each site
        for i in range(N):
            spin_up_prob = np.abs(psi[i])**2
            spin_down_prob = np.abs(psi[i + N])**2
            local_Sz[i] += 0.5 * (spin_up_prob - spin_down_prob)
    
    return local_Sz


def find_edge_sites(vertex_coords, percentile=85):
    """
    Identify edge sites based on radial distance from center
    
    Returns:
        edge_mask: Boolean array, True for edge sites
    """
    center = np.mean(vertex_coords, axis=0)
    distances = np.linalg.norm(vertex_coords - center, axis=1)
    threshold = np.percentile(distances, percentile)
    edge_mask = distances > threshold
    return edge_mask


def plot_probability_density_map(vertex_coords, prob_density, title, 
                                  save_path=None, vmax=None, edge_mask=None):
    """
    Plot probability density |psi_i|^2 on Penrose lattice
    
    Parameters:
        vertex_coords: (N, 2) array of positions
        prob_density: (N,) array of probability at each site
        title: Plot title
        save_path: Path to save figure
        vmax: Maximum value for colorbar
        edge_mask: Boolean array marking edge sites
    """
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    
    # Plot edges (light gray background)
    # Note: We don't have edge_list here, so skip for now
    
    # Plot probability density
    if vmax is None:
        vmax = np.max(prob_density)
    
    scatter = ax.scatter(vertex_coords[:, 0], vertex_coords[:, 1], 
                        c=prob_density, cmap='hot', s=50, 
                        vmin=0, vmax=vmax, edgecolors='black', linewidths=0.3)
    
    # Mark edge sites if provided
    if edge_mask is not None:
        edge_coords = vertex_coords[edge_mask]
        ax.scatter(edge_coords[:, 0], edge_coords[:, 1], 
                  marker='o', s=80, facecolors='none', 
                  edgecolors='cyan', linewidths=2, label='Edge sites')
    
    plt.colorbar(scatter, ax=ax, label='Probability Density $|\\psi_i|^2$')
    ax.set_aspect('equal')
    ax.set_xlabel('x', fontsize=12)
    ax.set_ylabel('y', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    if edge_mask is not None:
        ax.legend(loc='upper right')
    ax.grid(True, alpha=0.2)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"  ✓ Saved: {save_path}")
    
    plt.close()


def plot_local_spin_density(vertex_coords, local_Sz, zeeman, iteration, save_dir):
    """Plot local spin density <S_z_i> map"""
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    
    # Use diverging colormap centered at zero
    vmax = max(abs(np.min(local_Sz)), abs(np.max(local_Sz)))
    
    scatter = ax.scatter(vertex_coords[:, 0], vertex_coords[:, 1], 
                        c=local_Sz, cmap='RdBu_r', s=50, 
                        vmin=-vmax, vmax=vmax, edgecolors='black', linewidths=0.3)
    
    plt.colorbar(scatter, ax=ax, label='Local Spin Density $\\langle S_z^i \\rangle$')
    ax.set_aspect('equal')
    ax.set_xlabel('x', fontsize=12)
    ax.set_ylabel('y', fontsize=12)
    ax.set_title(f'Local Spin Density Map\n(Iteration {iteration}, Z={zeeman:.3f})', 
                 fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.2)
    
    plt.tight_layout()
    
    filename = f'local_spin_density_Z{zeeman:.3f}_iter{iteration}.png'
    filepath = os.path.join(save_dir, filename)
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved: {filepath}")
    
    plt.close()


def plot_dos_with_peaks(energies, dos, peak_energies, zeeman, iteration, save_dir):
    """Plot DOS with marked peak energies"""
    fig, ax = plt.subplots(1, 1, figsize=(12, 6))
    
    ax.plot(energies, dos, linewidth=2, color='darkblue', label='DOS')
    ax.fill_between(energies, dos, alpha=0.3, color='blue')
    
    # Mark peak energies
    for i, E_peak in enumerate(peak_energies):
        ax.axvline(x=E_peak, color='red', linestyle='--', linewidth=1.5, 
                   alpha=0.7, label=f'Peak {i+1}: E={E_peak:.3f}' if i < 5 else '')
    
    ax.axvline(x=0, color='green', linestyle=':', linewidth=2, alpha=0.7, label='E=0')
    
    ax.set_xlabel('Energy', fontsize=12)
    ax.set_ylabel('DOS (states/energy)', fontsize=12)
    ax.set_title(f'Density of States with Peak Markers\n(Iteration {iteration}, Z={zeeman:.3f})', 
                 fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper right', fontsize=9)
    
    plt.tight_layout()
    
    filename = f'dos_peaks_Z{zeeman:.3f}_iter{iteration}.png'
    filepath = os.path.join(save_dir, filename)
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved: {filepath}")
    
    plt.close()


def analyze_states_near_energy(eigenvalues, eigenvectors, vertex_coords, 
                               target_energy, energy_window, N, zeeman, 
                               iteration, save_dir, edge_mask):
    """
    Analyze and plot probability density for states near target energy
    """
    # Find states within energy window
    mask = np.abs(eigenvalues - target_energy) < energy_window
    state_indices = np.where(mask)[0]
    
    if len(state_indices) == 0:
        print(f"  ⚠ No states found near E={target_energy:.3f}")
        return
    
    print(f"\n  Found {len(state_indices)} states near E={target_energy:.3f} ± {energy_window:.3f}")
    print(f"    State indices: {state_indices[:10]}{'...' if len(state_indices) > 10 else ''}")
    print(f"    Energies: {eigenvalues[state_indices[:5]]}")
    
    # Plot probability density for first few states
    num_states_to_plot = min(3, len(state_indices))
    
    for i in range(num_states_to_plot):
        state_idx = state_indices[i]
        psi = eigenvectors[:, state_idx]
        
        # Total probability density (spin-up + spin-down)
        prob_density = np.abs(psi[:N])**2 + np.abs(psi[N:])**2
        
        # Check edge localization
        edge_prob = np.sum(prob_density[edge_mask])
        total_prob = np.sum(prob_density)
        edge_percentage = 100 * edge_prob / total_prob
        
        title = f'State #{state_idx}: E={eigenvalues[state_idx]:.4f}\n'
        title += f'Edge localization: {edge_percentage:.1f}% | Z={zeeman:.3f}'
        
        filename = f'prob_density_E{target_energy:.2f}_state{i}_Z{zeeman:.3f}_iter{iteration}.png'
        filepath = os.path.join(save_dir, filename)
        
        plot_probability_density_map(vertex_coords, prob_density, title, 
                                    filepath, edge_mask=edge_mask)
        
        print(f"    State {state_idx}: E={eigenvalues[state_idx]:.4f}, "
              f"Edge localization={edge_percentage:.1f}%")


def run_single_zeeman_analysis(zeeman, iteration, data_dir, save_dir, edge_percentile=85):
    """Run complete analysis for a single Zeeman value"""
    
    print(f"\n{'='*70}")
    print(f"ANALYZING ZEEMAN = {zeeman:.3f}")
    print(f"{'='*70}")
    
    # Load data
    print(f"\n[1/6] Loading data...")
    filename = os.path.join(data_dir, f'penrose_lattice_iter{iteration}.npz')
    vertex_coords, edge_list, N, E = load_penrose_data(filename)
    print(f"  ✓ Vertices: {N}, Edges: {E}")
    
    # Build Hamiltonian
    print(f"\n[2/6] Building Hamiltonian...")
    H = build_hamiltonian_with_zeeman(edge_list, N, t=-1.0, zeeman=zeeman)
    dim = 2 * N
    print(f"  ✓ Hamiltonian size: {dim} × {dim}")
    
    # Diagonalize
    print(f"\n[3/6] Diagonalizing...")
    t0 = time.time()
    H_dense = H.toarray()
    eigenvalues, eigenvectors = np.linalg.eigh(H_dense)
    t1 = time.time()
    print(f"  ✓ Time: {(t1-t0):.3f} seconds")
    print(f"  ✓ Energy range: [{eigenvalues.min():.4f}, {eigenvalues.max():.4f}]")
    
    # Calculate DOS and find peaks
    print(f"\n[4/6] Analyzing DOS...")
    energies, dos = calculate_dos(eigenvalues)
    peak_energies = find_degenerate_energy_peaks(eigenvalues, energies, dos, 
                                                  threshold_percentile=90)
    print(f"  ✓ Found {len(peak_energies)} DOS peaks")
    if len(peak_energies) > 0:
        print(f"    Peak energies: {[f'{E:.3f}' for E in peak_energies[:5]]}")
    
    # Plot DOS with peaks
    plot_dos_with_peaks(energies, dos, peak_energies, zeeman, iteration, save_dir)
    
    # Calculate magnetization
    print(f"\n[5/6] Calculating magnetic properties...")
    num_filled = N  # Half-filling
    M_tot = calculate_total_magnetization(eigenvectors, N, num_filled)
    print(f"  ✓ Total magnetization: M_tot = {M_tot:.6f}")
    
    # Calculate local spin density
    local_Sz = calculate_local_spin_density(eigenvectors, N, num_filled)
    print(f"  ✓ Local spin density range: [{local_Sz.min():.6f}, {local_Sz.max():.6f}]")
    
    # Plot local spin density map
    plot_local_spin_density(vertex_coords, local_Sz, zeeman, iteration, save_dir)
    
    # Find edge sites
    edge_mask = find_edge_sites(vertex_coords, percentile=edge_percentile)
    num_edge = np.sum(edge_mask)
    print(f"  ✓ Edge sites: {num_edge} ({100*num_edge/N:.1f}%)")
    
    # Analyze states near each peak energy
    print(f"\n[6/6] Analyzing states near peak energies...")
    energy_window = 0.05  # Energy window for finding nearby states
    
    # Always analyze states near E=0
    analyze_states_near_energy(eigenvalues, eigenvectors, vertex_coords, 
                              0.0, energy_window, N, zeeman, iteration, 
                              save_dir, edge_mask)
    
    # Analyze states near DOS peaks (up to 3 peaks)
    for i, E_peak in enumerate(peak_energies[:3]):
        if abs(E_peak) > 0.1:  # Skip if too close to E=0 (already analyzed)
            analyze_states_near_energy(eigenvalues, eigenvectors, vertex_coords, 
                                      E_peak, energy_window, N, zeeman, 
                                      iteration, save_dir, edge_mask)
    
    return M_tot, local_Sz, eigenvalues


def plot_magnetization_vs_zeeman(zeeman_values, magnetizations, iteration, save_dir):
    """Plot total magnetization vs Zeeman field"""
    fig, ax = plt.subplots(1, 1, figsize=(10, 7))
    
    ax.plot(zeeman_values, magnetizations, 'o-', linewidth=2, 
            markersize=8, color='darkblue', label='$M_{tot}$')
    
    ax.set_xlabel('Zeeman Field $h^z$', fontsize=13)
    ax.set_ylabel('Total Magnetization $M_{tot}$', fontsize=13)
    ax.set_title(f'Magnetization vs Zeeman Field\n(Iteration {iteration})', 
                 fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=12)
    
    plt.tight_layout()
    
    filename = f'magnetization_vs_zeeman_iter{iteration}.png'
    filepath = os.path.join(save_dir, filename)
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved: {filepath}")
    
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description='Comprehensive Zeeman effect analysis on Penrose lattice'
    )
    parser.add_argument('--iteration', '-i', type=int, default=4,
                        help='Penrose lattice iteration (default: 4)')
    parser.add_argument('--zeeman-values', '-z', type=float, nargs='+',
                        default=[0.0, 0.5, 1.0, 3.0],
                        help='List of Zeeman values to analyze (default: 0.0 0.5 1.0 3.0)')
    parser.add_argument('--data-dir', type=str, default='../vertex_model/data',
                        help='Directory containing Penrose lattice data')
    parser.add_argument('--output-dir', type=str, default='results_analysis',
                        help='Output directory for plots')
    parser.add_argument('--edge-percentile', type=float, default=85,
                        help='Percentile for edge site identification (default: 85)')
    
    args = parser.parse_args()
    
    print("="*70)
    print("COMPREHENSIVE ZEEMAN EFFECT ANALYSIS")
    print("="*70)
    print(f"\nParameters:")
    print(f"  Iteration: {args.iteration}")
    print(f"  Zeeman values: {args.zeeman_values}")
    print(f"  Data directory: {args.data_dir}")
    print(f"  Output directory: {args.output_dir}")
    print(f"  Edge percentile: {args.edge_percentile}")
    print("="*70)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Run analysis for each Zeeman value
    magnetizations = []
    
    for zeeman in args.zeeman_values:
        M_tot, local_Sz, eigenvalues = run_single_zeeman_analysis(
            zeeman, args.iteration, args.data_dir, args.output_dir, 
            args.edge_percentile
        )
        magnetizations.append(M_tot)
    
    # Plot M vs h^z
    print(f"\n{'='*70}")
    print(f"GENERATING MAGNETIZATION CURVE")
    print(f"{'='*70}")
    plot_magnetization_vs_zeeman(args.zeeman_values, magnetizations, 
                                 args.iteration, args.output_dir)
    
    # Summary
    print(f"\n{'='*70}")
    print("✅ ANALYSIS COMPLETED")
    print(f"{'='*70}")
    print(f"\nResults saved to: {args.output_dir}/")
    print(f"\nMagnetization summary:")
    for z, M in zip(args.zeeman_values, magnetizations):
        print(f"  Z={z:.3f}: M_tot={M:.6f}")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
