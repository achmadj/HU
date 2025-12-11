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
    
    # Rotate vertices 18 degrees clockwise (negative angle)
    angle = -18 * np.pi / 180  # Convert to radians, negative for clockwise
    cos_a, sin_a = np.cos(angle), np.sin(angle)
    rotation_matrix = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
    vertex_coords = vertex_coords @ rotation_matrix.T
    
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


def compute_bipartite_sublattices(edge_list, N):
    """
    Compute bipartite sublattice assignment using BFS.
    Penrose lattice is bipartite, so it can be divided into two sublattices
    where no edge connects sites in the same sublattice.
    
    Returns:
        sublattice_labels: Array of length N with 'A' or 'B' for each site
    """
    from collections import deque
    
    # Build adjacency list
    adjacency = {i: set() for i in range(N)}
    for (i, j) in edge_list:
        adjacency[i].add(j)
        adjacency[j].add(i)
    
    # Initialize sublattice map
    sublattice_labels = [''] * N
    queue = deque()
    
    # Start from site 0
    sublattice_labels[0] = 'A'
    queue.append(0)
    
    # BFS traversal
    while queue:
        u = queue.popleft()
        label_u = sublattice_labels[u]
        opposite_label = 'B' if label_u == 'A' else 'A'
        
        for v in adjacency[u]:
            if sublattice_labels[v] == '':
                sublattice_labels[v] = opposite_label
                queue.append(v)
    
    # Handle disconnected components
    for i in range(N):
        if sublattice_labels[i] == '':
            sublattice_labels[i] = 'A'
            queue.append(i)
            
            while queue:
                u = queue.popleft()
                label_u = sublattice_labels[u]
                opposite_label = 'B' if label_u == 'A' else 'A'
                
                for v in adjacency[u]:
                    if sublattice_labels[v] == '':
                        sublattice_labels[v] = opposite_label
                        queue.append(v)
    
    return np.array(sublattice_labels)


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


def plot_probability_density_map(vertex_coords, prob_density, sublattice_labels, title, 
                                  save_path=None, vmax=None, edge_mask=None,
                                  threshold_ratio=1e-9):
    """
    Plot probability density |psi_i|^2 on Penrose lattice with heatmap
    
    Parameters:
        vertex_coords: (N, 2) array of positions
        prob_density: (N,) array of probability at each site
        sublattice_labels: Array of 'A' or 'B' for each site (not used in heatmap)
        title: Plot title
        save_path: Path to save figure
        vmax: Maximum value for colorbar
        edge_mask: Boolean array marking edge sites
        threshold_ratio: Sites with prob_density < threshold_ratio * max are not plotted
    """
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    
    # Plot probability density
    if vmax is None:
        vmax = np.max(prob_density)
    
    # Filter out sites with values close to zero
    threshold = threshold_ratio * vmax
    significant_mask = prob_density >= threshold
    
    # Plot sites with significant probability using heatmap
    scatter = ax.scatter(vertex_coords[significant_mask, 0], vertex_coords[significant_mask, 1], 
                        c=prob_density[significant_mask], cmap='hot', s=20, 
                        vmin=0, vmax=vmax, edgecolors='black', linewidths=0.2)
    
    plt.colorbar(scatter, ax=ax, label='Probability Density $|\\psi_i|^2$')
    ax.set_aspect('equal')
    ax.set_xlabel('x', fontsize=12)
    ax.set_ylabel('y', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.2)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"  ✓ Saved: {save_path}")
    
    plt.close()


def plot_local_spin_density(vertex_coords, local_Sz, zeeman, iteration, save_dir, 
                            threshold_ratio=1e-9):
    """Plot local spin density <S_z_i> map
    
    Parameters:
        threshold_ratio: Sites with |local_Sz| < threshold_ratio * max are not plotted
    """
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    
    # Use diverging colormap centered at zero
    vmax = max(abs(np.min(local_Sz)), abs(np.max(local_Sz)))
    
    # Filter out sites with values close to zero
    threshold = threshold_ratio * vmax
    significant_mask = np.abs(local_Sz) >= threshold
    
    # Plot only significant sites
    scatter = ax.scatter(vertex_coords[significant_mask, 0], vertex_coords[significant_mask, 1], 
                        c=local_Sz[significant_mask], cmap='RdBu_r', s=20, 
                        vmin=-vmax, vmax=vmax, edgecolors='black', linewidths=0.2)
    
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
    """Plot DOS with two main peaks marked from eigenvalue degeneracy"""
    fig, ax = plt.subplots(1, 1, figsize=(12, 6))
    
    ax.plot(energies, dos, linewidth=2, color='darkblue')
    ax.fill_between(energies, dos, alpha=0.3, color='blue')
    
    # Sort peaks by DOS value to get the two highest peaks
    # peak_energies is already from find_degenerate_energy_peaks
    if len(peak_energies) >= 2:
        # Get DOS values at each peak energy
        peak_dos_values = []
        for E_peak in peak_energies:
            idx = np.argmin(np.abs(energies - E_peak))
            peak_dos_values.append(dos[idx])
        
        # Sort by DOS value (highest first)
        sorted_indices = np.argsort(peak_dos_values)[::-1]
        top_peaks = [peak_energies[i] for i in sorted_indices[:2]]
        top_peaks.sort()  # Sort by energy (lower first)
        
        # Mark first peak (lower energy)
        colors = ['red', 'blue']
        for i, E_peak in enumerate(top_peaks):
            ax.axvline(x=E_peak, color=colors[i], linestyle='--', linewidth=2, alpha=0.8, 
                       label=f'Peak {i+1}: E = {E_peak:.4f}')
    elif len(peak_energies) == 1:
        ax.axvline(x=peak_energies[0], color='red', linestyle='--', linewidth=2, alpha=0.8, 
                   label=f'Peak: E = {peak_energies[0]:.4f}')
    
    ax.set_xlabel('Energy', fontsize=12)
    ax.set_ylabel('DOS (states/energy)', fontsize=12)
    ax.set_title(f'Density of States\n(Iteration {iteration}, Z={zeeman:.3f})', 
                 fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper right', fontsize=11)
    
    plt.tight_layout()
    
    filename = f'dos_Z{zeeman:.3f}_iter{iteration}.png'
    filepath = os.path.join(save_dir, filename)
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved: {filepath}")
    
    plt.close()


def find_highest_edge_localized_states(eigenvalues, eigenvectors, N, edge_mask):
    """
    Loop through all eigenstates and find states with highest edge localization
    in E- (negative energy) and E+ (positive energy) regions.
    
    Returns:
        best_neg_idx: State index with highest edge localization at E < 0
        best_pos_idx: State index with highest edge localization at E > 0
        edge_localizations: Array of edge localization percentages for all states
    """
    num_states = len(eigenvalues)
    edge_localizations = np.zeros(num_states)
    
    # Calculate edge localization for all states
    for state_idx in range(num_states):
        psi = eigenvectors[:, state_idx]
        prob_density = np.abs(psi[:N])**2 + np.abs(psi[N:])**2
        edge_prob = np.sum(prob_density[edge_mask])
        total_prob = np.sum(prob_density)
        edge_localizations[state_idx] = 100 * edge_prob / total_prob
    
    # Find best state in E- region (negative energies)
    neg_mask = eigenvalues < 0
    if np.any(neg_mask):
        neg_indices = np.where(neg_mask)[0]
        best_neg_local_idx = np.argmax(edge_localizations[neg_mask])
        best_neg_idx = neg_indices[best_neg_local_idx]
    else:
        best_neg_idx = None
    
    # Find best state in E+ region (positive energies)
    pos_mask = eigenvalues > 0
    if np.any(pos_mask):
        pos_indices = np.where(pos_mask)[0]
        best_pos_local_idx = np.argmax(edge_localizations[pos_mask])
        best_pos_idx = pos_indices[best_pos_local_idx]
    else:
        best_pos_idx = None
    
    return best_neg_idx, best_pos_idx, edge_localizations


def plot_state_probability_density(state_idx, eigenvalues, eigenvectors, vertex_coords, 
                                   N, zeeman, iteration, save_dir, edge_mask, 
                                   sublattice_labels, edge_localizations, label):
    """
    Plot probability density for a specific state
    """
    actual_energy = eigenvalues[state_idx]
    psi = eigenvectors[:, state_idx]
    
    # Total probability density (spin-up + spin-down)
    prob_density = np.abs(psi[:N])**2 + np.abs(psi[N:])**2
    
    # Calculate spin polarization for this state
    spin_up_prob = np.sum(np.abs(psi[:N])**2)
    spin_down_prob = np.sum(np.abs(psi[N:])**2)
    polarization = (spin_up_prob - spin_down_prob) / (spin_up_prob + spin_down_prob)
    
    edge_percentage = edge_localizations[state_idx]
    
    title = f'State #{state_idx} ({label}): E={actual_energy:.4f}\n'
    title += f'Edge localization: {edge_percentage:.1f}% | Z={zeeman:.3f}'
    
    filename = f'prob_density_{label}_E{actual_energy:.4f}_Z{zeeman:.3f}_iter{iteration}.png'
    filepath = os.path.join(save_dir, filename)
    
    plot_probability_density_map(vertex_coords, prob_density, sublattice_labels, title, 
                                filepath, edge_mask=edge_mask)
    
    print(f"  ✓ {label} State {state_idx}: E={actual_energy:.4f}")
    print(f"    Edge localization: {edge_percentage:.1f}%")
    print(f"    Spin polarization: P = {polarization:.6f} (↑: {spin_up_prob:.4f}, ↓: {spin_down_prob:.4f})")


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
    
    # Compute bipartite sublattice labels
    sublattice_labels = compute_bipartite_sublattices(edge_list, N)
    num_A = np.sum(sublattice_labels == 'A')
    num_B = np.sum(sublattice_labels == 'B')
    print(f"  ✓ Bipartite: Sublattice A={num_A}, Sublattice B={num_B}")
    
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
    
    # Analyze states: find highest edge localized states in E- and E+ regions
    print(f"\n[6/6] Finding states with highest edge localization...")
    
    # Calculate edge localization for all states
    best_neg_idx, best_pos_idx, edge_localizations = find_highest_edge_localized_states(
        eigenvalues, eigenvectors, N, edge_mask
    )
    
    print(f"  ✓ Calculated edge localization for all {len(eigenvalues)} states")
    
    # Plot state with highest edge localization in E- region
    if best_neg_idx is not None:
        print(f"\n  [E- region] Best edge-localized state:")
        plot_state_probability_density(best_neg_idx, eigenvalues, eigenvectors, vertex_coords,
                                       N, zeeman, iteration, save_dir, edge_mask,
                                       sublattice_labels, edge_localizations, "E-")
    
    # Plot state with highest edge localization in E+ region
    if best_pos_idx is not None:
        print(f"\n  [E+ region] Best edge-localized state:")
        plot_state_probability_density(best_pos_idx, eigenvalues, eigenvectors, vertex_coords,
                                       N, zeeman, iteration, save_dir, edge_mask,
                                       sublattice_labels, edge_localizations, "E+")
    
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
