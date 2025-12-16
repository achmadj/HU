"""
Edge Localization Analysis for Penrose Lattice (Vertex Model)

This script loops through all eigenstates and finds the state with the highest
edge localization (probability density concentrated at edge sites).

No Zeeman effect - pure tight-binding model.
"""

import numpy as np
import matplotlib.pyplot as plt
import argparse
import time
import os
from collections import deque


def load_penrose_data(filename):
    """Load Penrose lattice data from .npz file"""
    data = np.load(filename)
    vertex_coords = data['vertex_coords']
    edge_list = data['edge_list']
    N = int(data['N'])
    E = int(data['E'])
    
    # Rotate vertices 18 degrees clockwise (negative angle)
    angle = -18 * np.pi / 180
    cos_a, sin_a = np.cos(angle), np.sin(angle)
    rotation_matrix = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
    vertex_coords = vertex_coords @ rotation_matrix.T
    
    return vertex_coords, edge_list, N, E


def build_tight_binding_hamiltonian(edge_list, N, t=-1.0):
    """
    Build tight-binding Hamiltonian (no spin, no Zeeman)
    
    H = -t * sum_{<i,j>} (c_i^dagger c_j + h.c.)
    """
    H = np.zeros((N, N))
    
    for i, j in edge_list:
        H[i, j] = t
        H[j, i] = t
    
    return H


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


def compute_bipartite_sublattices(edge_list, N):
    """
    Compute bipartite sublattice assignment using BFS.
    
    Returns:
        sublattice_map: Dict mapping site_id -> 'A' or 'B'
    """
    # Build adjacency list
    adjacency = {i: set() for i in range(N)}
    for (i, j) in edge_list:
        adjacency[i].add(j)
        adjacency[j].add(i)
    
    # Initialize sublattice map
    sublattice_map = {}
    queue = deque()
    
    # Start from site 0
    sublattice_map[0] = 'A'
    queue.append(0)
    
    # BFS traversal
    while queue:
        u = queue.popleft()
        label_u = sublattice_map[u]
        opposite_label = 'B' if label_u == 'A' else 'A'
        
        for v in adjacency[u]:
            if v not in sublattice_map:
                sublattice_map[v] = opposite_label
                queue.append(v)
    
    # Handle disconnected components
    for i in range(N):
        if i not in sublattice_map:
            sublattice_map[i] = 'A'
            queue.append(i)
            
            while queue:
                u = queue.popleft()
                label_u = sublattice_map[u]
                opposite_label = 'B' if label_u == 'A' else 'A'
                
                for v in adjacency[u]:
                    if v not in sublattice_map:
                        sublattice_map[v] = opposite_label
                        queue.append(v)
    
    return sublattice_map


def calculate_edge_localization_all_states(eigenvalues, eigenvectors, edge_mask):
    """
    Loop through all eigenstates and calculate edge localization percentage
    
    Returns:
        edge_localizations: Array of edge localization percentages for all states
    """
    num_states = len(eigenvalues)
    edge_localizations = np.zeros(num_states)
    
    for state_idx in range(num_states):
        psi = eigenvectors[:, state_idx]
        prob_density = np.abs(psi)**2
        edge_prob = np.sum(prob_density[edge_mask])
        total_prob = np.sum(prob_density)
        edge_localizations[state_idx] = 100 * edge_prob / total_prob
    
    return edge_localizations


def find_best_edge_localized_states(eigenvalues, edge_localizations, top_n=10):
    """
    Find states with highest edge localization in E- and E+ regions
    
    Returns:
        best_neg: List of (state_idx, energy, edge_loc%) for E < 0
        best_pos: List of (state_idx, energy, edge_loc%) for E > 0
        best_overall: List of (state_idx, energy, edge_loc%) overall
    """
    num_states = len(eigenvalues)
    
    # Overall best
    sorted_overall = np.argsort(edge_localizations)[::-1]
    best_overall = [(idx, eigenvalues[idx], edge_localizations[idx]) 
                    for idx in sorted_overall[:top_n]]
    
    # E- region (negative energies)
    neg_mask = eigenvalues < 0
    neg_indices = np.where(neg_mask)[0]
    if len(neg_indices) > 0:
        neg_sorted = neg_indices[np.argsort(edge_localizations[neg_indices])[::-1]]
        best_neg = [(idx, eigenvalues[idx], edge_localizations[idx]) 
                    for idx in neg_sorted[:top_n]]
    else:
        best_neg = []
    
    # E+ region (positive energies)
    pos_mask = eigenvalues > 0
    pos_indices = np.where(pos_mask)[0]
    if len(pos_indices) > 0:
        pos_sorted = pos_indices[np.argsort(edge_localizations[pos_indices])[::-1]]
        best_pos = [(idx, eigenvalues[idx], edge_localizations[idx]) 
                    for idx in pos_sorted[:top_n]]
    else:
        best_pos = []
    
    return best_neg, best_pos, best_overall


def plot_edge_localization_vs_energy(eigenvalues, edge_localizations, iteration, save_dir):
    """Plot edge localization percentage vs energy"""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    scatter = ax.scatter(eigenvalues, edge_localizations, c=edge_localizations, 
                         cmap='hot', s=10, alpha=0.7)
    
    ax.axvline(x=0, color='gray', linestyle='--', linewidth=1, alpha=0.7)
    ax.axhline(y=15, color='blue', linestyle=':', linewidth=1, alpha=0.7, 
               label='Expected random (15%)')
    
    plt.colorbar(scatter, ax=ax, label='Edge Localization (%)')
    ax.set_xlabel('Energy E', fontsize=12)
    ax.set_ylabel('Edge Localization (%)', fontsize=12)
    ax.set_title(f'Edge Localization vs Energy\n(Iteration {iteration}, N={len(eigenvalues)} states)', 
                 fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper right')
    
    plt.tight_layout()
    
    filename = f'edge_localization_vs_energy_iter{iteration}.png'
    filepath = os.path.join(save_dir, filename)
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved: {filepath}")
    
    plt.close()


def plot_probability_density(state_idx, eigenvalues, eigenvectors, vertex_coords, 
                             edge_list, edge_localizations, iteration, save_dir, 
                             sublattice_map, N, label=""):
    """Plot probability density for a specific state with bipartite coloring (red/blue)"""
    energy = eigenvalues[state_idx]
    psi = eigenvectors[:, state_idx]
    prob_density = np.abs(psi)**2
    edge_loc = edge_localizations[state_idx]
    
    # Calculate participation ratio
    participation_ratio = 1.0 / np.sum(prob_density**2)
    
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # Draw edges (lattice structure)
    for i, j in edge_list:
        x = [vertex_coords[i, 0], vertex_coords[j, 0]]
        y = [vertex_coords[i, 1], vertex_coords[j, 1]]
        ax.plot(x, y, color='gray', linewidth=0.3, alpha=0.3, zorder=1)
    
    # Threshold for finite probability density
    size_threshold = 1e-16
    color_threshold = 1e-15
    
    # Size based on probability
    sizes = np.where(prob_density < size_threshold, 0.01,
                    np.where(prob_density >= color_threshold, 50.0, 5.0))
    
    # Plot all sites with heatmap coloring first (background)
    x_coords = vertex_coords[:, 0]
    y_coords = vertex_coords[:, 1]
    
    sc = ax.scatter(x_coords, y_coords, s=sizes, c=prob_density, 
                   cmap='hot', alpha=0.8, edgecolors='black', 
                   linewidth=0.3, zorder=2, vmin=0, vmax=np.max(prob_density))
    
    # Plot sites with finite probability using bipartite coloring (red/blue)
    high_prob_mask = prob_density >= color_threshold
    if np.any(high_prob_mask):
        high_prob_indices = np.where(high_prob_mask)[0]
        
        # Sublattice A (red)
        sublattice_A_mask = np.array([sublattice_map[i] == 'A' for i in high_prob_indices])
        if np.any(sublattice_A_mask):
            indices_A = high_prob_indices[sublattice_A_mask]
            ax.scatter(x_coords[indices_A], y_coords[indices_A], 
                      s=sizes[indices_A], c='red', 
                      alpha=1.0, edgecolors='black', linewidth=0.5, zorder=3)
        
        # Sublattice B (blue)
        sublattice_B_mask = np.array([sublattice_map[i] == 'B' for i in high_prob_indices])
        if np.any(sublattice_B_mask):
            indices_B = high_prob_indices[sublattice_B_mask]
            ax.scatter(x_coords[indices_B], y_coords[indices_B], 
                      s=sizes[indices_B], c='blue', 
                      alpha=1.0, edgecolors='black', linewidth=0.5, zorder=3)
    
    ax.set_aspect('equal')
    ax.axis('off')
    
    title = f'State #{state_idx}: E = {energy:.6f}\n'
    title += f'Edge Loc: {edge_loc:.1f}% | PR={participation_ratio:.1f}/{N}'
    if label:
        title = f'{label}\n' + title
    ax.set_title(title, fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    
    label_str = f"_{label}" if label else ""
    filename = f'prob_density{label_str}_state{state_idx}_E{energy:.4f}_iter{iteration}.png'
    filepath = os.path.join(save_dir, filename)
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved: {filepath}")
    
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description='Find eigenstates with highest edge localization (vertex model)'
    )
    parser.add_argument('--iteration', '-i', type=int, default=4,
                        help='Penrose lattice iteration (default: 4)')
    parser.add_argument('--hopping', '-t', type=float, default=-1.0,
                        help='Hopping parameter (default: -1.0)')
    parser.add_argument('--data-dir', type=str, default='data',
                        help='Directory containing Penrose lattice data')
    parser.add_argument('--output-dir', type=str, default='imgs/edge_analysis',
                        help='Output directory for plots')
    parser.add_argument('--edge-percentile', type=float, default=85,
                        help='Percentile for edge site identification (default: 85)')
    parser.add_argument('--top-n', type=int, default=5,
                        help='Number of top states to display/plot (default: 5)')
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("EDGE LOCALIZATION ANALYSIS - VERTEX MODEL")
    print("=" * 70)
    print(f"\nParameters:")
    print(f"  Iteration: {args.iteration}")
    print(f"  Hopping parameter: {args.hopping}")
    print(f"  Edge percentile: {args.edge_percentile}")
    print(f"  Top N states: {args.top_n}")
    print("=" * 70)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load data
    print(f"\n[1/5] Loading Penrose lattice data...")
    filename = os.path.join(args.data_dir, f'penrose_lattice_iter{args.iteration}.npz')
    
    if not os.path.exists(filename):
        print(f"\n✗ ERROR: File not found: {filename}")
        return
    
    vertex_coords, edge_list, N, E = load_penrose_data(filename)
    print(f"  ✓ Vertices: {N}")
    print(f"  ✓ Edges: {E}")
    
    # Build Hamiltonian
    print(f"\n[2/5] Building tight-binding Hamiltonian...")
    H = build_tight_binding_hamiltonian(edge_list, N, t=args.hopping)
    print(f"  ✓ Hamiltonian size: {N} × {N}")
    
    # Diagonalize
    print(f"\n[3/5] Diagonalizing Hamiltonian...")
    t0 = time.time()
    eigenvalues, eigenvectors = np.linalg.eigh(H)
    t1 = time.time()
    print(f"  ✓ Time: {(t1-t0):.3f} seconds")
    print(f"  ✓ Energy range: [{eigenvalues.min():.4f}, {eigenvalues.max():.4f}]")
    
    # Find edge sites
    print(f"\n[4/5] Calculating edge localization for all {N} states...")
    edge_mask = find_edge_sites(vertex_coords, percentile=args.edge_percentile)
    num_edge = np.sum(edge_mask)
    print(f"  ✓ Edge sites: {num_edge} ({100*num_edge/N:.1f}%)")
    
    # Calculate edge localization for ALL states
    t0 = time.time()
    edge_localizations = calculate_edge_localization_all_states(
        eigenvalues, eigenvectors, edge_mask
    )
    t1 = time.time()
    print(f"  ✓ Calculated edge localization for all states in {(t1-t0):.3f} s")
    
    # Find best states
    best_neg, best_pos, best_overall = find_best_edge_localized_states(
        eigenvalues, edge_localizations, top_n=args.top_n
    )
    
    # Print results
    print(f"\n[5/5] Results...")
    
    print(f"\n  === TOP {args.top_n} OVERALL EDGE-LOCALIZED STATES ===")
    print(f"  {'Rank':<6} {'State':<8} {'Energy':<14} {'Edge Loc (%)':<12}")
    print(f"  {'-'*42}")
    for rank, (idx, E, loc) in enumerate(best_overall, 1):
        print(f"  {rank:<6} {idx:<8} {E:<14.6f} {loc:<12.2f}")
    
    print(f"\n  === TOP {args.top_n} IN E- REGION (E < 0) ===")
    print(f"  {'Rank':<6} {'State':<8} {'Energy':<14} {'Edge Loc (%)':<12}")
    print(f"  {'-'*42}")
    for rank, (idx, E, loc) in enumerate(best_neg, 1):
        print(f"  {rank:<6} {idx:<8} {E:<14.6f} {loc:<12.2f}")
    
    print(f"\n  === TOP {args.top_n} IN E+ REGION (E > 0) ===")
    print(f"  {'Rank':<6} {'State':<8} {'Energy':<14} {'Edge Loc (%)':<12}")
    print(f"  {'-'*42}")
    for rank, (idx, E, loc) in enumerate(best_pos, 1):
        print(f"  {rank:<6} {idx:<8} {E:<14.6f} {loc:<12.2f}")
    
    # Statistics
    print(f"\n  === STATISTICS ===")
    print(f"  Mean edge localization: {np.mean(edge_localizations):.2f}%")
    print(f"  Std edge localization: {np.std(edge_localizations):.2f}%")
    print(f"  Min edge localization: {np.min(edge_localizations):.2f}%")
    print(f"  Max edge localization: {np.max(edge_localizations):.2f}%")
    
    # Compute bipartite sublattices for coloring
    sublattice_map = compute_bipartite_sublattices(edge_list, N)
    num_A = sum(1 for v in sublattice_map.values() if v == 'A')
    num_B = sum(1 for v in sublattice_map.values() if v == 'B')
    print(f"  ✓ Bipartite: Sublattice A={num_A}, Sublattice B={num_B}")
    
    # Generate plots
    print(f"\n[6/5] Generating plots...")
    
    # Plot edge localization vs energy
    plot_edge_localization_vs_energy(eigenvalues, edge_localizations, 
                                     args.iteration, args.output_dir)
    
    # Plot top state overall
    if best_overall:
        idx, E, loc = best_overall[0]
        plot_probability_density(idx, eigenvalues, eigenvectors, vertex_coords,
                                 edge_list, edge_localizations, args.iteration, 
                                 args.output_dir, sublattice_map, N, label="Best_Overall")
    
    # Plot best E- state
    if best_neg:
        idx, E, loc = best_neg[0]
        plot_probability_density(idx, eigenvalues, eigenvectors, vertex_coords,
                                 edge_list, edge_localizations, args.iteration, 
                                 args.output_dir, sublattice_map, N, label="Best_E-")
    
    # Plot best E+ state
    if best_pos:
        idx, E, loc = best_pos[0]
        plot_probability_density(idx, eigenvalues, eigenvectors, vertex_coords,
                                 edge_list, edge_localizations, args.iteration, 
                                 args.output_dir, sublattice_map, N, label="Best_E+")
    
    # Save data
    data_filename = f'edge_localization_data_iter{args.iteration}.npz'
    data_filepath = os.path.join(args.output_dir, data_filename)
    np.savez(data_filepath,
             eigenvalues=eigenvalues,
             edge_localizations=edge_localizations,
             edge_mask=edge_mask,
             iteration=args.iteration,
             N=N)
    print(f"  ✓ Data saved: {data_filepath}")
    
    print("\n" + "=" * 70)
    print("✅ ANALYSIS COMPLETED")
    print("=" * 70)
    print(f"\nResults saved to: {args.output_dir}/")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
