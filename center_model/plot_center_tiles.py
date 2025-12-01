"""
Visualisasi Center Tiles - Plot rhombi dan center sites
Script ini membaca vertex model dan menampilkan:
1. Rhombi (tiles) dari vertex model
2. Center sites (pusat rhombi)
3. Dual edges (koneksi antar center)
"""

import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import argparse


def plot_center_tiles(vertex_filename='vertex_model/data/penrose_lattice_data.npz',
                      output_filename='center_model/imgs/center_tiles_visualization.png',
                      show_vertex_model=True,
                      show_rhombi=True,
                      show_centers=True,
                      show_dual_edges=True):
    """
    Plot visualisasi center tiles
    
    Parameters:
    -----------
    vertex_filename : str
        Path ke file vertex model npz
    output_filename : str
        Path output untuk save plot
    show_vertex_model : bool
        Tampilkan vertex model (merah) sebagai background
    show_rhombi : bool
        Tampilkan rhombi/tiles (kotak hijau)
    show_centers : bool
        Tampilkan center sites (biru)
    show_dual_edges : bool
        Tampilkan dual edges (biru putus-putus)
    """
    
    print("="*80)
    print("CENTER TILES VISUALIZATION")
    print("="*80)
    
    # Load vertex model data
    print(f"\n[1/5] Loading vertex model from {vertex_filename}...")
    try:
        data = np.load(vertex_filename)
        vertex_coords = data['vertex_coords']
        edge_list = data['edge_list']
        N = len(vertex_coords)
        iteration = int(data.get('iteration', 0))
        print(f"  ✓ Loaded N={N} vertices, iteration={iteration}")
    except FileNotFoundError:
        print(f"  ✗ File not found: {vertex_filename}")
        print("    Run: python vertex_model/penrose_tiling_fast.py first")
        return
    
    # Build adjacency list
    print(f"\n[2/5] Building adjacency list...")
    adj = defaultdict(set)
    edges_set = set()
    for u, v in edge_list:
        adj[u].add(v)
        adj[v].add(u)
        edges_set.add(tuple(sorted((u, v))))
    print(f"  ✓ Built adjacency for {N} vertices")
    
    # Detect rhombi (4-cycles)
    print(f"\n[3/5] Detecting rhombi (tiles)...")
    faces = []
    processed_quads = set()
    
    for u in range(N):
        neighbors = list(adj[u])
        for i in range(len(neighbors)):
            for j in range(i + 1, len(neighbors)):
                v = neighbors[i]
                w = neighbors[j]
                
                # Find common neighbor of v and w (excluding u)
                common = adj[v].intersection(adj[w])
                common.discard(u)
                
                if common:
                    x = common.pop()
                    
                    # Verify valid rhombus: check all 4 edges exist
                    quad_edges = [
                        tuple(sorted((u, v))),
                        tuple(sorted((v, x))),
                        tuple(sorted((x, w))),
                        tuple(sorted((w, u)))
                    ]
                    
                    if all(e in edges_set for e in quad_edges):
                        quad = tuple(sorted((u, v, w, x)))
                        if quad not in processed_quads:
                            faces.append(quad)
                            processed_quads.add(quad)
    
    print(f"  ✓ Found {len(faces)} rhombi")
    
    # Compute center coordinates
    print(f"\n[4/5] Computing center sites...")
    centers = []
    for quad in faces:
        coords = vertex_coords[list(quad)]
        center = np.mean(coords, axis=0)
        centers.append(center)
    
    centers = np.array(centers)
    N_centers = len(centers)
    print(f"  ✓ Computed {N_centers} center sites")
    
    # Build dual edges
    print(f"\n[5/5] Building dual graph edges...")
    edge_to_faces = defaultdict(list)
    
    for idx, quad in enumerate(faces):
        for i in range(4):
            for j in range(i+1, 4):
                edge = tuple(sorted((quad[i], quad[j])))
                if edge in edges_set:
                    edge_to_faces[edge].append(idx)
    
    dual_edges = []
    for edge, face_indices in edge_to_faces.items():
        if len(face_indices) == 2:
            f1, f2 = face_indices
            dual_edges.append((f1, f2))
    
    E_dual = len(dual_edges)
    print(f"  ✓ Built {E_dual} dual edges")
    
    # Rotation -18 degrees
    theta = -18.0
    theta_rad = np.deg2rad(theta)
    cos_theta = np.cos(theta_rad)
    sin_theta = np.sin(theta_rad)
    rotation_matrix = np.array([[cos_theta, -sin_theta], 
                                [sin_theta, cos_theta]])
    
    vertex_coords_rot = vertex_coords @ rotation_matrix.T
    centers_rot = centers @ rotation_matrix.T
    
    # Plotting
    print(f"\n[PLOTTING] Creating visualization...")
    fig, ax = plt.subplots(1, 1, figsize=(14, 14), facecolor='white')
    
    # Plot vertex model edges (gray, thin, very faint)
    if show_vertex_model:
        for u, v in edge_list:
            p1, p2 = vertex_coords_rot[u], vertex_coords_rot[v]
            ax.plot([p1[0], p2[0]], [p1[1], p2[1]], 'k-', 
                   lw=0.5, alpha=0.05, zorder=1)
        
        # Plot vertex sites (small red)
        ax.scatter(vertex_coords_rot[:,0], vertex_coords_rot[:,1], 
                  s=8, c='red', alpha=0.3, label='Vertex sites', zorder=2)
    
    # Plot rhombi outlines (green)
    if show_rhombi:
        for quad in faces:
            quad_coords = vertex_coords_rot[list(quad)]
            # Order vertices to form proper rhombus
            # Simple approach: plot all 4 edges
            for i in range(4):
                for j in range(i+1, 4):
                    edge = tuple(sorted((quad[i], quad[j])))
                    if edge in edges_set:
                        p1 = vertex_coords_rot[quad[i]]
                        p2 = vertex_coords_rot[quad[j]]
                        ax.plot([p1[0], p2[0]], [p1[1], p2[1]], 
                               'g-', lw=1.5, alpha=0.05, zorder=3)
    
    # Plot dual edges (blue dashed)
    if show_dual_edges:
        for f1, f2 in dual_edges:
            p1, p2 = centers_rot[f1], centers_rot[f2]
            ax.plot([p1[0], p2[0]], [p1[1], p2[1]], 
                   'b--', lw=1.2, alpha=0.7, zorder=4, 
                   label='Dual edges' if f1 == dual_edges[0][0] else '')
    
    # Plot center sites (blue squares)
    if show_centers:
        ax.scatter(centers_rot[:,0], centers_rot[:,1], 
                  s=50, c='blue', marker='s', edgecolors='black',
                  linewidth=0.5, label='Center sites', zorder=5)
    
    # Title and formatting
    title = f'Center Model Tiles Visualization (Iteration {iteration})\n'
    title += f'Vertex Model: N={N} | Center Model: N={N_centers}, E={E_dual}'
    ax.set_title(title, fontsize=14, fontweight='bold')
    
    # Legend (remove duplicates)
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), loc='upper right', fontsize=10)
    
    ax.set_aspect('equal')
    ax.axis('off')
    
    plt.tight_layout()
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    print(f"\n  ✓ Saved plot: {output_filename}")
    
    print("\n" + "="*80)
    print("✅ Visualization completed!")
    print("="*80)
    print(f"\nSummary:")
    print(f"  Vertex model:  N = {N} vertices")
    print(f"  Center model:  N = {N_centers} centers, E = {E_dual} dual edges")
    print(f"  Output:        {output_filename}")
    print()
    
    return fig, ax


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Visualize center tiles from Penrose lattice')
    parser.add_argument('--input', type=str, 
                       default='vertex_model/data/penrose_lattice_data.npz',
                       help='Input vertex model file (npz)')
    parser.add_argument('--output', type=str,
                       default='center_model/imgs/center_tiles_visualization.png',
                       help='Output image file')
    parser.add_argument('--iteration', type=int, default=None,
                       help='Use specific iteration (e.g., 4, 5, 6)')
    parser.add_argument('--no-vertex', action='store_true',
                       help='Hide vertex model (show only centers)')
    parser.add_argument('--no-rhombi', action='store_true',
                       help='Hide rhombi outlines')
    parser.add_argument('--no-centers', action='store_true',
                       help='Hide center sites')
    parser.add_argument('--no-dual-edges', action='store_true',
                       help='Hide dual edges')
    
    args = parser.parse_args()
    
    # Determine input file
    if args.iteration is not None:
        input_file = f'vertex_model/data/penrose_lattice_iter{args.iteration}.npz'
        output_file = f'center_model/imgs/center_tiles_iter{args.iteration}.png'
    else:
        input_file = args.input
        output_file = args.output
    
    # Run visualization
    plot_center_tiles(
        vertex_filename=input_file,
        output_filename=output_file,
        show_vertex_model=not args.no_vertex,
        show_rhombi=not args.no_rhombi,
        show_centers=not args.no_centers,
        show_dual_edges=not args.no_dual_edges
    )


if __name__ == "__main__":
    main()
