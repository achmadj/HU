"""
Penrose Lattice - FAST VERSION with Optional Plotting
Combines optimized generation with plotting capabilities
"""

import numpy as np
import matplotlib.pyplot as plt
import pickle
import time
import argparse
from collections import defaultdict


class PenroseLatticeOptimized:
    """
    Penrose Lattice dengan optimasi performa untuk iterasi besar
    
    Optimizations:
    1. Spatial hashing untuk vertex deduplication (O(1) lookup)
    2. Batch operations untuk edge processing
    """
   
    def __init__(self):
        """Inisialisasi Penrose Lattice"""
        self.phi = (1 + np.sqrt(5)) / 2
        self.phi_inv = 1 / self.phi
       
        # Vertices stored as numpy array for efficiency
        self.vertex_positions = []
        self.vertex_counter = 0
        
        # Spatial hash for O(1) lookup with tolerance handling
        self.vertex_hash = {}  # hash -> list of vertex IDs
        self.hash_precision = 8  # decimal precision for hashing grid
        self.tolerance = 1e-9  # tolerance for vertex matching
        
        # Edges stored as lists for batch processing
        self.edge_sources = []
        self.edge_targets = []
        self.edge_types = []
   
    def _hash_position(self, pos):
        """
        Hash posisi ke grid cell untuk spatial lookup
        
        Input:
            pos: koordinat [x, y]
        
        Output:
            Tuple[int, int]: hash key
        """
        scale = 10 ** self.hash_precision
        return (int(np.round(pos[0] * scale)), int(np.round(pos[1] * scale)))
    
    def add_vertex(self, position):
        """
        Tambahkan vertex dengan spatial hashing + tolerance check
        
        Input:
            position: koordinat [x, y]
        
        Output:
            int: vertex ID
        """
        hash_key = self._hash_position(position)
        
        # Check if hash bucket exists and search within it
        if hash_key in self.vertex_hash:
            for vid in self.vertex_hash[hash_key]:
                if np.allclose(self.vertex_positions[vid], position, atol=self.tolerance):
                    return vid
        
        # Not found - add new vertex
        vid = self.vertex_counter
        self.vertex_positions.append(np.array(position, dtype=np.float64))
        
        if hash_key not in self.vertex_hash:
            self.vertex_hash[hash_key] = []
        self.vertex_hash[hash_key].append(vid)
        
        self.vertex_counter += 1
        return vid
   
    def add_edge(self, i, j, arrow_type):
        """Tambahkan edge"""
        if i != j:
            self.edge_sources.append(i)
            self.edge_targets.append(j)
            self.edge_types.append(arrow_type)
   
    def create_seed_cluster(self):
        """Membuat seed cluster awal (Iterasi 0)"""
        center = self.add_vertex(np.array([0.0, 0.0]))
       
        # 5 vertex inner ring
        inner_vertices = []
        for k in range(5):
            angle = k * 72
            x = np.cos(np.deg2rad(angle))
            y = np.sin(np.deg2rad(angle))
            vid = self.add_vertex(np.array([x, y]))
            inner_vertices.append(vid)
       
        # 5 vertex outer ring
        outer_vertices = []
        for k in range(5):
            angle = k * 72 + 36
            r = self.phi
            x = r * np.cos(np.deg2rad(angle))
            y = r * np.sin(np.deg2rad(angle))
            vid = self.add_vertex(np.array([x, y]))
            outer_vertices.append(vid)
       
        # Tambahkan edges
        for k in range(5):
            k_next = (k + 1) % 5
            self.add_edge(inner_vertices[k], center, arrow_type=1)
            self.add_edge(outer_vertices[k], inner_vertices[k], arrow_type=2)
            self.add_edge(outer_vertices[k], inner_vertices[k_next], arrow_type=2)
   
    def deflate_once(self):
        """
        Lakukan satu iterasi deflasi (OPTIMIZED)
        """
        # Convert current edges to numpy arrays for batch processing
        old_sources = np.array(self.edge_sources, dtype=np.int32)
        old_targets = np.array(self.edge_targets, dtype=np.int32)
        old_types = np.array(self.edge_types, dtype=np.int32)
        
        # Convert vertex positions to numpy array
        vertex_array = np.array(self.vertex_positions, dtype=np.float64)
        
        # Clear current edges
        self.edge_sources.clear()
        self.edge_targets.clear()
        self.edge_types.clear()
        
        # Process all edges in batches
        new_edges_list = []
        
        for idx in range(len(old_sources)):
            i = old_sources[idx]
            j = old_targets[idx]
            arrow_type = old_types[idx]
            
            pos_i = vertex_array[i]
            pos_j = vertex_array[j]
            
            if arrow_type == 1:
                new_edges = self._deflate_single_arrow_fast(pos_i, pos_j, i, j)
            else:  # arrow_type == 2
                new_edges = self._deflate_double_arrow_fast(pos_i, pos_j, i, j)
            
            new_edges_list.extend(new_edges)
        
        # Deduplication using set with custom logic
        edge_dict = {}
        
        for vi, vj, atype in new_edges_list:
            key = (vi, vj)
            reverse_key = (vj, vi)
            
            if key in edge_dict:
                if atype > edge_dict[key]:
                    edge_dict[key] = atype
            elif reverse_key in edge_dict:
                if atype > edge_dict[reverse_key]:
                    del edge_dict[reverse_key]
                    edge_dict[key] = atype
            else:
                edge_dict[key] = atype
        
        # Add deduplicated edges
        for (vi, vj), atype in edge_dict.items():
            self.add_edge(vi, vj, atype)
        
        # Scale all vertices by φ
        self.vertex_positions = [pos * self.phi for pos in self.vertex_positions]
        
        # Rebuild hash with new positions
        self.vertex_hash.clear()
        for vid, pos in enumerate(self.vertex_positions):
            hash_key = self._hash_position(pos)
            if hash_key not in self.vertex_hash:
                self.vertex_hash[hash_key] = []
            self.vertex_hash[hash_key].append(vid)
    
    def _deflate_single_arrow_fast(self, pos_i, pos_j, i, j):
        """Deflasi single arrow (optimized)"""
        vec_ij = pos_j - pos_i
        
        # Pre-compute rotation matrices
        cos_72 = np.cos(np.deg2rad(72))
        sin_72 = np.sin(np.deg2rad(72))
        
        pos_k = pos_j - self.phi_inv * vec_ij
        
        # Rotate -72 degrees
        vec_rot_neg = np.array([
            cos_72 * vec_ij[0] + sin_72 * vec_ij[1],
            -sin_72 * vec_ij[0] + cos_72 * vec_ij[1]
        ])
        pos_l = pos_i + self.phi_inv * vec_rot_neg
        
        # Rotate +72 degrees
        vec_rot_pos = np.array([
            cos_72 * vec_ij[0] - sin_72 * vec_ij[1],
            sin_72 * vec_ij[0] + cos_72 * vec_ij[1]
        ])
        pos_m = pos_i + self.phi_inv * vec_rot_pos
        
        k = self.add_vertex(pos_k)
        l = self.add_vertex(pos_l)
        m = self.add_vertex(pos_m)
        
        return [(i, l, 2), (k, l, 1), (k, m, 1), (i, m, 2), (j, k, 2)]
    
    def _deflate_double_arrow_fast(self, pos_i, pos_j, i, j):
        """Deflasi double arrow (optimized)"""
        vec_ij = pos_j - pos_i
        
        # Pre-compute rotation matrices
        cos_36 = np.cos(np.deg2rad(36))
        sin_36 = np.sin(np.deg2rad(36))
        
        # Rotate -36 degrees
        vec_rot_neg = np.array([
            cos_36 * vec_ij[0] + sin_36 * vec_ij[1],
            -sin_36 * vec_ij[0] + cos_36 * vec_ij[1]
        ])
        pos_g = pos_i + self.phi_inv * vec_rot_neg
        
        # Rotate +36 degrees
        vec_rot_pos = np.array([
            cos_36 * vec_ij[0] - sin_36 * vec_ij[1],
            sin_36 * vec_ij[0] + cos_36 * vec_ij[1]
        ])
        pos_h = pos_i + self.phi_inv * vec_rot_pos
        
        g = self.add_vertex(pos_g)
        h = self.add_vertex(pos_h)
        
        return [(g, i, 1), (j, g, 2), (j, h, 2), (h, i, 1)]
    
    def get_statistics(self):
        """Return (N, E)"""
        return len(self.vertex_positions), len(self.edge_sources)
    
    def get_edge_type_count(self):
        """Return (single_count, double_count)"""
        single = sum(1 for t in self.edge_types if t == 1)
        double = sum(1 for t in self.edge_types if t == 2)
        return single, double
    
    def to_dict_format(self):
        """Convert to dictionary format for saving"""
        vertices = {i: pos for i, pos in enumerate(self.vertex_positions)}
        edges = {(self.edge_sources[i], self.edge_targets[i]): self.edge_types[i] 
                 for i in range(len(self.edge_sources))}
        return {'vertices': vertices, 'edges': edges}


def analyze_overlapping_edges(edges, vertices):
    """
    Analisis edges yang tumpang tindih (berbagi kedua vertex yang sama)
    
    Returns:
    --------
    overlapping_groups : list of lists
        Setiap grup berisi edges yang overlap
    unique_edges : dict
        Edges yang tidak overlap
    """
    # Group edges by unordered vertex pair
    edge_groups = defaultdict(list)
    
    for (i, j), arrow_type in edges.items():
        edge_key = frozenset([i, j])
        edge_groups[edge_key].append(((i, j), arrow_type))
    
    overlapping_groups = []
    unique_edges = {}
    
    for edge_key, edge_list in edge_groups.items():
        if len(edge_list) > 1:
            # Ada overlap
            overlapping_groups.append(edge_list)
        else:
            # Unique edge
            (i, j), arrow_type = edge_list[0]
            unique_edges[(i, j)] = arrow_type
    
    return overlapping_groups, unique_edges


def plot_with_overlapping_analysis(state, ax, show_arrows=True, offset_overlapping=False):
    """
    Plot dengan analisis overlapping edges
    
    Parameters:
    -----------
    offset_overlapping : bool
        Jika True, pisahkan edges yang overlap dengan offset
    """
    vertices = state['vertices']
    edges = state['edges']
    
    # Analisis overlapping
    overlapping_groups, unique_edges = analyze_overlapping_edges(edges, vertices)
    
    # Plot unique edges (tidak overlap)
    for (i, j), arrow_type in unique_edges.items():
        x1, y1 = vertices[i]
        x2, y2 = vertices[j]
        
        color = 'blue' if arrow_type == 1 else 'red'
        linewidth = 1.2 if arrow_type == 1 else 1.8
        
        ax.plot([x1, x2], [y1, y2], 
                color=color, linewidth=linewidth, alpha=0.6, zorder=1,
                label='_nolegend_')
    
    # Plot overlapping edges
    for group in overlapping_groups:
        for idx, ((i, j), arrow_type) in enumerate(group):
            x1, y1 = vertices[i]
            x2, y2 = vertices[j]
            
            color = 'blue' if arrow_type == 1 else 'red'
            linewidth = 1.2 if arrow_type == 1 else 1.8
            
            if offset_overlapping and len(group) > 1:
                # Offset perpendicular ke edge
                dx = x2 - x1
                dy = y2 - y1
                length = np.sqrt(dx**2 + dy**2)
                
                if length > 0:
                    # Vektor perpendicular (rotate 90 degrees)
                    perp_x = -dy / length
                    perp_y = dx / length
                    
                    # Offset amount
                    offset_amount = 0.05 * length * (idx - 0.5)
                    
                    x1_off = x1 + perp_x * offset_amount
                    y1_off = y1 + perp_y * offset_amount
                    x2_off = x2 + perp_x * offset_amount
                    y2_off = y2 + perp_y * offset_amount
                    
                    # Plot dengan style khusus untuk overlap
                    ax.plot([x1_off, x2_off], [y1_off, y2_off], 
                            color=color, linewidth=linewidth, alpha=0.8, 
                            linestyle='--' if idx > 0 else '-',
                            zorder=2, label='_nolegend_')
            else:
                # Plot normal (tumpuk)
                ax.plot([x1, x2], [y1, y2], 
                        color=color, linewidth=linewidth + idx*0.5, alpha=0.6, 
                        zorder=1, label='_nolegend_')
    
    # Plot arrows
    if show_arrows:
        for (i, j), arrow_type in edges.items():
            x1, y1 = vertices[i]
            x2, y2 = vertices[j]
            
            mid_x = (x1 + x2) / 2
            mid_y = (y1 + y2) / 2
            
            dx = x2 - x1
            dy = y2 - y1
            
            length = np.sqrt(dx**2 + dy**2)
            if length > 0:
                arrow_scale = 0.15
                dx_norm = (dx / length) * length * arrow_scale
                dy_norm = (dy / length) * length * arrow_scale
                
                color = 'darkblue' if arrow_type == 1 else 'darkred'
                
                ax.quiver(mid_x, mid_y, dx_norm, dy_norm,
                        angles='xy', scale_units='xy', scale=1,
                        color=color, width=0.008, headwidth=4, headlength=5,
                        alpha=0.9, zorder=3)
    
    # Plot vertices
    xs = [pos[0] for pos in vertices.values()]
    ys = [pos[1] for pos in vertices.values()]
    ax.scatter(xs, ys, c='black', s=18, zorder=4, alpha=0.9, 
                edgecolors='white', linewidths=0.5)
    
    # Formatting
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.2, linestyle='--')
    
    # Count overlapping
    n_overlap = len(overlapping_groups)
    total_overlap_edges = sum(len(g) for g in overlapping_groups)
    unique_count = len(unique_edges)
    
    title = f"{state['title']}\n"
    title += f"N={state['N']}, E={state['E']} "
    title += f"(Unique: {unique_count}, Overlapping: {total_overlap_edges} in {n_overlap} pairs)"
    
    ax.set_title(title, fontsize=11, fontweight='bold')
    ax.set_xlabel('x', fontsize=10)
    ax.set_ylabel('y', fontsize=10)
    
    # Equal limits
    if len(xs) > 0:
        max_coord = max(max(abs(x) for x in xs), max(abs(y) for y in ys))
        margin = max_coord * 0.15
        ax.set_xlim(-max_coord - margin, max_coord + margin)
        ax.set_ylim(-max_coord - margin, max_coord + margin)


def plot_overlapping_iterations(state0, state1, ax, show_arrows=False):
    vertices0 = state0['vertices']
    edges0 = state0['edges']
    vertices1 = state1['vertices']
    edges1 = state1['edges']
    
    # Golden ratio untuk scaling
    phi = (1 + np.sqrt(5)) / 2
    phi_inv = 1 / phi
    
    # Plot edges iterasi 0 (hitam)
    for (i, j), arrow_type in edges0.items():
        x1, y1 = vertices0[i]
        x2, y2 = vertices0[j]
        
        ax.plot([x1, x2], [y1, y2], 
                color='black', linewidth=1.5, alpha=0.7, zorder=1,
                label='_nolegend_')
    
    # Plot edges iterasi 1 (merah) - scaled dengan phi^-1
    for (i, j), arrow_type in edges1.items():
        x1, y1 = vertices1[i] * phi_inv
        x2, y2 = vertices1[j] * phi_inv
        
        ax.plot([x1, x2], [y1, y2], 
                color='red', linewidth=1.5, alpha=0.7, zorder=2,
                label='_nolegend_')
    
    # Plot vertices iterasi 0 (hitam)
    xs0 = [pos[0] for pos in vertices0.values()]
    ys0 = [pos[1] for pos in vertices0.values()]
    ax.scatter(xs0, ys0, c='black', s=20, zorder=5, alpha=0.8,
                edgecolors='white', linewidths=0.5)
    
    # Plot vertices iterasi 1 (merah) - scaled dengan phi^-1
    xs1 = [pos[0] * phi_inv for pos in vertices1.values()]
    ys1 = [pos[1] * phi_inv for pos in vertices1.values()]
    ax.scatter(xs1, ys1, c='red', s=20, zorder=6, alpha=0.8,
                edgecolors='white', linewidths=0.5)
    
    # Formatting
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.2, linestyle='--')
    
    title = f"Iteration 0 (Black) & Iteration 1 (Red, scaled φ⁻¹) Overlapped\n"
    title += f"Iter 0: N={state0['N']}, E={state0['E']} | "
    title += f"Iter 1: N={state1['N']}, E={state1['E']}"
    
    ax.set_title(title, fontsize=13, fontweight='bold')
    ax.set_xlabel('x', fontsize=10)
    ax.set_ylabel('y', fontsize=10)
    
    # Equal limits based on both iterations
    all_xs = xs0 + xs1
    all_ys = ys0 + ys1
    if len(all_xs) > 0:
        max_coord = max(max(abs(x) for x in all_xs), max(abs(y) for y in all_ys))
        margin = max_coord * 0.15
        ax.set_xlim(-max_coord - margin, max_coord + margin)
        ax.set_ylim(-max_coord - margin, max_coord + margin)


def print_separator(char="=", length=80):
    """Print separator line"""
    print(char * length)


def main():
    """Main program - FAST VERSION with optional plotting"""
    parser = argparse.ArgumentParser(description='Generate Penrose Lattice')
    parser.add_argument('--plot', action='store_true', help='Generate and save plots')
    args = parser.parse_args()
    
    print_separator()
    print("PENROSE LATTICE - FAST VERSION")
    if args.plot:
        print("With plotting enabled")
    else:
        print("Plotting disabled (use --plot to enable)")
    print_separator()
    
    penrose = PenroseLatticeOptimized()
    
    # Seed cluster
    print("\n[Iteration 0] Creating seed cluster...")
    t0 = time.time()
    penrose.create_seed_cluster()
    t1 = time.time()
    
    N, E = penrose.get_statistics()
    single, double = penrose.get_edge_type_count()
    
    print(f"  ✓ Vertices (N): {N}")
    print(f"  ✓ Edges (E):    {E}")
    print(f"    - Single arrows:  {single}")
    print(f"    - Double arrows:  {double}")
    print(f"  ⏱️  Time: {(t1-t0)*1000:.2f} ms")
    
    # Store statistics
    stats = [{
        'iteration': 0,
        'N': N,
        'E': E,
        'single': single,
        'double': double,
        'time_ms': (t1-t0)*1000
    }]
    
    # Iterasi 1-7
    max_iterations = 7
    
    for iteration in range(1, max_iterations + 1):
        print(f"\n[Iteration {iteration}] Performing deflation...")
        t0 = time.time()
        penrose.deflate_once()
        t1 = time.time()
        
        N, E = penrose.get_statistics()
        single, double = penrose.get_edge_type_count()
        
        print(f"  ✓ Vertices (N): {N}")
        print(f"  ✓ Edges (E):    {E}")
        print(f"    - Single arrows:  {single}")
        print(f"    - Double arrows:  {double}")
        print(f"  ⏱️  Time: {(t1-t0)*1000:.2f} ms")
        
        stats.append({
            'iteration': iteration,
            'N': N,
            'E': E,
            'single': single,
            'double': double,
            'time_ms': (t1-t0)*1000
        })
    
    # Summary table
    print("\n")
    print_separator()
    print("SUMMARY TABLE")
    print_separator()
    print(f"{'Iter':<6} {'N':<10} {'E':<10} {'Single':<10} {'Double':<10} {'Time (ms)':<12}")
    print_separator("-")
    
    for stat in stats:
        print(f"{stat['iteration']:<6} {stat['N']:<10} {stat['E']:<10} "
              f"{stat['single']:<10} {stat['double']:<10} {stat['time_ms']:<12.2f}")
    
    print_separator("-")
    
    # Growth analysis
    print("\n")
    print_separator()
    print("GROWTH ANALYSIS")
    print_separator()
    
    phi = (1 + np.sqrt(5)) / 2
    phi_squared = phi ** 2
    
    print(f"Golden ratio φ = {phi:.6f}")
    print(f"φ² = {phi_squared:.6f}\n")
    
    for i in range(1, len(stats)):
        prev_N = stats[i-1]['N']
        curr_N = stats[i]['N']
        prev_E = stats[i-1]['E']
        curr_E = stats[i]['E']
        
        N_ratio = curr_N / prev_N if prev_N > 0 else 0
        E_ratio = curr_E / prev_E if prev_E > 0 else 0
        
        print(f"Iteration {i-1} → {i}:")
        print(f"  N: {prev_N:6d} → {curr_N:6d}  (ratio: {N_ratio:.4f})")
        print(f"  E: {prev_E:6d} → {curr_E:6d}  (ratio: {E_ratio:.4f})")
    
    # Total time
    print("\n")
    print_separator()
    print("PERFORMANCE SUMMARY")
    print_separator()
    total_time = sum(stat['time_ms'] for stat in stats)
    print(f"Total computation time: {total_time:.2f} ms ({total_time/1000:.3f} seconds)")
    print(f"Average time per iteration: {total_time/len(stats):.2f} ms")
    print(f"Final vertex count: {stats[-1]['N']}")
    print(f"Final edge count: {stats[-1]['E']}")
    
    # Save data
    print("\n")
    print_separator()
    print("SAVING DATA")
    print_separator()
    
    final_stat = stats[-1]
    
    # Convert to dictionary format
    data_dict = penrose.to_dict_format()
    
    data_to_save = {
        'vertices': data_dict['vertices'],
        'edges': data_dict['edges'],
        'N': final_stat['N'],
        'E': final_stat['E'],
        'iteration': final_stat['iteration'],
        'phi': penrose.phi,
        'single_arrows': final_stat['single'],
        'double_arrows': final_stat['double']
    }
    
    # Save as pickle
    pickle_file = 'penrose_lattice_data.pkl'
    with open(pickle_file, 'wb') as f:
        pickle.dump(data_to_save, f)
    print(f"  ✓ Saved to pickle: {pickle_file}")
    
    # Save as numpy arrays
    npz_file = 'penrose_lattice_data.npz'
    
    vertex_ids = np.arange(len(penrose.vertex_positions), dtype=np.int32)
    vertex_coords = np.array(penrose.vertex_positions, dtype=np.float64)
    
    edge_list = np.column_stack([penrose.edge_sources, penrose.edge_targets]).astype(np.int32)
    edge_types = np.array(penrose.edge_types, dtype=np.int32)
    
    np.savez(npz_file,
             vertex_ids=vertex_ids,
             vertex_coords=vertex_coords,
             edge_list=edge_list,
             edge_types=edge_types,
             N=final_stat['N'],
             E=final_stat['E'],
             iteration=final_stat['iteration'],
             phi=penrose.phi)
    print(f"  ✓ Saved to numpy: {npz_file}")
    
    # Plotting if requested
    if args.plot:
        print("\n")
        print_separator()
        print("GENERATING PLOTS")
        print_separator()
        
        # Save states for plotting
        states = []
        penrose_plot = PenroseLatticeOptimized()
        penrose_plot.create_seed_cluster()
        
        N, E = penrose_plot.get_statistics()
        states.append({
            'title': 'Iteration 0 (Seed)',
            'vertices': {i: pos for i, pos in enumerate(penrose_plot.vertex_positions)},
            'edges': {(penrose_plot.edge_sources[i], penrose_plot.edge_targets[i]): penrose_plot.edge_types[i] 
                     for i in range(len(penrose_plot.edge_sources))},
            'N': N,
            'E': E
        })
        
        for it in range(1, 6):
            penrose_plot.deflate_once()
            N, E = penrose_plot.get_statistics()
            states.append({
                'title': f'Iteration {it}',
                'vertices': {i: pos for i, pos in enumerate(penrose_plot.vertex_positions)},
                'edges': {(penrose_plot.edge_sources[i], penrose_plot.edge_targets[i]): penrose_plot.edge_types[i] 
                         for i in range(len(penrose_plot.edge_sources))},
                'N': N,
                'E': E
            })
        
        # Plot 1: Overview semua iterasi
        fig1, axes = plt.subplots(2, 2, figsize=(20, 20))
        axes_flat = axes.flatten()
        
        for idx, state in enumerate(states[:4]):  # Plot first 4 iterations
            plot_with_overlapping_analysis(state, axes_flat[idx], 
                                            show_arrows=True, 
                                            offset_overlapping=False)
        
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], color='blue', linewidth=2.5, 
                    marker='>', markersize=8, markerfacecolor='darkblue',
                    label='Single Arrow'),
            Line2D([0], [0], color='red', linewidth=2.5, 
                    marker='>', markersize=8, markerfacecolor='darkred',
                    label='Double Arrow')
        ]
        axes_flat[0].legend(handles=legend_elements, loc='upper right', fontsize=11)
        
        plt.suptitle('Penrose Lattice - All Iterations (Fast Version)', 
                        fontsize=16, fontweight='bold', y=0.998)
        plt.tight_layout()
        plt.savefig('penrose_all_iterations_fast.png', dpi=200, bbox_inches='tight')
        print("  ✓ Saved plot: penrose_all_iterations_fast.png")
        
        # Plot 2: Iterasi 0 dan 1 tumpang tindih
        fig2, ax2 = plt.subplots(1, 1, figsize=(12, 12))
        
        plot_overlapping_iterations(states[0], states[1], ax2, show_arrows=False)
        
        legend_elements_2 = [
            Line2D([0], [0], color='black', linewidth=2.5, 
                    label='Iteration 0'),
            Line2D([0], [0], color='red', linewidth=2.5, 
                    label='Iteration 1')
        ]
        ax2.legend(handles=legend_elements_2, loc='upper right', fontsize=11)
        
        plt.tight_layout()
        plt.savefig('penrose_iter0_iter1_overlapped_fast.png', dpi=200, bbox_inches='tight')
        print("  ✓ Saved plot: penrose_iter0_iter1_overlapped_fast.png")
        
        plt.close('all')
    
    print_separator()
    print("\n✅ Fast computation completed!")
    print(f"🚀 Successfully generated lattice up to iteration {max_iterations}")
    print(f"⚡ Total time: {total_time/1000:.3f} seconds")
    if args.plot:
        print("📊 Plots saved")
    print()


if __name__ == "__main__":
    main()