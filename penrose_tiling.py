import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
class PenroseLattice:
    """
    Penrose Lattice menggunakan metode deflasi
    """
   
    def __init__(self):
        # Golden ratio φ = (1+√5)/2
        self.phi = (1 + np.sqrt(5)) / 2
        self.phi_inv = 1 / self.phi
       
        # Vertices dan edges
        self.vertices = {}
        self.edges = {}
        self.vertex_counter = 0
       
    def rotation_matrix(self, theta_deg):
        """Matriks rotasi R(θ)"""
        theta_rad = np.deg2rad(theta_deg)
        cos_t = np.cos(theta_rad)
        sin_t = np.sin(theta_rad)
        return np.array([[cos_t, -sin_t], [sin_t, cos_t]])
   
    def add_vertex(self, position):
        """Tambahkan vertex atau return ID jika sudah ada"""
        for vid, pos in self.vertices.items():
            if np.allclose(pos, position, atol=1e-9):
                return vid
        vid = self.vertex_counter
        self.vertices[vid] = np.array(position, dtype=float)
        self.vertex_counter += 1
        return vid
   
    def add_edge(self, i, j, arrow_type):
        """Tambahkan directed edge"""
        if i != j:
            self.edges[(i, j)] = arrow_type
   
    def create_seed_cluster(self):
        """
        Seed cluster N=11, E=15
        - Radial: inner → center (single arrow)
        - Kite: outer → inner (double arrow)
        """
        center = self.add_vertex([0, 0])
       
        # 5 vertex inner ring
        inner_vertices = []
        for k in range(5):
            angle = k * 72
            x = np.cos(np.deg2rad(angle))
            y = np.sin(np.deg2rad(angle))
            vid = self.add_vertex([x, y])
            inner_vertices.append(vid)
       
        # 5 vertex outer ring
        outer_vertices = []
        for k in range(5):
            angle = k * 72 + 36
            r = self.phi
            x = r * np.cos(np.deg2rad(angle))
            y = r * np.sin(np.deg2rad(angle))
            vid = self.add_vertex([x, y])
            outer_vertices.append(vid)
       
        # Tambahkan edges
        for k in range(5):
            k_next = (k + 1) % 5
           
            # Radial: inner → center (single arrow)
            self.add_edge(inner_vertices[k], center, arrow_type=1)
           
            # Kite: outer → inner (double arrow)
            self.add_edge(outer_vertices[k], inner_vertices[k], arrow_type=2)
            self.add_edge(outer_vertices[k], inner_vertices[k_next], arrow_type=2)
       
        print(f"Seed cluster: N={len(self.vertices)}, E={len(self.edges)}")
   
    def deflate_single_arrow(self, i, j):
        """Transformasi single arrow - Persamaan (1.3)"""
        pos_i = self.vertices[i]
        pos_j = self.vertices[j]
        vec_ij = pos_j - pos_i
       
        pos_k = pos_j - self.phi_inv * vec_ij
        R_neg72 = self.rotation_matrix(-72)
        pos_l = pos_i + self.phi_inv * (R_neg72 @ vec_ij)
        R_pos72 = self.rotation_matrix(72)
        pos_m = pos_i + self.phi_inv * (R_pos72 @ vec_ij)
       
        k = self.add_vertex(pos_k)
        l = self.add_vertex(pos_l)
        m = self.add_vertex(pos_m)
       
        # Original directions
        new_edges = [
            (i, l, 2),
            (k, l, 1),
            (k, m, 1),
            (i, m, 2),
            (j, k, 2),
        ]
        return new_edges
   
    def deflate_double_arrow(self, i, j):
        """Transformasi double arrow - Persamaan (1.4)"""
        pos_i = self.vertices[i]
        pos_j = self.vertices[j]
        vec_ij = pos_j - pos_i
       
        R_neg36 = self.rotation_matrix(-36)
        pos_g = pos_i + self.phi_inv * (R_neg36 @ vec_ij)
        R_pos36 = self.rotation_matrix(36)
        pos_h = pos_i + self.phi_inv * (R_pos36 @ vec_ij)
       
        g = self.add_vertex(pos_g)
        h = self.add_vertex(pos_h)
       
        # Reversed directions to fix the issue
        new_edges = [
            (g, i, 1),
            (j, g, 2),
            (j, h, 2),
            (h, i, 1),
        ]
        return new_edges
   
    def deflate_once(self):
        """Satu iterasi deflasi"""
        old_edges = dict(self.edges)
        self.edges.clear()
       
        edges_to_add = []
       
        for (i, j), arrow_type in old_edges.items():
            if arrow_type == 1:
                new_edges = self.deflate_single_arrow(i, j)
            elif arrow_type == 2:
                new_edges = self.deflate_double_arrow(i, j)
            else:
                continue
            edges_to_add.extend(new_edges)
       
        # Improved deduplication to avoid double counting
        edge_dict = {}
        for vi, vj, atype in edges_to_add:
            key = (vi, vj)
            reverse_key = (vj, vi)
           
            # Check if edge already exists in either direction
            if key in edge_dict:
                # Keep the edge with higher arrow_type priority
                if atype > edge_dict[key]:
                    edge_dict[key] = atype
            elif reverse_key in edge_dict:
                # Edge exists in reverse direction
                # Keep only one direction with higher priority
                if atype > edge_dict[reverse_key]:
                    del edge_dict[reverse_key]
                    edge_dict[key] = atype
                # If same priority, keep the existing one (no change)
            else:
                # New edge, add it
                edge_dict[key] = atype
       
        # Add deduplicated edges
        for (vi, vj), atype in edge_dict.items():
            self.add_edge(vi, vj, atype)
       
        # Normalisasi
        for vid in self.vertices:
            self.vertices[vid] *= self.phi
       
        print(f"After deflation: N={len(self.vertices)}, E={len(self.edges)}")
   
    def get_statistics(self):
        """Return N and E"""
        return len(self.vertices), len(self.edges)


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
        # Create unordered key (frozenset agar {i,j} sama dengan {j,i})
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
    """
    Plot iterasi 0 dan 1 yang saling tumpang tindih dalam satu gambar
    - Iterasi 0: warna hitam
    - Iterasi 1: warna merah
    - Arrow dihilangkan untuk clarity
    """
    vertices0 = state0['vertices']
    edges0 = state0['edges']
    vertices1 = state1['vertices']
    edges1 = state1['edges']
    
    # Plot edges iterasi 0 (hitam)
    for (i, j), arrow_type in edges0.items():
        x1, y1 = vertices0[i]
        x2, y2 = vertices0[j]
        
        ax.plot([x1, x2], [y1, y2], 
                color='black', linewidth=1.5, alpha=0.7, zorder=1,
                label='_nolegend_')
    
    # Plot edges iterasi 1 (merah)
    for (i, j), arrow_type in edges1.items():
        x1, y1 = vertices1[i]
        x2, y2 = vertices1[j]
        
        ax.plot([x1, x2], [y1, y2], 
                color='red', linewidth=1.5, alpha=0.7, zorder=2,
                label='_nolegend_')
    
    # Plot vertices iterasi 0 (hitam)
    xs0 = [pos[0] for pos in vertices0.values()]
    ys0 = [pos[1] for pos in vertices0.values()]
    ax.scatter(xs0, ys0, c='black', s=20, zorder=5, alpha=0.8,
                edgecolors='white', linewidths=0.5)
    
    # Plot vertices iterasi 1 (merah)
    xs1 = [pos[0] for pos in vertices1.values()]
    ys1 = [pos[1] for pos in vertices1.values()]
    ax.scatter(xs1, ys1, c='red', s=20, zorder=6, alpha=0.8,
                edgecolors='white', linewidths=0.5)
    
    # Formatting
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.2, linestyle='--')
    
    title = f"Iteration 0 (Black) & Iteration 1 (Red) Overlapped\n"
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


def main():
    """Main program"""
    print("="*70)
    print("PENROSE LATTICE - FINAL VERSION")
    print("="*70)
    
    penrose = PenroseLattice()
    
    # Seed
    print("\n=== Seed Cluster ===")
    penrose.create_seed_cluster()
    
    # Save states
    states = []
    N, E = penrose.get_statistics()
    states.append({
        'title': 'Iteration 0 (Seed)',
        'vertices': dict(penrose.vertices),
        'edges': dict(penrose.edges),
        'N': N,
        'E': E
    })
    
    # Iterasi 1-3
    for it in range(1, 6):
        print(f"\n=== Iteration {it} ===")
        penrose.deflate_once()
        
        N, E = penrose.get_statistics()
        states.append({
            'title': f'Iteration {it}',
            'vertices': dict(penrose.vertices),
            'edges': dict(penrose.edges),
            'N': N,
            'E': E
        })
    
    # ========== PLOT 1: Overview semua iterasi ==========
    fig1, axes = plt.subplots(2, 2, figsize=(20, 20))
    axes = axes.flatten()
    
    for idx, state in enumerate(states):
        plot_with_overlapping_analysis(state, axes[idx], 
                                        show_arrows=True, 
                                        offset_overlapping=False)
    
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color='blue', linewidth=2.5, 
                marker='>', markersize=8, markerfacecolor='darkblue',
                label='Single Arrow (72°)'),
        Line2D([0], [0], color='red', linewidth=2.5, 
                marker='>', markersize=8, markerfacecolor='darkred',
                label='Double Arrow (36°)')
    ]
    axes[0].legend(handles=legend_elements, loc='upper right', fontsize=11)
    
    plt.suptitle('Penrose Lattice - Semua Iterasi', 
                    fontsize=16, fontweight='bold', y=0.998)
    plt.tight_layout()
    plt.savefig('penrose_all_iterations_corrected.png', dpi=200, bbox_inches='tight')
    print("\n[SAVED] penrose_all_iterations_corrected.png")
    
    # ========== PLOT 2: Iterasi 0 dan 1 Tumpang Tindih ==========
    fig2, ax2 = plt.subplots(1, 1, figsize=(12, 12))
    
    # show_arrows=False untuk menghilangkan arrow
    plot_overlapping_iterations(states[0], states[1], ax2, show_arrows=False)
    
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color='black', linewidth=2.5, 
                label='Iteration 0'),
        Line2D([0], [0], color='red', linewidth=2.5, 
                label='Iteration 1')
    ]
    ax2.legend(handles=legend_elements, loc='upper right', fontsize=11)
    
    plt.tight_layout()
    plt.savefig('penrose_iter0_iter1_overlapped_corrected.png', dpi=200, bbox_inches='tight')
    print("\n[SAVED] penrose_iter0_iter1_overlapped_corrected.png")
    
    # ========== Summary ==========
    print("\n" + "="*70)
    print("HASIL (Setelah Perbaikan):")
    print("="*70)
    for state in states:
        print(f"{state['title']:25s}: N={state['N']:4d}, E={state['E']:4d}")
    
    print("\n" + "="*70)
    print("EXPECTED (dari supervisor Fig. 6):")
    print("="*70)
    print(f"{'Iteration 0 (Seed)':25s}: N={11:4d}")
    print(f"{'Iteration 1':25s}: N={31:4d}")
    print(f"{'Iteration 2':25s}: N={86:4d}")
    print(f"{'Iteration 3':25s}: N={226:4d}")
    
    print("\n" + "="*70)
    print("ANALISIS OVERLAPPING (Setelah Perbaikan):")
    print("="*70)
    for state in states:
        overlapping_groups, unique_edges = analyze_overlapping_edges(
            state['edges'], state['vertices']
        )
        n_overlap = len(overlapping_groups)
        total_overlap = sum(len(g) for g in overlapping_groups)
        
        print(f"{state['title']:25s}: "
                f"Unique edges={len(unique_edges)}, "
                f"Overlapping={total_overlap} edges in {n_overlap} pairs")
    
    print("="*70)
    
    # plt.show()


if __name__ == "__main__":
    main()