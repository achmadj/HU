"""
Penrose Lattice dengan Metode Deflasi
Versi dengan Static Typing Lengkap untuk Kemudahan Pemahaman
"""

import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from typing import Dict, List, Tuple, Any
from numpy.typing import NDArray


class PenroseLattice:
    """
    Penrose Lattice menggunakan metode deflasi
    
    Attributes:
        phi (float): Golden ratio φ = (1+√5)/2 ≈ 1.618
        phi_inv (float): Inverse golden ratio 1/φ ≈ 0.618
        vertices (Dict[int, NDArray[np.float64]]): Dictionary vertex_id → koordinat [x, y]
        edges (Dict[Tuple[int, int], int]): Dictionary (i, j) → arrow_type (1 atau 2)
        vertex_counter (int): Counter untuk ID vertex berikutnya
    """
   
    def __init__(self) -> None:
        """
        Inisialisasi Penrose Lattice
        
        Input: None
        Output: None
        """
        # Golden ratio φ = (1+√5)/2
        self.phi: float = (1 + np.sqrt(5)) / 2
        self.phi_inv: float = 1 / self.phi
       
        # Vertices dan edges
        self.vertices: Dict[int, NDArray[np.float64]] = {}
        self.edges: Dict[Tuple[int, int], int] = {}
        self.vertex_counter: int = 0
       
    def rotation_matrix(self, theta_deg: float) -> NDArray[np.float64]:
        """
        Membuat matriks rotasi 2D
        
        Input:
            theta_deg (float): Sudut rotasi dalam derajat
        
        Output:
            NDArray[np.float64]: Matriks rotasi 2×2
                                [[cos(θ), -sin(θ)],
                                 [sin(θ),  cos(θ)]]
        
        Contoh:
            >>> R = rotation_matrix(90)
            >>> R @ [1, 0]  # Rotasi vektor [1,0] sebesar 90°
            array([0, 1])
        """
        theta_rad: float = np.deg2rad(theta_deg)
        cos_t: float = np.cos(theta_rad)
        sin_t: float = np.sin(theta_rad)
        return np.array([[cos_t, -sin_t], [sin_t, cos_t]])
   
    def add_vertex(self, position: List[float] | NDArray[np.float64]) -> int:
        """
        Tambahkan vertex atau return ID jika sudah ada
        
        Input:
            position (List[float] | NDArray): Koordinat [x, y] vertex
        
        Output:
            int: ID vertex (baru atau yang sudah ada)
        
        Algoritma:
            1. Loop semua vertex yang ada
            2. Jika posisi sudah ada (toleransi 1e-9), return ID lama
            3. Jika belum ada, buat vertex baru dengan ID dari counter
            4. Increment counter dan return ID baru
        
        Contoh:
            >>> vid1 = add_vertex([0.0, 0.0])  # vid1 = 0 (baru)
            >>> vid2 = add_vertex([0.0, 0.0])  # vid2 = 0 (duplikat)
            >>> vid3 = add_vertex([1.0, 0.0])  # vid3 = 1 (baru)
        """
        for vid, pos in self.vertices.items():
            if np.allclose(pos, position, atol=1e-9):
                return vid
        
        vid: int = self.vertex_counter
        self.vertices[vid] = np.array(position, dtype=float)
        self.vertex_counter += 1
        return vid
   
    def add_edge(self, i: int, j: int, arrow_type: int) -> None:
        """
        Tambahkan directed edge dari vertex i ke vertex j
        
        Input:
            i (int): Vertex ID sumber (tail)
            j (int): Vertex ID tujuan (head)
            arrow_type (int): Tipe panah (1 = single, 2 = double)
        
        Output:
            None (modifikasi self.edges in-place)
        
        Catatan:
            - Self-loop (i == j) diabaikan
            - Edge adalah directed: (i, j) ≠ (j, i)
        
        Contoh:
            >>> add_edge(0, 1, 1)  # Edge 0→1 dengan single arrow
            >>> add_edge(1, 0, 2)  # Edge 1→0 dengan double arrow (berbeda!)
        """
        if i != j:
            self.edges[(i, j)] = arrow_type
   
    def create_seed_cluster(self) -> None:
        """
        Membuat seed cluster awal (Iterasi 0)
        
        Input: None
        
        Output: None (modifikasi self.vertices dan self.edges in-place)
        
        Struktur yang dibuat:
            - 1 vertex center di [0, 0]
            - 5 vertex inner ring pada radius 1, sudut 0°, 72°, 144°, 216°, 288°
            - 5 vertex outer ring pada radius φ, sudut 36°, 108°, 180°, 252°, 324°
            - 5 radial edges: inner → center (single arrow)
            - 10 kite edges: outer → inner (double arrow)
        
        Total: N=11 vertices, E=15 edges
        
        Matematis:
            Inner ring: r_k = (cos(72°k), sin(72°k)), k=0..4
            Outer ring: r_k = φ(cos(72°k+36°), sin(72°k+36°)), k=0..4
        """
        center: int = self.add_vertex([0, 0])
       
        # 5 vertex inner ring
        inner_vertices: List[int] = []
        for k in range(5):
            angle: float = k * 72
            x: float = np.cos(np.deg2rad(angle))
            y: float = np.sin(np.deg2rad(angle))
            vid: int = self.add_vertex([x, y])
            inner_vertices.append(vid)
       
        # 5 vertex outer ring
        outer_vertices: List[int] = []
        for k in range(5):
            angle: float = k * 72 + 36
            r: float = self.phi
            x: float = r * np.cos(np.deg2rad(angle))
            y: float = r * np.sin(np.deg2rad(angle))
            vid: int = self.add_vertex([x, y])
            outer_vertices.append(vid)
       
        # Tambahkan edges
        for k in range(5):
            k_next: int = (k + 1) % 5
           
            # Radial: inner → center (single arrow)
            self.add_edge(inner_vertices[k], center, arrow_type=1)
           
            # Kite: outer → inner (double arrow)
            self.add_edge(outer_vertices[k], inner_vertices[k], arrow_type=2)
            self.add_edge(outer_vertices[k], inner_vertices[k_next], arrow_type=2)
       
        print(f"Seed cluster: N={len(self.vertices)}, E={len(self.edges)}")
   
    def deflate_single_arrow(self, i: int, j: int) -> List[Tuple[int, int, int]]:
        """
        Transformasi deflasi untuk single arrow (Persamaan 1.3)
        
        Input:
            i (int): Vertex ID sumber edge
            j (int): Vertex ID tujuan edge
        
        Output:
            List[Tuple[int, int, int]]: List of (vi, vj, arrow_type) untuk edges baru
                                        Format: [(i, l, 2), (k, l, 1), (k, m, 1), 
                                                 (i, m, 2), (j, k, 2)]
        
        Algoritma:
            1. Hitung vektor v_ij = r_j - r_i
            2. Buat 3 vertex baru:
               - k: di sepanjang ij, jarak φ⁻¹ dari j
               - l: rotasi -72° dari i, jarak φ⁻¹
               - m: rotasi +72° dari i, jarak φ⁻¹
            3. Return 5 edges baru yang menggantikan 1 edge lama
        
        Matematis:
            r_k = r_j - φ⁻¹ v_ij
            r_l = r_i + φ⁻¹ R(-72°) v_ij
            r_m = r_i + φ⁻¹ R(+72°) v_ij
        """
        pos_i: NDArray[np.float64] = self.vertices[i]
        pos_j: NDArray[np.float64] = self.vertices[j]
        vec_ij: NDArray[np.float64] = pos_j - pos_i
       
        pos_k: NDArray[np.float64] = pos_j - self.phi_inv * vec_ij
        R_neg72: NDArray[np.float64] = self.rotation_matrix(-72)
        pos_l: NDArray[np.float64] = pos_i + self.phi_inv * (R_neg72 @ vec_ij)
        R_pos72: NDArray[np.float64] = self.rotation_matrix(72)
        pos_m: NDArray[np.float64] = pos_i + self.phi_inv * (R_pos72 @ vec_ij)
       
        k: int = self.add_vertex(pos_k)
        l: int = self.add_vertex(pos_l)
        m: int = self.add_vertex(pos_m)
       
        # Original directions
        new_edges: List[Tuple[int, int, int]] = [
            (i, l, 2),
            (k, l, 1),
            (k, m, 1),
            (i, m, 2),
            (j, k, 2),
        ]
        return new_edges
   
    def deflate_double_arrow(self, i: int, j: int) -> List[Tuple[int, int, int]]:
        """
        Transformasi deflasi untuk double arrow (Persamaan 1.4)
        
        Input:
            i (int): Vertex ID sumber edge
            j (int): Vertex ID tujuan edge
        
        Output:
            List[Tuple[int, int, int]]: List of (vi, vj, arrow_type) untuk edges baru
                                        Format: [(g, i, 1), (j, g, 2), 
                                                 (j, h, 2), (h, i, 1)]
        
        Algoritma:
            1. Hitung vektor v_ij = r_j - r_i
            2. Buat 2 vertex baru:
               - g: rotasi -36° dari i, jarak φ⁻¹
               - h: rotasi +36° dari i, jarak φ⁻¹
            3. Return 4 edges baru yang menggantikan 1 edge lama
        
        Matematis:
            r_g = r_i + φ⁻¹ R(-36°) v_ij
            r_h = r_i + φ⁻¹ R(+36°) v_ij
        """
        pos_i: NDArray[np.float64] = self.vertices[i]
        pos_j: NDArray[np.float64] = self.vertices[j]
        vec_ij: NDArray[np.float64] = pos_j - pos_i
       
        R_neg36: NDArray[np.float64] = self.rotation_matrix(-36)
        pos_g: NDArray[np.float64] = pos_i + self.phi_inv * (R_neg36 @ vec_ij)
        R_pos36: NDArray[np.float64] = self.rotation_matrix(36)
        pos_h: NDArray[np.float64] = pos_i + self.phi_inv * (R_pos36 @ vec_ij)
       
        g: int = self.add_vertex(pos_g)
        h: int = self.add_vertex(pos_h)
       
        # Reversed directions to fix the issue
        new_edges: List[Tuple[int, int, int]] = [
            (g, i, 1),
            (j, g, 2),
            (j, h, 2),
            (h, i, 1),
        ]
        return new_edges
   
    def deflate_once(self) -> None:
        """
        Lakukan satu iterasi deflasi pada semua edges
        
        Input: None
        
        Output: None (modifikasi self.vertices dan self.edges in-place)
        
        Algoritma:
            1. Simpan semua edges lama
            2. Clear edges
            3. Untuk setiap edge lama:
               - Jika single arrow: panggil deflate_single_arrow()
               - Jika double arrow: panggil deflate_double_arrow()
            4. Deduplikasi edges (hapus duplikat)
            5. Normalisasi semua vertex: r' = φ * r
        
        Efek:
            - Jumlah vertices meningkat ~φ² kali
            - Jumlah edges meningkat ~φ² kali
            - Pattern menjadi lebih detail
        """
        old_edges: Dict[Tuple[int, int], int] = dict(self.edges)
        self.edges.clear()
       
        edges_to_add: List[Tuple[int, int, int]] = []
       
        for (i, j), arrow_type in old_edges.items():
            if arrow_type == 1:
                new_edges: List[Tuple[int, int, int]] = self.deflate_single_arrow(i, j)
            elif arrow_type == 2:
                new_edges: List[Tuple[int, int, int]] = self.deflate_double_arrow(i, j)
            else:
                continue
            edges_to_add.extend(new_edges)
       
        # Improved deduplication to avoid double counting
        edge_dict: Dict[Tuple[int, int], int] = {}
        for vi, vj, atype in edges_to_add:
            key: Tuple[int, int] = (vi, vj)
            reverse_key: Tuple[int, int] = (vj, vi)
           
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
       
        # Normalisasi: scale semua vertices dengan φ
        for vid in self.vertices:
            self.vertices[vid] *= self.phi
       
        print(f"After deflation: N={len(self.vertices)}, E={len(self.edges)}")
   
    def get_statistics(self) -> Tuple[int, int]:
        """
        Return statistik lattice
        
        Input: None
        
        Output:
            Tuple[int, int]: (N, E) di mana
                            N = jumlah vertices
                            E = jumlah edges
        
        Contoh:
            >>> N, E = get_statistics()
            >>> print(f"N={N}, E={E}")
            N=11, E=15
        """
        return len(self.vertices), len(self.edges)


def analyze_overlapping_edges(
    edges: Dict[Tuple[int, int], int],
    vertices: Dict[int, NDArray[np.float64]]
) -> Tuple[List[List[Tuple[Tuple[int, int], int]]], Dict[Tuple[int, int], int]]:
    """
    Analisis edges yang tumpang tindih (berbagi kedua vertex yang sama)
    
    Input:
        edges (Dict[Tuple[int, int], int]): Dictionary (i, j) → arrow_type
        vertices (Dict[int, NDArray]): Dictionary vertex_id → koordinat [x, y]
    
    Output:
        Tuple berisi:
            overlapping_groups (List[List[Tuple[Tuple[int, int], int]]]): 
                List of groups, setiap group berisi edges yang overlap
                Format: [[((i1, j1), type1), ((i2, j2), type2)], ...]
            
            unique_edges (Dict[Tuple[int, int], int]): 
                Dictionary edges yang tidak overlap
                Format: {(i, j): arrow_type}
    
    Algoritma:
        1. Group edges berdasarkan unordered vertex pair
        2. Jika group punya >1 edge, masukkan ke overlapping_groups
        3. Jika group punya 1 edge, masukkan ke unique_edges
    
    Contoh:
        >>> edges = {(0, 1): 1, (1, 0): 2}  # Overlap (vertex pair sama)
        >>> overlapping, unique = analyze_overlapping_edges(edges, vertices)
        >>> len(overlapping)  # 1 group
        1
    """
    # Group edges by unordered vertex pair
    edge_groups: Dict[frozenset, List[Tuple[Tuple[int, int], int]]] = defaultdict(list)
    
    for (i, j), arrow_type in edges.items():
        # Create unordered key (frozenset agar {i,j} sama dengan {j,i})
        edge_key: frozenset = frozenset([i, j])
        edge_groups[edge_key].append(((i, j), arrow_type))
    
    overlapping_groups: List[List[Tuple[Tuple[int, int], int]]] = []
    unique_edges: Dict[Tuple[int, int], int] = {}
    
    for edge_key, edge_list in edge_groups.items():
        if len(edge_list) > 1:
            # Ada overlap
            overlapping_groups.append(edge_list)
        else:
            # Unique edge
            (i, j), arrow_type = edge_list[0]
            unique_edges[(i, j)] = arrow_type
    
    return overlapping_groups, unique_edges


def plot_with_overlapping_analysis(
    state: Dict[str, Any],
    ax: plt.Axes,
    show_arrows: bool = True,
    offset_overlapping: bool = False
) -> None:
    """
    Plot Penrose lattice dengan analisis overlapping edges
    
    Input:
        state (Dict[str, Any]): Dictionary berisi:
            - 'vertices': Dict[int, NDArray] - vertex positions
            - 'edges': Dict[Tuple[int, int], int] - edge connections
            - 'title': str - judul plot
            - 'N': int - jumlah vertices
            - 'E': int - jumlah edges
        
        ax (plt.Axes): Matplotlib axes untuk plotting
        
        show_arrows (bool): Jika True, tampilkan arrows di mid-point edges
        
        offset_overlapping (bool): Jika True, pisahkan edges yang overlap
                                   dengan offset perpendicular
    
    Output:
        None (modifikasi ax in-place)
    
    Algoritma:
        1. Analisis overlapping dengan analyze_overlapping_edges()
        2. Plot unique edges dengan warna sesuai arrow_type
        3. Plot overlapping edges (dengan/tanpa offset)
        4. Plot arrows di mid-point (opsional)
        5. Plot vertices sebagai scatter points
        6. Set title, labels, dan formatting
    """
    vertices: Dict[int, NDArray[np.float64]] = state['vertices']
    edges: Dict[Tuple[int, int], int] = state['edges']
    
    # Analisis overlapping
    overlapping_groups, unique_edges = analyze_overlapping_edges(edges, vertices)
    
    # Plot unique edges (tidak overlap)
    for (i, j), arrow_type in unique_edges.items():
        x1, y1 = vertices[i]
        x2, y2 = vertices[j]
        
        color: str = 'blue' if arrow_type == 1 else 'red'
        linewidth: float = 1.2 if arrow_type == 1 else 1.8
        
        ax.plot([x1, x2], [y1, y2], 
                color=color, linewidth=linewidth, alpha=0.6, zorder=1,
                label='_nolegend_')
    
    # Plot overlapping edges
    for group in overlapping_groups:
        for idx, ((i, j), arrow_type) in enumerate(group):
            x1, y1 = vertices[i]
            x2, y2 = vertices[j]
            
            color: str = 'blue' if arrow_type == 1 else 'red'
            linewidth: float = 1.2 if arrow_type == 1 else 1.8
            
            if offset_overlapping and len(group) > 1:
                # Offset perpendicular ke edge
                dx: float = x2 - x1
                dy: float = y2 - y1
                length: float = np.sqrt(dx**2 + dy**2)
                
                if length > 0:
                    # Vektor perpendicular (rotate 90 degrees)
                    perp_x: float = -dy / length
                    perp_y: float = dx / length
                    
                    # Offset amount
                    offset_amount: float = 0.05 * length * (idx - 0.5)
                    
                    x1_off: float = x1 + perp_x * offset_amount
                    y1_off: float = y1 + perp_y * offset_amount
                    x2_off: float = x2 + perp_x * offset_amount
                    y2_off: float = y2 + perp_y * offset_amount
                    
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
            
            mid_x: float = (x1 + x2) / 2
            mid_y: float = (y1 + y2) / 2
            
            dx: float = x2 - x1
            dy: float = y2 - y1
            
            length: float = np.sqrt(dx**2 + dy**2)
            if length > 0:
                arrow_scale: float = 0.15
                dx_norm: float = (dx / length) * length * arrow_scale
                dy_norm: float = (dy / length) * length * arrow_scale
                
                color: str = 'darkblue' if arrow_type == 1 else 'darkred'
                
                ax.quiver(mid_x, mid_y, dx_norm, dy_norm,
                        angles='xy', scale_units='xy', scale=1,
                        color=color, width=0.008, headwidth=4, headlength=5,
                        alpha=0.9, zorder=3)
    
    # Plot vertices
    xs: List[float] = [pos[0] for pos in vertices.values()]
    ys: List[float] = [pos[1] for pos in vertices.values()]
    ax.scatter(xs, ys, c='black', s=18, zorder=4, alpha=0.9, 
                edgecolors='white', linewidths=0.5)
    
    # Formatting
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.2, linestyle='--')
    
    # Count overlapping
    n_overlap: int = len(overlapping_groups)
    total_overlap_edges: int = sum(len(g) for g in overlapping_groups)
    unique_count: int = len(unique_edges)
    
    title: str = f"{state['title']}\n"
    title += f"N={state['N']}, E={state['E']} "
    title += f"(Unique: {unique_count}, Overlapping: {total_overlap_edges} in {n_overlap} pairs)"
    
    ax.set_title(title, fontsize=11, fontweight='bold')
    ax.set_xlabel('x', fontsize=10)
    ax.set_ylabel('y', fontsize=10)
    
    # Equal limits
    if len(xs) > 0:
        max_coord: float = max(max(abs(x) for x in xs), max(abs(y) for y in ys))
        margin: float = max_coord * 0.15
        ax.set_xlim(-max_coord - margin, max_coord + margin)
        ax.set_ylim(-max_coord - margin, max_coord + margin)


def plot_overlapping_iterations(
    state0: Dict[str, Any],
    state1: Dict[str, Any],
    ax: plt.Axes,
    show_arrows: bool = False
) -> None:
    """
    Plot 2 iterasi yang saling tumpang tindih dalam satu gambar
    
    Input:
        state0 (Dict[str, Any]): State iterasi pertama
            - 'vertices': Dict[int, NDArray]
            - 'edges': Dict[Tuple[int, int], int]
            - 'N': int
            - 'E': int
        
        state1 (Dict[str, Any]): State iterasi kedua
            - 'vertices': Dict[int, NDArray]
            - 'edges': Dict[Tuple[int, int], int]
            - 'N': int
            - 'E': int
        
        ax (plt.Axes): Matplotlib axes untuk plotting
        
        show_arrows (bool): Jika True, tampilkan arrows (default: False)
    
    Output:
        None (modifikasi ax in-place)
    
    Visualisasi:
        - Iterasi 0: warna hitam
        - Iterasi 1: warna merah
        - Ditumpuk dalam satu plot untuk melihat pola deflasi
    """
    vertices0: Dict[int, NDArray[np.float64]] = state0['vertices']
    edges0: Dict[Tuple[int, int], int] = state0['edges']
    vertices1: Dict[int, NDArray[np.float64]] = state1['vertices']
    edges1: Dict[Tuple[int, int], int] = state1['edges']
    
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
    xs0: List[float] = [pos[0] for pos in vertices0.values()]
    ys0: List[float] = [pos[1] for pos in vertices0.values()]
    ax.scatter(xs0, ys0, c='black', s=20, zorder=5, alpha=0.8,
                edgecolors='white', linewidths=0.5)
    
    # Plot vertices iterasi 1 (merah)
    xs1: List[float] = [pos[0] for pos in vertices1.values()]
    ys1: List[float] = [pos[1] for pos in vertices1.values()]
    ax.scatter(xs1, ys1, c='red', s=20, zorder=6, alpha=0.8,
                edgecolors='white', linewidths=0.5)
    
    # Formatting
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.2, linestyle='--')
    
    title: str = f"Iteration 0 (Black) & Iteration 1 (Red) Overlapped\n"
    title += f"Iter 0: N={state0['N']}, E={state0['E']} | "
    title += f"Iter 1: N={state1['N']}, E={state1['E']}"
    
    ax.set_title(title, fontsize=13, fontweight='bold')
    ax.set_xlabel('x', fontsize=10)
    ax.set_ylabel('y', fontsize=10)
    
    # Equal limits based on both iterations
    all_xs: List[float] = xs0 + xs1
    all_ys: List[float] = ys0 + ys1
    if len(all_xs) > 0:
        max_coord: float = max(max(abs(x) for x in all_xs), max(abs(y) for y in all_ys))
        margin: float = max_coord * 0.15
        ax.set_xlim(-max_coord - margin, max_coord + margin)
        ax.set_ylim(-max_coord - margin, max_coord + margin)


def main() -> None:
    """
    Main program untuk generate dan visualisasi Penrose Lattice
    
    Input: None
    
    Output: None
    
    Proses:
        1. Buat seed cluster (iterasi 0)
        2. Lakukan deflasi 3 kali (iterasi 1, 2, 3)
        3. Simpan state setiap iterasi
        4. Plot semua iterasi dalam 2×2 grid
        5. Plot iterasi 0 dan 1 yang tumpang tindih
        6. Save gambar ke file PNG
        7. Print statistik dan analisis
    
    Output files:
        - penrose_all_iterations_corrected.png
        - penrose_iter0_iter1_overlapped_corrected.png
    """
    print("="*70)
    print("PENROSE LATTICE - FINAL VERSION (WITH STATIC TYPING)")
    print("="*70)
    
    penrose: PenroseLattice = PenroseLattice()
    
    # Seed
    print("\n=== Seed Cluster ===")
    penrose.create_seed_cluster()
    
    # Save states
    states: List[Dict[str, Any]] = []
    N, E = penrose.get_statistics()
    states.append({
        'title': 'Iteration 0 (Seed)',
        'vertices': dict(penrose.vertices),
        'edges': dict(penrose.edges),
        'N': N,
        'E': E
    })
    
    # Iterasi 1-3
    for it in range(1, 4):
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
    axes_flat: NDArray = axes.flatten()
    
    for idx, state in enumerate(states):
        plot_with_overlapping_analysis(state, axes_flat[idx], 
                                        show_arrows=True, 
                                        offset_overlapping=False)
    
    from matplotlib.lines import Line2D
    legend_elements: List[Line2D] = [
        Line2D([0], [0], color='blue', linewidth=2.5, 
                marker='>', markersize=8, markerfacecolor='darkblue',
                label='Single Arrow'),
        Line2D([0], [0], color='red', linewidth=2.5, 
                marker='>', markersize=8, markerfacecolor='darkred',
                label='Double Arrow')
    ]
    axes_flat[0].legend(handles=legend_elements, loc='upper right', fontsize=11)
    
    plt.suptitle('Penrose Lattice - Semua Iterasi (Static Typing Version)', 
                    fontsize=16, fontweight='bold', y=0.998)
    plt.tight_layout()
    plt.savefig('penrose_all_iterations_typed.png', dpi=200, bbox_inches='tight')
    print("\n[SAVED] penrose_all_iterations_typed.png")
    
    # ========== PLOT 2: Iterasi 0 dan 1 Tumpang Tindih ==========
    fig2, ax2 = plt.subplots(1, 1, figsize=(12, 12))
    
    # show_arrows=False untuk menghilangkan arrow
    plot_overlapping_iterations(states[0], states[1], ax2, show_arrows=False)
    
    legend_elements_2: List[Line2D] = [
        Line2D([0], [0], color='black', linewidth=2.5, 
                label='Iteration 0'),
        Line2D([0], [0], color='red', linewidth=2.5, 
                label='Iteration 1')
    ]
    ax2.legend(handles=legend_elements_2, loc='upper right', fontsize=11)
    
    plt.tight_layout()
    plt.savefig('penrose_iter0_iter1_overlapped_typed.png', dpi=200, bbox_inches='tight')
    print("\n[SAVED] penrose_iter0_iter1_overlapped_typed.png")
    
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
        n_overlap: int = len(overlapping_groups)
        total_overlap: int = sum(len(g) for g in overlapping_groups)
        
        print(f"{state['title']:25s}: "
                f"Unique edges={len(unique_edges)}, "
                f"Overlapping={total_overlap} edges in {n_overlap} pairs")
    
    print("="*70)
    print("\n✅ File dengan static typing lengkap telah dibuat!")
    print("📝 Setiap method/function sekarang memiliki type hints yang jelas")
    
    # plt.show()


if __name__ == "__main__":
    main()
