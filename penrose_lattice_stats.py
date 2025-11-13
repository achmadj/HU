"""
Penrose Lattice - Statistik Vertices dan Edges
Versi tanpa plotting, fokus pada perhitungan dan analisis statistik
"""

import numpy as np
import pickle
from typing import Dict, List, Tuple
from numpy.typing import NDArray


class PenroseLatticeStats:
    """
    Penrose Lattice dengan fokus pada statistik vertices dan edges
    
    Attributes:
        phi (float): Golden ratio φ = (1+√5)/2 ≈ 1.618
        phi_inv (float): Inverse golden ratio 1/φ ≈ 0.618
        vertices (Dict[int, NDArray[np.float64]]): Dictionary vertex_id → koordinat [x, y]
        edges (Dict[Tuple[int, int], int]): Dictionary (i, j) → arrow_type (1 atau 2)
        vertex_counter (int): Counter untuk ID vertex berikutnya
    """
   
    def __init__(self) -> None:
        """Inisialisasi Penrose Lattice"""
        self.phi: float = (1 + np.sqrt(5)) / 2
        self.phi_inv: float = 1 / self.phi
       
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
            i (int): Vertex ID sumber
            j (int): Vertex ID tujuan
            arrow_type (int): Tipe panah (1 = single, 2 = double)
        
        Output:
            None (modifikasi self.edges in-place)
        """
        if i != j:
            self.edges[(i, j)] = arrow_type
   
    def create_seed_cluster(self) -> None:
        """
        Membuat seed cluster awal (Iterasi 0)
        
        Output: None (modifikasi self.vertices dan self.edges in-place)
        
        Total: N=11 vertices, E=15 edges
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
   
    def deflate_single_arrow(self, i: int, j: int) -> List[Tuple[int, int, int]]:
        """
        Transformasi deflasi untuk single arrow
        
        Input:
            i (int): Vertex ID sumber edge
            j (int): Vertex ID tujuan edge
        
        Output:
            List[Tuple[int, int, int]]: List of (vi, vj, arrow_type)
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
        Transformasi deflasi untuk double arrow
        
        Input:
            i (int): Vertex ID sumber edge
            j (int): Vertex ID tujuan edge
        
        Output:
            List[Tuple[int, int, int]]: List of (vi, vj, arrow_type)
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
        
        Output: None (modifikasi self.vertices dan self.edges in-place)
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
       
        # Deduplication
        edge_dict: Dict[Tuple[int, int], int] = {}
        for vi, vj, atype in edges_to_add:
            key: Tuple[int, int] = (vi, vj)
            reverse_key: Tuple[int, int] = (vj, vi)
           
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
       
        # Normalisasi: scale semua vertices dengan φ
        for vid in self.vertices:
            self.vertices[vid] *= self.phi
   
    def get_statistics(self) -> Tuple[int, int]:
        """
        Return statistik lattice
        
        Output:
            Tuple[int, int]: (N, E) = (jumlah vertices, jumlah edges)
        """
        return len(self.vertices), len(self.edges)
    
    def get_edge_type_count(self) -> Tuple[int, int]:
        """
        Hitung jumlah single arrow dan double arrow
        
        Output:
            Tuple[int, int]: (single_count, double_count)
        """
        single_count: int = sum(1 for arrow_type in self.edges.values() if arrow_type == 1)
        double_count: int = sum(1 for arrow_type in self.edges.values() if arrow_type == 2)
        return single_count, double_count
    
    def get_bounding_box(self) -> Tuple[float, float, float, float]:
        """
        Hitung bounding box dari semua vertices
        
        Output:
            Tuple[float, float, float, float]: (x_min, x_max, y_min, y_max)
        """
        if len(self.vertices) == 0:
            return (0.0, 0.0, 0.0, 0.0)
        
        xs: List[float] = [pos[0] for pos in self.vertices.values()]
        ys: List[float] = [pos[1] for pos in self.vertices.values()]
        
        return (min(xs), max(xs), min(ys), max(ys))
    
    def get_average_edge_length(self) -> float:
        """
        Hitung rata-rata panjang edge
        
        Output:
            float: Average edge length
        """
        if len(self.edges) == 0:
            return 0.0
        
        total_length: float = 0.0
        for (i, j) in self.edges.keys():
            pos_i: NDArray[np.float64] = self.vertices[i]
            pos_j: NDArray[np.float64] = self.vertices[j]
            length: float = float(np.linalg.norm(pos_j - pos_i))
            total_length += length
        
        return total_length / len(self.edges)


def print_separator(char: str = "=", length: int = 80) -> None:
    """Print separator line"""
    print(char * length)


def main() -> None:
    """
    Main program untuk generate statistik Penrose Lattice
    """
    print_separator()
    print("PENROSE LATTICE - STATISTICS ONLY")
    print("Focus: Vertices & Edges Count per Iteration")
    print_separator()
    
    penrose: PenroseLatticeStats = PenroseLatticeStats()
    
    # Seed cluster
    print("\n[Iteration 0] Creating seed cluster...")
    penrose.create_seed_cluster()
    
    N, E = penrose.get_statistics()
    single, double = penrose.get_edge_type_count()
    x_min, x_max, y_min, y_max = penrose.get_bounding_box()
    avg_edge = penrose.get_average_edge_length()
    
    print(f"  ✓ Vertices (N): {N}")
    print(f"  ✓ Edges (E):    {E}")
    print(f"    - Single arrows:  {single}")
    print(f"    - Double arrows:  {double}")
    print(f"  ✓ Bounding box: x ∈ [{x_min:.3f}, {x_max:.3f}], y ∈ [{y_min:.3f}, {y_max:.3f}]")
    print(f"  ✓ Avg edge length: {avg_edge:.6f}")
    
    # Store statistics for analysis
    stats: List[Dict[str, any]] = [{
        'iteration': 0,
        'N': N,
        'E': E,
        'single': single,
        'double': double,
        'avg_edge_length': avg_edge
    }]
    
    # Iterasi 1-3
    import time
    for iteration in range(1, 6):
        print(f"\n[Iteration {iteration}] Performing deflation...")
        t0 = time.time()
        penrose.deflate_once()
        t1 = time.time()
        
        N, E = penrose.get_statistics()
        single, double = penrose.get_edge_type_count()
        x_min, x_max, y_min, y_max = penrose.get_bounding_box()
        avg_edge = penrose.get_average_edge_length()
        
        print(f"  ✓ Vertices (N): {N}")
        print(f"  ✓ Edges (E):    {E}")
        print(f"    - Single arrows:  {single}")
        print(f"    - Double arrows:  {double}")
        print(f"  ✓ Bounding box: x ∈ [{x_min:.3f}, {x_max:.3f}], y ∈ [{y_min:.3f}, {y_max:.3f}]")
        print(f"  ✓ Avg edge length: {avg_edge:.6f}")
        print(f"  ⏱️  Time: {(t1-t0)*1000:.2f} ms")
        
        stats.append({
            'iteration': iteration,
            'N': N,
            'E': E,
            'single': single,
            'double': double,
            'avg_edge_length': avg_edge
        })
    
    # Summary table
    print("\n")
    print_separator()
    print("SUMMARY TABLE")
    print_separator()
    print(f"{'Iteration':<12} {'N':<8} {'E':<8} {'Single':<8} {'Double':<8} {'Avg Edge':<12}")
    print_separator("-")
    
    for stat in stats:
        print(f"{stat['iteration']:<12} {stat['N']:<8} {stat['E']:<8} "
              f"{stat['single']:<8} {stat['double']:<8} {stat['avg_edge_length']:<12.6f}")
    
    print_separator("-")
    
    # Growth analysis
    print("\n")
    print_separator()
    print("GROWTH ANALYSIS")
    print_separator()
    
    phi: float = (1 + np.sqrt(5)) / 2
    phi_squared: float = phi ** 2
    
    print(f"Golden ratio φ = {phi:.6f}")
    print(f"φ² = {phi_squared:.6f}")
    print()
    
    for i in range(1, len(stats)):
        prev_N: int = stats[i-1]['N']
        curr_N: int = stats[i]['N']
        prev_E: int = stats[i-1]['E']
        curr_E: int = stats[i]['E']
        
        N_ratio: float = curr_N / prev_N if prev_N > 0 else 0
        E_ratio: float = curr_E / prev_E if prev_E > 0 else 0
        
        print(f"Iteration {i-1} → {i}:")
        print(f"  N growth: {prev_N:4d} → {curr_N:4d}  (ratio: {N_ratio:.4f}, expected: ~{phi_squared:.4f})")
        print(f"  E growth: {prev_E:4d} → {curr_E:4d}  (ratio: {E_ratio:.4f}, expected: ~{phi_squared:.4f})")
        print()
    
    # Expected values comparison
    print_separator()
    print("COMPARISON WITH EXPECTED VALUES")
    print_separator()
    
    expected_N: List[int] = [11, 31, 86, 226]
    
    print(f"{'Iteration':<12} {'N (Actual)':<15} {'N (Expected)':<15} {'Match':<10}")
    print_separator("-")
    
    for i, stat in enumerate(stats):
        actual: int = stat['N']
        expected: int = expected_N[i] if i < len(expected_N) else -1
        match: str = "✓" if actual == expected else "✗"
        
        print(f"{i:<12} {actual:<15} {expected:<15} {match:<10}")
    
    print_separator("-")
    
    # Final statistics
    print("\n")
    print_separator()
    print("FINAL STATISTICS (After Iteration 3)")
    print_separator()
    
    final_stat = stats[-1]
    print(f"Total vertices:        {final_stat['N']}")
    print(f"Total edges:           {final_stat['E']}")
    print(f"  - Single arrows:     {final_stat['single']}")
    print(f"  - Double arrows:     {final_stat['double']}")
    print(f"Average edge length:   {final_stat['avg_edge_length']:.6f}")
    print(f"Vertex counter:        {penrose.vertex_counter}")
    
    print_separator()
    
    # Save data untuk tight binding model
    print("\n")
    print_separator()
    print("SAVING DATA FOR TIGHT BINDING MODEL")
    print_separator()
    
    # Prepare data to save
    data_to_save = {
        'vertices': dict(penrose.vertices),
        'edges': dict(penrose.edges),
        'N': final_stat['N'],
        'E': final_stat['E'],
        'iteration': len(stats) - 1,
        'phi': penrose.phi,
        'single_arrows': final_stat['single'],
        'double_arrows': final_stat['double']
    }
    
    # Save as pickle
    pickle_file = 'penrose_lattice_data.pkl'
    with open(pickle_file, 'wb') as f:
        pickle.dump(data_to_save, f)
    print(f"  ✓ Saved to pickle: {pickle_file}")
    
    # Save as numpy arrays (alternative format)
    npz_file = 'penrose_lattice_data.npz'
    
    # Convert vertices dict to arrays
    vertex_ids = np.array(list(penrose.vertices.keys()), dtype=np.int32)
    vertex_coords = np.array(list(penrose.vertices.values()), dtype=np.float64)
    
    # Convert edges dict to arrays
    edge_list = np.array(list(penrose.edges.keys()), dtype=np.int32)  # (E, 2)
    edge_types = np.array(list(penrose.edges.values()), dtype=np.int32)  # (E,)
    
    np.savez(npz_file,
             vertex_ids=vertex_ids,
             vertex_coords=vertex_coords,
             edge_list=edge_list,
             edge_types=edge_types,
             N=final_stat['N'],
             E=final_stat['E'],
             iteration=len(stats) - 1,
             phi=penrose.phi)
    print(f"  ✓ Saved to numpy: {npz_file}")
    
    print_separator()
    print("\n✅ Computation completed successfully!")
    print("📊 All statistics generated without plotting")
    print("💾 Data saved for tight binding model analysis")
    print()


if __name__ == "__main__":
    main()
