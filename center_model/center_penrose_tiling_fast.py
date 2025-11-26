"""
Center Penrose Tiling - Dual Graph (Center Model)
Generate center model dari vertex model Penrose lattice
"""

import numpy as np
import pickle
import time
import argparse
from collections import defaultdict


def print_separator(char="=", length=80):
    """Print separator line"""
    print(char * length)


class CenterPenroseLattice:
    """
    Generator Center Model (Dual Graph) dari Penrose Lattice
    
    Center model: setiap rhombus (face) menjadi vertex baru,
    dua center dihubungkan jika rhombi-nya berbagi satu edge.
    """
    
    def __init__(self):
        """Inisialisasi"""
        self.vertices = {}  # Original vertex coordinates
        self.edges = []     # Original edges
        self.N = 0          # Original vertex count
        
        # Center model data
        self.faces = []     # List of rhombi (4-tuples of vertex indices)
        self.centers = []   # Center coordinates untuk setiap face
        self.dual_edges = []  # Edges antara centers
        self.N_center = 0   # Jumlah center vertices
        self.E_center = 0   # Jumlah center edges
        self.iteration = 0  # Iteration number
    
    def load_vertex_model(self, filename='vertex_model/data/penrose_lattice_data.npz'):
        """Load vertex model dari file npz"""
        print(f"\n[LOADING] Reading vertex model from {filename}...")
        
        try:
            data = np.load(filename)
            vertex_coords = data['vertex_coords']
            edge_list = data['edge_list']
            
            self.N = len(vertex_coords)
            self.vertices = {i: coord for i, coord in enumerate(vertex_coords)}
            self.edges = edge_list
            self.iteration = int(data.get('iteration', 0))
            
            print(f"  ✓ Loaded N={self.N} vertices")
            print(f"  ✓ Loaded E={len(edge_list)} edges")
            print(f"  ✓ Iteration: {self.iteration}")
            
        except FileNotFoundError:
            print(f"  ✗ File not found: {filename}")
            print("    Run penrose_tiling_fast.py first to generate vertex model")
            exit(1)
    
    def detect_rhombi(self):
        """
        Deteksi rhombi (4-vertex cycles) dalam graph
        Setiap rhombus menjadi satu face di center model
        """
        print(f"\n[DETECTING RHOMBI] Finding 4-cycles...")
        t0 = time.time()
        
        # Build adjacency list
        adj = defaultdict(set)
        edges_set = set()
        
        for u, v in self.edges:
            adj[u].add(v)
            adj[v].add(u)
            edges_set.add(tuple(sorted((u, v))))
        
        # Find 4-cycles (rhombi)
        processed_quads = set()
        
        for u in range(self.N):
            neighbors = list(adj[u])
            
            # Cek setiap pasangan tetangga dari u
            for i in range(len(neighbors)):
                for j in range(i + 1, len(neighbors)):
                    v = neighbors[i]
                    w = neighbors[j]
                    
                    # Cari tetangga bersama antara v dan w (selain u)
                    common = adj[v].intersection(adj[w])
                    common.discard(u)
                    
                    if common:
                        x = common.pop()  # Ambil tetangga bersama
                        
                        # Verifikasi ini valid rhombus: u-v-x-w-u
                        # Check apakah keempat edge ada
                        quad_edges = [
                            tuple(sorted((u, v))),
                            tuple(sorted((v, x))),
                            tuple(sorted((x, w))),
                            tuple(sorted((w, u)))
                        ]
                        
                        if all(e in edges_set for e in quad_edges):
                            quad = tuple(sorted((u, v, w, x)))
                            if quad not in processed_quads:
                                self.faces.append(quad)
                                processed_quads.add(quad)
        
        t1 = time.time()
        print(f"  ✓ Found {len(self.faces)} rhombi in {(t1-t0)*1000:.2f} ms")
    
    def compute_centers(self):
        """Hitung koordinat pusat untuk setiap rhombus"""
        print(f"\n[COMPUTING CENTERS] Calculating center coordinates...")
        t0 = time.time()
        
        vertex_coords = np.array([self.vertices[i] for i in range(self.N)])
        
        for quad in self.faces:
            # Koordinat rata-rata dari 4 sudut
            coords = vertex_coords[list(quad)]
            center = np.mean(coords, axis=0)
            self.centers.append(center)
        
        self.centers = np.array(self.centers)
        self.N_center = len(self.centers)
        
        t1 = time.time()
        print(f"  ✓ Computed {self.N_center} center coordinates in {(t1-t0)*1000:.2f} ms")
    
    def build_dual_edges(self):
        """
        Bangun edges antara centers (dual graph)
        Dua center dihubungkan jika rhombi-nya berbagi satu edge
        """
        print(f"\n[BUILDING DUAL EDGES] Constructing dual graph...")
        t0 = time.time()
        
        # Build edge-to-faces mapping
        edge_to_faces = defaultdict(list)
        edges_set = set()
        
        for u, v in self.edges:
            edges_set.add(tuple(sorted((u, v))))
        
        for idx, quad in enumerate(self.faces):
            # Setiap rhombus punya 4 edges
            # Cek semua kombinasi 2 dari 4 vertices
            for i in range(4):
                for j in range(i + 1, 4):
                    edge = tuple(sorted((quad[i], quad[j])))
                    if edge in edges_set:
                        edge_to_faces[edge].append(idx)
        
        # Jika dua faces berbagi satu edge, hubungkan center-nya
        for edge, face_indices in edge_to_faces.items():
            if len(face_indices) == 2:
                f1, f2 = face_indices
                self.dual_edges.append((f1, f2))
        
        self.E_center = len(self.dual_edges)
        
        t1 = time.time()
        print(f"  ✓ Built {self.E_center} dual edges in {(t1-t0)*1000:.2f} ms")
    
    def save_center_model(self):
        """Save center model ke file npz dan pkl"""
        print(f"\n[SAVING] Writing center model to files...")
        
        # Prepare data
        center_ids = np.arange(self.N_center, dtype=np.int32)
        center_coords = self.centers
        edge_list = np.array(self.dual_edges, dtype=np.int32)
        
        # Save as numpy
        npz_file = 'center_model/data/center_model_penrose_lattice.npz'
        np.savez(npz_file,
                 vertex_ids=center_ids,
                 vertex_coords=center_coords,
                 edge_list=edge_list,
                 N=self.N_center,
                 E=self.E_center,
                 iteration=self.iteration,
                 phi=(1 + np.sqrt(5)) / 2)
        print(f"  ✓ Saved to numpy: {npz_file}")
        
        # Save as pickle (dengan format dictionary)
        pkl_file = 'center_model/data/center_model_penrose_lattice.pkl'
        
        data_dict = {
            'vertices': {i: coord for i, coord in enumerate(center_coords)},
            'edges': {tuple(e): 1 for e in edge_list},  # All edges have same type
            'N': self.N_center,
            'E': self.E_center,
            'iteration': self.iteration,
            'phi': (1 + np.sqrt(5)) / 2,
            'model_type': 'center_model'
        }
        
        with open(pkl_file, 'wb') as f:
            pickle.dump(data_dict, f)
        print(f"  ✓ Saved to pickle: {pkl_file}")
    
    def print_statistics(self):
        """Print statistik center model"""
        print_separator()
        print("CENTER MODEL STATISTICS")
        print_separator()
        print(f"Iteration: {self.iteration}")
        print()
        print(f"Original vertex model:")
        print(f"  Vertices (atoms):    {self.N}")
        print(f"  Edges (bonds):       {len(self.edges)}")
        print()
        print(f"Center model (dual graph):")
        print(f"  Centers (faces):     {self.N_center}")
        print(f"  Dual edges:          {self.E_center}")
        print()
        print(f"Ratio:")
        print(f"  N_center/N_vertex:   {self.N_center/self.N:.4f}")
        print(f"  E_center/E_vertex:   {self.E_center/len(self.edges):.4f}")
        print_separator()


def main():
    """Main program"""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Generate Center Model from Penrose Lattice')
    parser.add_argument('--iteration', type=int, default=None,
                        help='Iteration number to use (default: use latest penrose_lattice_data.npz)')
    args = parser.parse_args()
    
    print_separator()
    print("PENROSE LATTICE - CENTER MODEL GENERATOR")
    print("Converting Vertex Model to Dual Graph (Center Model)")
    print_separator()
    
    # Determine which file to load
    if args.iteration is not None:
        filename = f'vertex_model/data/penrose_lattice_iter{args.iteration}.npz'
        print(f"\nUsing iteration {args.iteration} data: {filename}")
    else:
        filename = 'vertex_model/data/penrose_lattice_data.npz'
        print(f"\nUsing latest data: {filename}")
    
    # Initialize
    center_lattice = CenterPenroseLattice()
    
    # Load vertex model
    center_lattice.load_vertex_model(filename)
    
    # Detect rhombi
    center_lattice.detect_rhombi()
    
    # Compute center coordinates
    center_lattice.compute_centers()
    
    # Build dual edges
    center_lattice.build_dual_edges()
    
    # Save center model
    center_lattice.save_center_model()
    
    # Print statistics
    print()
    center_lattice.print_statistics()
    
    print()
    print("✅ Center model generation completed!")
    print("📁 Files saved:")
    print("   - center_model/data/center_model_penrose_lattice.npz")
    print("   - center_model/data/center_model_penrose_lattice.pkl")
    print()


if __name__ == "__main__":
    main()
