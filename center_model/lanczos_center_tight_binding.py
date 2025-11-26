"""
Penrose Lattice Center Model - Lanczos Tight Binding
Kalkulasi tight binding menggunakan Lanczos method pada center model (dual graph)
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix, eye
from scipy.sparse.linalg import spsolve
from collections import deque, Counter
import time


class CenterModelLanczos:
    """
    Implementasi Lanczos Tight Binding untuk Center Model Penrose Lattice.
    Menggunakan scipy.sparse untuk efisiensi pada sistem besar.
    """
    
    def __init__(self, t: float = 1.0):
        """
        Inisialisasi
        
        Parameters:
        -----------
        t : float
            Hopping parameter (default: 1.0)
        """
        self.t = t
        self.vertices = {}
        self.raw_edges = []
        self.N = 0
        self.hamiltonian = None
        
        # Variabel Lanczos
        self.T_matrix = None
        self.krylov_basis = []
        
        # Hasil Akhir
        self.final_energy = None
        self.final_wavefunction = None
        
    def load_data(self, filename: str = 'center_model/data/center_model_penrose_lattice.npz'):
        """
        Load center model data dari file
        
        Parameters:
        -----------
        filename : str
            Path ke file npz center model
        """
        print(f"[1] Membaca center model data dari {filename}...")
        try:
            data = np.load(filename)
            self.N = int(data['N'])
            vertex_coords = data['vertex_coords']
            self.vertices = {i: coord for i, coord in enumerate(vertex_coords)}
            self.raw_edges = data['edge_list']
            print(f"    ✓ N = {self.N} center atoms")
            print(f"    ✓ E = {len(self.raw_edges)} dual edges")
        except FileNotFoundError:
            print("    ❌ File tidak ditemukan.")
            print("       Jalankan center_penrose_tiling_fast.py terlebih dahulu!")
            exit(1)
    
    def build_hamiltonian(self):
        """
        Membangun Hamiltonian sebagai SPARSE Matrix (CSR format).
        H_ij = -t jika i dan j terhubung, 0 otherwise
        """
        print(f"[2] Membangun Matriks Hamiltonian SPARSE...")
        
        # Prepare data for sparse matrix construction
        row_indices = []
        col_indices = []
        data = []
        
        # Isi elemen (Hopping)
        for i, j in self.raw_edges:
            row_indices.extend([i, j])
            col_indices.extend([j, i])
            data.extend([-self.t, -self.t])
        
        # Create sparse matrix in CSR format
        self.hamiltonian = csr_matrix((data, (row_indices, col_indices)), 
                                      shape=(self.N, self.N))
        
        print(f"    ✓ Matriks Sparse {self.N}x{self.N} siap.")
        # Cek ukuran memori (hanya elemen non-zero)
        nnz = self.hamiltonian.nnz
        mem_size = (nnz * 8 + nnz * 4 * 2) / (1024**2)  # data + indices
        print(f"    ℹ️  Non-zero elements: {nnz}, Ukuran Memori: {mem_size:.2f} MB")
    
    def run_lanczos_shift_invert(self, k_steps=3, sigma=1e-9):
        """
        Algoritma Lanczos dengan LINEAR SOLVER (spsolve).
        Menyelesaikan (H - σI)w' = v tanpa menghitung invers eksplisit.
        
        Parameters:
        -----------
        k_steps : int
            Jumlah iterasi Lanczos
        sigma : float
            Shift parameter untuk targeting eigenvalue di sekitar sigma
        """
        print(f"\n[3] Menyiapkan Shifted Matrix...")
        t0 = time.time()
        
        # 1. Matriks Shifted: M = H - sigma*I
        I = eye(self.N, format='csr')
        M = self.hamiltonian - sigma * I
        
        print(f"    ✓ Shifted matrix siap dalam {time.time()-t0:.4f} detik.")
        print(f"\n[4-9] Memulai Iterasi Lanczos ({k_steps} langkah)...")
        
        # Inisialisasi Vektor Awal
        np.random.seed(42)
        v_prev = np.zeros(self.N)
        v_curr = np.random.rand(self.N) - 0.5
        v_curr /= np.linalg.norm(v_curr)
        
        beta_prev = 0.0
        alphas = []
        betas = []
        self.krylov_basis = [v_curr]
        
        for j in range(k_steps):
            # Solve (H - σI)w' = v instead of computing w' = (H - σI)^-1 * v
            w = spsolve(M, v_curr)
            
            # Sisanya sama dengan Lanczos biasa
            alpha = np.dot(v_curr, w)
            alphas.append(alpha)
            
            w = w - alpha * v_curr - beta_prev * v_prev
            beta = np.linalg.norm(w)
            
            if beta < 1e-10: 
                print(f"    ! Konvergensi dini pada iterasi {j+1}.")
                break
                
            v_next = w / beta
            betas.append(beta)
            self.krylov_basis.append(v_next)
            
            v_prev = v_curr
            v_curr = v_next
            beta_prev = beta
        
        # Matriks Tridiagonal
        k_actual = len(alphas)
        self.T_matrix = np.zeros((k_actual, k_actual))
        for i in range(k_actual):
            self.T_matrix[i, i] = alphas[i]
            if i < k_actual - 1:
                self.T_matrix[i, i+1] = betas[i]
                self.T_matrix[i+1, i] = betas[i]
        
        print(f"    ✓ Lanczos selesai: {k_actual} iterasi")
    
    def diagonalize_T_and_reconstruct(self, sigma=1e-9):
        """
        Diagonalisasi T matrix dan rekonstruksi eigenstate
        
        Parameters:
        -----------
        sigma : float
            Shift parameter yang sama dengan run_lanczos_shift_invert
        """
        print(f"\n[10] Rekonstruksi Solusi...")
        
        # Diagonalisasi T matrix
        evals_T, evecs_T = np.linalg.eigh(self.T_matrix)
        
        # Pilih eigenvalue magnitude terbesar (karena shift-invert)
        idx_max = np.argmax(np.abs(evals_T))
        lambda_best = evals_T[idx_max]
        
        # Konversi kembali ke eigenvalue asli
        self.final_energy = sigma + (1.0 / lambda_best)
        print(f"    Energi (E): {self.final_energy:.10f}")
        
        # Rekonstruksi Wavefunction
        y_vec = evecs_T[:, idx_max]
        self.final_wavefunction = np.zeros(self.N)
        
        k = len(y_vec)
        for i in range(k):
            self.final_wavefunction += y_vec[i] * self.krylov_basis[i]
        self.final_wavefunction /= np.linalg.norm(self.final_wavefunction)
        
        print(f"    ✓ Wavefunction direkonstruksi (normalisasi: {np.linalg.norm(self.final_wavefunction):.6f})")
    
    def compute_bipartite_sublattices(self):
        """
        Hitung sublattice assignment menggunakan BFS
        (untuk pewarnaan bipartite graph)
        
        Returns:
        --------
        dict : mapping dari vertex_id ke 'A' atau 'B'
        """
        adj = {i: [] for i in range(self.N)}
        for i, j in self.raw_edges:
            adj[i].append(j)
            adj[j].append(i)
        
        sublattice_map = {}
        queue = deque()
        
        for start in range(self.N):
            if start not in sublattice_map:
                sublattice_map[start] = 'A'
                queue.append(start)
                
                while queue:
                    u = queue.popleft()
                    lbl = sublattice_map[u]
                    opp = 'B' if lbl == 'A' else 'A'
                    
                    for v in adj[u]:
                        if v not in sublattice_map:
                            sublattice_map[v] = opp
                            queue.append(v)
        
        count_A = list(sublattice_map.values()).count('A')
        count_B = list(sublattice_map.values()).count('B')
        print(f"    Sublattice A: {count_A}, Sublattice B: {count_B}")
        
        return sublattice_map
    
    def analyze_localization(self):
        """
        Analisis Lokalisasi Wavefunction berdasarkan Koordinasi (z).
        Menghitung persentase probabilitas |psi|^2 pada situs dengan z berbeda.
        """
        print("\n[ANALISIS LOKALISASI]")
        
        # 1. Hitung Degree (Jumlah Tetangga) untuk setiap center atom
        degrees = np.zeros(self.N, dtype=int)
        for i, j in self.raw_edges:
            degrees[i] += 1
            degrees[j] += 1
        
        # 2. Hitung Distribusi Probabilitas
        prob = np.abs(self.final_wavefunction)**2
        prob_by_degree = {}
        
        # Center model biasanya punya koordinasi berbeda dari vertex model
        for d in range(1, 10):
            mask = (degrees == d)
            if np.any(mask):
                prob_sum = np.sum(prob[mask])
                prob_by_degree[d] = prob_sum
        
        print("  Distribusi Probabilitas berdasarkan Koordinasi (z):")
        total_low_coord = 0
        for z, p in sorted(prob_by_degree.items()):
            print(f"    z={z}: {p*100:.2f}%")
            if z <= 3:  # Low coordination sites
                total_low_coord += p
        
        # Kesimpulan
        if total_low_coord > 0.90:
            print("  Kesimpulan: ✅ CONFINED STATE (Dominasi low-coordination)")
        else:
            print("  Kesimpulan: ❌ EXTENDED / BOUNDARY STATE")
        
        return prob_by_degree
    
    def plot_result(self, filename='center_model/imgs/lanczos_center_result.png'):
        """
        Visualisasi wavefunction pada center model
        
        Parameters:
        -----------
        filename : str
            Path untuk menyimpan gambar
        """
        print(f"\n[VISUALISASI] Plotting...")
        
        # Analisis lokalisasi
        self.analyze_localization()
        
        psi = self.final_wavefunction
        prob_density = np.abs(psi)**2
        participation_ratio = 1.0 / np.sum(prob_density**2)
        
        # Get coordinates
        coords = np.array([self.vertices[i] for i in range(self.N)])
        
        # Rotasi -18 derajat (konsisten dengan visualisasi lain)
        theta = -18.0
        theta_rad = np.deg2rad(theta)
        cos_theta = np.cos(theta_rad)
        sin_theta = np.sin(theta_rad)
        rotation_matrix = np.array([[cos_theta, -sin_theta], 
                                    [sin_theta, cos_theta]])
        coords_rotated = coords @ rotation_matrix.T
        x_coords = coords_rotated[:, 0]
        y_coords = coords_rotated[:, 1]
        
        # Compute sublattices
        sublattice_map = self.compute_bipartite_sublattices()
        
        # Create figure
        plt.figure(figsize=(10, 10), facecolor='white')
        ax = plt.gca()
        
        # Plot edges (jika tidak terlalu banyak)
        if self.N <= 10000:
            from matplotlib.collections import LineCollection
            lines = []
            for i, j in self.raw_edges:
                lines.append([coords_rotated[i], coords_rotated[j]])
            lc = LineCollection(lines, colors='gray', linewidths=0.3, alpha=0.3, zorder=1)
            ax.add_collection(lc)
        
        # Size and color thresholds
        size_threshold = 1e-16
        color_threshold = 1e-7
        
        # Compute sizes
        sizes = np.where(prob_density < size_threshold, 0.01,
                        np.where(prob_density >= color_threshold, 20, 1.0))
        
        # Plot all centers with probability density
        scatter = ax.scatter(x_coords, y_coords, s=sizes, c=prob_density, 
                            cmap='hot', alpha=0.8, edgecolors='black', 
                            linewidth=0.3, zorder=2, vmin=0, vmax=np.max(prob_density))
        
        # Highlight high probability sites with sublattice colors
        high_prob_mask = prob_density >= color_threshold
        if np.any(high_prob_mask):
            indices_high = np.where(high_prob_mask)[0]
            indices_A = [i for i in indices_high if sublattice_map[i] == 'A']
            indices_B = [i for i in indices_high if sublattice_map[i] == 'B']
            
            if indices_A:
                ax.scatter(x_coords[indices_A], y_coords[indices_A], 
                          s=sizes[indices_A], c='red', 
                          alpha=1.0, edgecolors='black', linewidth=0.5, zorder=3, 
                          label='Sublattice A')
            if indices_B:
                ax.scatter(x_coords[indices_B], y_coords[indices_B], 
                          s=sizes[indices_B], c='blue', 
                          alpha=1.0, edgecolors='black', linewidth=0.5, zorder=3, 
                          label='Sublattice B')
        
        # Add legend if sublattices shown
        if np.any(high_prob_mask):
            ax.legend(loc='upper right', fontsize=8)
        
        # Title and labels
        ax.set_title(rf"Center Model Wavefunction E $\approx$ {self.final_energy:.2f}" + "\n" +
                    f"PR={participation_ratio:.1f}/{self.N} (N={self.N})", 
                    fontsize=12, fontweight='bold')
        ax.set_aspect('equal')
        ax.axis('off')
        
        plt.tight_layout()
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"    ✓ Gambar disimpan di {filename}")
        plt.close()


def print_separator(char="=", length=80):
    """Print separator line"""
    print(char * length)


def main():
    """Main program"""
    print_separator()
    print("   PENROSE CENTER MODEL - LANCZOS TIGHT BINDING")
    print_separator()
    
    # Initialize solver
    solver = CenterModelLanczos(t=1.0)
    
    # Load center model data
    solver.load_data('center_model/data/center_model_penrose_lattice.npz')
    
    # Build Hamiltonian
    solver.build_hamiltonian()
    
    # Run Lanczos algorithm
    solver.run_lanczos_shift_invert(k_steps=50, sigma=1e-9)
    
    # Diagonalize and reconstruct
    solver.diagonalize_T_and_reconstruct(sigma=1e-9)
    
    # Plot result
    solver.plot_result()
    
    print("\n✅ Selesai.")
    print_separator()


if __name__ == "__main__":
    main()
