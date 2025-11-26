"""
Penrose Lattice - Manual Lanczos (Sparse Matrix + Linear Solver)
Ditambah Analisis Lokalisasi (z-coordination distribution)
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix, eye
from scipy.sparse.linalg import spsolve
from collections import deque
import time

class PenroseLanczosDense:
    """
    Implementasi Lanczos dengan Sparse Matrix.
    Menggunakan scipy.sparse.linalg.spsolve untuk Shift-Invert.
    """
    def __init__(self, t: float = 1.0):
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

    def load_data(self, filename: str):
        """Load data geometri"""
        print(f"[1] Membaca data dari {filename}...")
        try:
            data = np.load(filename)
            self.N = int(data['N'])
            vertex_coords = data['vertex_coords']
            self.vertices = {i: coord for i, coord in enumerate(vertex_coords)}
            self.raw_edges = data['edge_list']
            print(f"    ✓ N = {self.N} atoms")
        except FileNotFoundError:
            print("    ❌ File tidak ditemukan.")
            exit()

    def build_hamiltonian(self):
        """
        Membangun Hamiltonian sebagai SPARSE Matrix (CSR format).
        Jauh lebih efisien untuk matriks besar dengan banyak elemen nol.
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

    def run_lanczos_shift_invert(self, k_steps=50, sigma=1e-9):
        """
        Algoritma Lanczos dengan LINEAR SOLVER (spsolve).
        Menyelesaikan (H - σI)w' = v tanpa menghitung invers eksplisit.
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
            # --- PERUBAHAN UTAMA ---
            # Solve (H - σI)w' = v instead of computing w' = (H - σI)^-1 * v
            # This is O(N^1.5) for sparse matrices vs O(N^3) for dense inverse
            w = spsolve(M, v_curr)
            
            # Sisanya sama dengan Lanczos biasa
            alpha = np.dot(v_curr, w)
            alphas.append(alpha)
            
            w = w - alpha * v_curr - beta_prev * v_prev
            beta = np.linalg.norm(w)
            
            if beta < 1e-10: 
                print("    ! Konvergensi dini.")
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

    def diagonalize_T_and_reconstruct(self, sigma=1e-9):
        """Diagonalisasi T menggunakan np.linalg.eigh (Standar)"""
        print(f"\n[10] Rekonstruksi Solusi...")
        
        # Pakai eigh biasa sesuai request
        evals_T, evecs_T = np.linalg.eigh(self.T_matrix)
        
        # Pilih eigenvalue magnitude terbesar (karena shift-invert)
        idx_max = np.argmax(np.abs(evals_T))
        lambda_best = evals_T[idx_max]
        
        self.final_energy = sigma + (1.0 / lambda_best)
        print(f"    Energi (E): {self.final_energy:.10f}")
        
        # Rekonstruksi Wavefunction
        y_vec = evecs_T[:, idx_max]
        self.final_wavefunction = np.zeros(self.N)
        
        k = len(y_vec)
        for i in range(k):
            self.final_wavefunction += y_vec[i] * self.krylov_basis[i]
        self.final_wavefunction /= np.linalg.norm(self.final_wavefunction)

    def compute_bipartite_sublattices(self):
        """Helper BFS"""
        adj = {i:[] for i in range(self.N)}
        for i, j in self.raw_edges:
            adj[i].append(j); adj[j].append(i)
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
        print(f"Sublattice A: {list(sublattice_map.values()).count('A')}, Sublattice B: {list(sublattice_map.values()).count('B')}")
        return sublattice_map

    def analyze_localization(self):
        """
        Analisis Lokalisasi Wavefunction berdasarkan Koordinasi (z).
        Menghitung persentase probabilitas |psi|^2 pada situs dengan z=2,3,4,5,6,7.
        """
        print("\n[ANALISIS LOKALISASI]")
        
        # 1. Hitung Degree (Jumlah Tetangga) untuk setiap atom
        # Karena kita pakai Dense Matrix, kita hitung manual non-zero di setiap baris
        # (Tapi lebih cepat pakai raw_edges karena dense matrix penuh float)
        degrees = np.zeros(self.N, dtype=int)
        for i, j in self.raw_edges:
            degrees[i] += 1
            degrees[j] += 1
            
        # 2. Hitung Distribusi Probabilitas
        prob = np.abs(self.final_wavefunction)**2
        prob_by_degree = {}
        
        # Penrose biasanya z=3 sampai 7, tapi kita cek 1-9 untuk aman
        for d in range(1, 10):
            mask = (degrees == d)
            if np.any(mask):
                prob_sum = np.sum(prob[mask])
                prob_by_degree[d] = prob_sum
        
        print("  Distribusi Probabilitas berdasarkan Koordinasi (z):")
        total_confined = 0
        for z, p in prob_by_degree.items():
            print(f"    z={z}: {p*100:.2f}%")
            if z in [3, 5]: # Kriteria Confined State (Inoue/Arai)
                total_confined += p
        
        # Kesimpulan Sederhana
        if total_confined > 0.90:
            print("  Kesimpulan: ✅ CONFINED STATE (Dominasi z=3 & z=5)")
        else:
            print("  Kesimpulan: ❌ EXTENDED / BOUNDARY STATE")

    def plot_result(self, filename='vertex_model/imgs/lanczos_result.png'):
        """Visualisasi Identik"""
        
        # Panggil Analisis dulu sebelum plotting
        self.analyze_localization()
        
        print(f"\n[VISUALISASI] Plotting...")
        
        psi = self.final_wavefunction
        prob_density = np.abs(psi)**2
        participation_ratio = 1.0 / np.sum(prob_density**2)
        
        coords = np.array([self.vertices[i] for i in range(self.N)])
        theta = -18.0
        theta_rad = np.deg2rad(theta)
        cos_theta = np.cos(theta_rad)
        sin_theta = np.sin(theta_rad)
        rotation_matrix = np.array([[cos_theta, -sin_theta], [sin_theta, cos_theta]])
        coords_rotated = coords @ rotation_matrix.T
        x_coords = coords_rotated[:, 0]
        y_coords = coords_rotated[:, 1]

        sublattice_map = self.compute_bipartite_sublattices()
        
        plt.figure(figsize=(10, 10), facecolor='white')
        ax = plt.gca()
        
        if self.N <= 10000:
            from matplotlib.collections import LineCollection
            lines = []
            for i, j in self.raw_edges:
                lines.append([coords_rotated[i], coords_rotated[j]])
            lc = LineCollection(lines, colors='gray', linewidths=0.3, alpha=0.3, zorder=1)
            ax.add_collection(lc)

        size_threshold = 1e-16
        color_threshold = 1e-7
        
        sizes = np.where(prob_density < size_threshold, 0.01,
                        np.where(prob_density >= color_threshold, 20, 1.0))
        
        ax.scatter(x_coords, y_coords, s=sizes, c=prob_density, 
                   cmap='hot', alpha=0.8, edgecolors='black', 
                   linewidth=0.3, zorder=2, vmin=0, vmax=np.max(prob_density))
        
        high_prob_mask = prob_density >= color_threshold
        if np.any(high_prob_mask):
            indices_high = np.where(high_prob_mask)[0]
            indices_A = [i for i in indices_high if sublattice_map[i] == 'A']
            indices_B = [i for i in indices_high if sublattice_map[i] == 'B']
            
            if indices_A:
                ax.scatter(x_coords[indices_A], y_coords[indices_A], 
                          s=sizes[indices_A], c='red', 
                          alpha=1.0, edgecolors='black', linewidth=0.5, zorder=3, label='Sublattice A')
            if indices_B:
                ax.scatter(x_coords[indices_B], y_coords[indices_B], 
                          s=sizes[indices_B], c='blue', 
                          alpha=1.0, edgecolors='black', linewidth=0.5, zorder=3, label='Sublattice B')

        ax.set_title(rf"Wavefunction E $\approx$ {self.final_energy:.2f}" + "\n" +
                    f"PR={participation_ratio:.1f}/{self.N} ", 
                    fontsize=12, fontweight='bold')
        ax.set_aspect('equal')
        ax.axis('off')
        
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"    ✓ Gambar disimpan di {filename}")

def main():
    print("="*60)
    print("   SPARSE MATRIX LANCZOS (LINEAR SOLVER)")
    print("="*60)
    
    solver = PenroseLanczosDense()
    solver.load_data('vertex_model/data/penrose_lattice_data.npz')
    solver.build_hamiltonian()
    
    # Jalankan solver dengan lebih banyak iterasi untuk konvergensi lebih baik
    solver.run_lanczos_shift_invert(k_steps=3, sigma=1e-9)
    
    solver.diagonalize_T_and_reconstruct()
    
    # Plot hasil (akan otomatis memanggil analyze_localization)
    solver.plot_result()
    print("\n✅ Selesai.")

if __name__ == "__main__":
    main()