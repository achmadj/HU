import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

def visualize_vertex_vs_center(filename='vertex_model/data/penrose_lattice_data.npz'):
    print("Memuat data kisi...")
    try:
        data = np.load(filename)
        vertex_coords = data['vertex_coords']
        edge_list = data['edge_list']
        N = len(vertex_coords)
        print(f"  N = {N} vertices")
    except FileNotFoundError:
        print("File tidak ditemukan! Pastikan path benar.")
        return

    # 1. Bangun Adjacency List (Vertex Graph)
    adj = defaultdict(set)
    edges_set = set()
    for u, v in edge_list:
        adj[u].add(v)
        adj[v].add(u)
        edges_set.add(tuple(sorted((u, v))))

    # 2. Deteksi Wajah (Faces/Rhombi) untuk Center Model
    # Algoritma: Cari siklus 4-atom (u-v-x-w-u)
    faces = []
    processed_quads = set()

    print("Mendeteksi ubin (rhombi)...")
    for u in range(N):
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
                    x = common.pop() # Ambil satu tetangga bersama (harusnya cuma 1 utk rhombus)
                    
                    # Kita menemukan rhombus u-v-x-w
                    quad = tuple(sorted((u, v, w, x)))
                    if quad not in processed_quads:
                        faces.append(quad)
                        processed_quads.add(quad)

    print(f"  Ditemukan {len(faces)} ubin (rhombi).")

    # 3. Hitung Pusat Ubin (Center Sites)
    centers = []
    face_map = {} # Mapping dari index face ke koordinat center
    
    # Mapping dari Edge ke Face (untuk membangun Dual Graph)
    edge_to_faces = defaultdict(list)

    for idx, quad in enumerate(faces):
        # Koordinat rata-rata dari 4 sudut
        coords = vertex_coords[list(quad)]
        center = np.mean(coords, axis=0)
        centers.append(center)
        face_map[idx] = center
        
        # Daftarkan setiap sisi ubin ini
        # Sisi: (u,v), (v,x), (x,w), (w,u) -> harus urut
        # Cara cepat: Ambil semua kombinasi 2 dari 4 yang ada di edges_set
        for i in range(4):
            for j in range(i+1, 4):
                edge = tuple(sorted((quad[i], quad[j])))
                if edge in edges_set:
                    edge_to_faces[edge].append(idx)

    centers = np.array(centers)

    # 4. Bangun Dual Edges (Koneksi Center Model)
    # Jika dua ubin berbagi satu sisi (edge), maka hubungkan pusatnya
    dual_edges = []
    for edge, face_indices in edge_to_faces.items():
        if len(face_indices) == 2: # Edge dishare oleh 2 ubin
            f1, f2 = face_indices
            dual_edges.append((f1, f2))

    # --- ROTASI 18 DERAJAT SEARAH JARUM JAM ---
    theta = -18.0  # Negatif untuk searah jarum jam
    theta_rad = np.deg2rad(theta)
    cos_theta = np.cos(theta_rad)
    sin_theta = np.sin(theta_rad)
    rotation_matrix = np.array([[cos_theta, -sin_theta], 
                                [sin_theta, cos_theta]])
    
    # Rotasi vertex coordinates
    vertex_coords_rotated = vertex_coords @ rotation_matrix.T
    
    # Rotasi center coordinates
    centers_rotated = centers @ rotation_matrix.T
    
    # Update face_map dengan koordinat yang sudah dirotasi
    face_map_rotated = {}
    for idx, center in enumerate(centers_rotated):
        face_map_rotated[idx] = center

    # --- PLOTTING ---
    print("Membuat Plot...")
    plt.figure(figsize=(12, 10), facecolor='white')
    ax = plt.gca()

    # A. Plot Vertex Model (Asli)
    # Edges Asli (Abu-abu)
    for u, v in edge_list:
        p1, p2 = vertex_coords_rotated[u], vertex_coords_rotated[v]
        ax.plot([p1[0], p2[0]], [p1[1], p2[1]], 'k-', lw=0.5, alpha=0.3, zorder=1)
    
    # Vertex Asli (Merah)
    ax.scatter(vertex_coords_rotated[:,0], vertex_coords_rotated[:,1], s=15, c='red', 
               label='Vertex Model (Atoms)', zorder=2)

    # B. Plot Center Model (Dual)
    # Dual Edges (Biru Putus-putus)
    for f1, f2 in dual_edges:
        p1, p2 = face_map_rotated[f1], face_map_rotated[f2]
        ax.plot([p1[0], p2[0]], [p1[1], p2[1]], 'b--', lw=1.5, alpha=0.6, zorder=3)

    # Center Sites (Biru)
    if len(centers_rotated) > 0:
        ax.scatter(centers_rotated[:,0], centers_rotated[:,1], s=30, c='blue', marker='s', 
                   label='Center Model (Atoms)', zorder=4)

    # Formatting
    plt.title("Perbandingan Visual: Vertex Model vs Center Model (Dual)", fontsize=14)
    plt.legend(loc='upper right')
    plt.axis('equal')
    plt.axis('off')
    
    plt.tight_layout()
    plt.savefig('center_model/imgs/vertex_vs_center_model.png', dpi=300)
    print("✅ Gambar disimpan di 'imgs/vertex_vs_center_model.png'")
    plt.show()

if __name__ == "__main__":
    # Pastikan file npz ada
    visualize_vertex_vs_center()