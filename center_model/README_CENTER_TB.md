# Center Model Tight Binding - Cara Penggunaan

## Deskripsi

Script `center_penrose_tight_binding.py` melakukan analisis tight binding pada Penrose lattice **center model** (dual graph). Script ini mirip dengan vertex model, namun:
- **Vertices**: Pusat dari setiap rhombus (ubin) dalam Penrose lattice
- **Edges**: Koneksi antar pusat jika dua rhombus berbagi satu sisi

## Visualisasi Center Tiles

Untuk **mengecek apakah center model sudah benar**, gunakan script visualisasi:

```bash
# Visualisasi default (menggunakan data terbaru)
python center_model/plot_center_tiles.py

# Visualisasi iteration tertentu
python center_model/plot_center_tiles.py --iteration 4

# Hanya tampilkan centers dan dual edges (tanpa vertex model)
python center_model/plot_center_tiles.py --no-vertex

# Tampilkan hanya vertex model dan rhombi (tanpa dual graph)
python center_model/plot_center_tiles.py --no-centers --no-dual-edges
```

**Output**: `center_model/imgs/center_tiles_visualization.png`
- **Merah**: Vertex sites (atom asli)
- **Hijau**: Rhombi outlines (tiles/ubin)
- **Biru kotak**: Center sites (pusat rhombi)
- **Biru putus-putus**: Dual edges (koneksi antar center)

## Files yang Dibuat

Script ini menghasilkan empat plot:

1. **Energy Spectrum** (`center_model/imgs/center_energy_spectrum.png`)
   - Plot energi eigenvalue vs state index
   - Menunjukkan distribusi energi semua state

2. **Density of States (DOS)** (`center_model/imgs/center_dos.png`)
   - Histogram normalized dari distribusi energi
   - Menunjukkan berapa banyak state di setiap energi

3. **Integrated DOS (IDOS)** (`center_model/imgs/center_idos.png`)
   - Cumulative distribution function dari energi
   - Menunjukkan fraksi state dengan energi ≤ E

4. **Wavefunction Probability Density** (`center_model/imgs/center_wavefunctions_E2.png`)
   - Visualisasi fungsi gelombang untuk state dengan E ≈ 2
   - Ukuran titik menunjukkan probabilitas
   - Warna merah/biru menunjukkan sublattice A/B untuk probabilitas tinggi

## Cara Menjalankan

### 1. Menggunakan Data yang Sudah Ada (Iteration 7)

```bash
cd /workspaces/HU
python center_model/center_penrose_tight_binding.py
```

**Catatan**: Iteration 7 memiliki ~10,715 vertices. Diagonalisasi matrix 10715×10715 membutuhkan waktu cukup lama dan memory yang besar.

### 2. Menggunakan Iteration Lebih Kecil (Recommended untuk Testing)

Pertama, generate center model dari iteration yang lebih kecil:

```bash
cd /workspaces/HU

# Generate center model dari iteration 4 (N ~ 596 vertices → ~1000 centers)
python center_model/center_penrose_tiling_fast.py --iteration 4
```

Kemudian jalankan tight binding analysis:

```bash
python center_model/center_penrose_tight_binding.py
```

### 3. Menggunakan Test Script (All-in-One)

Test script otomatis generate center model dari iteration 4 dan langsung analisis:

```bash
cd /workspaces/HU
python center_model/test_center_tb.py
```

File output akan disimpan dengan nama `*_iter4.png` di `center_model/imgs/`.

## Perbandingan dengan Vertex Model

| Aspek | Vertex Model | Center Model |
|-------|--------------|--------------|
| **Vertices** | Atom/node asli | Pusat rhombus (dual) |
| **Edges** | Bonds langsung | Koneksi dual (shared edge) |
| **Jumlah N** | ~18,000 (iter 7) | ~10,715 (iter 7) |
| **Jumlah E** | ~33,000 (iter 7) | ~21,140 (iter 7) |
| **Script** | `vertex_model/penrose_tight_binding.py` | `center_model/center_penrose_tight_binding.py` |
| **Output** | `vertex_model/imgs/*.png` | `center_model/imgs/*.png` |

## Output Files

Setelah menjalankan script, file berikut akan dibuat:

```
center_model/imgs/
├── center_energy_spectrum.png      # Energy spectrum plot
├── center_dos.png                  # Density of States
├── center_idos.png                 # Integrated DOS
└── center_wavefunctions_E2.png     # Wavefunction at E≈2
```

Jika menggunakan test script (iteration 4):

```
center_model/imgs/
├── center_energy_spectrum_iter4.png
├── center_dos_iter4.png
├── center_idos_iter4.png
└── center_wavefunctions_E2_iter4.png
```

## Troubleshooting

### Memory Error
Jika mendapat memory error saat diagonalisasi:
- Gunakan iteration lebih kecil (3 atau 4)
- Atau gunakan Lanczos method (file terpisah: `lanczos_center_tight_binding.py`)

### File Not Found
Pastikan sudah generate center model terlebih dahulu:
```bash
python center_model/center_penrose_tiling_fast.py --iteration 4
```

### Import Error
Pastikan menjalankan dari root directory `/workspaces/HU/`:
```bash
cd /workspaces/HU
python center_model/center_penrose_tight_binding.py
```

## Kode yang Sama dengan Vertex Model

Implementasi tight binding untuk center model menggunakan **kode yang persis sama** dengan vertex model:
- Hamiltonian construction
- Diagonalization (numpy.linalg.eigh)
- DOS/IDOS calculation
- Plotting functions

Yang berbeda hanya:
1. Input data (center sites vs vertex sites)
2. Nama file output
3. Label plot ("Center Model" vs "Vertex Model")

## Referensi

Lihat dokumentasi lengkap di:
- `README.md` - Dokumentasi utama project
- `USAGE.md` - Quick usage guide
- `vertex_model/penrose_tight_binding.py` - Reference implementation
