# Plot Center Tiles - Visualisasi Center Model

## Deskripsi

Script `plot_center_tiles.py` memvisualisasikan bagaimana **center model** dibentuk dari **vertex model** Penrose lattice. Script ini sangat berguna untuk:
- ✅ Mengecek apakah center model sudah benar
- ✅ Memahami hubungan vertex model vs center model (dual graph)
- ✅ Melihat struktur rhombi dan pusat ubin

## Cara Penggunaan

### Basic Usage

```bash
# Visualisasi menggunakan data terbaru
python center_model/plot_center_tiles.py

# Visualisasi iteration tertentu
python center_model/plot_center_tiles.py --iteration 4
python center_model/plot_center_tiles.py --iteration 6
```

### Advanced Options

```bash
# Hanya tampilkan center model (tanpa vertex model background)
python center_model/plot_center_tiles.py --no-vertex

# Hanya tampilkan rhombi dan centers (tanpa dual edges)
python center_model/plot_center_tiles.py --no-dual-edges

# Hanya tampilkan struktur dasar (vertex + rhombi)
python center_model/plot_center_tiles.py --no-centers --no-dual-edges

# Custom input/output
python center_model/plot_center_tiles.py \
  --input vertex_model/data/penrose_lattice_iter5.npz \
  --output center_model/imgs/my_custom_plot.png
```

## Output

Script menghasilkan plot dengan elemen-elemen:

| Elemen | Warna | Marker | Deskripsi |
|--------|-------|--------|-----------|
| **Vertex sites** | Merah | Titik kecil | Atom/node asli dari vertex model |
| **Vertex edges** | Hitam tipis | Garis | Bonds asli dari vertex model |
| **Rhombi outlines** | Hijau | Garis tebal | Outline ubin/tiles (4-cycles) |
| **Center sites** | Biru | Kotak | Pusat rhombi (vertex di center model) |
| **Dual edges** | Biru putus-putus | Garis | Koneksi antar center (jika rhombi berbagi sisi) |

## Contoh Output

File output: `center_model/imgs/center_tiles_visualization.png`

Visualisasi menunjukkan:
1. **Vertex Model** (merah + hitam) - struktur asli
2. **Rhombi** (hijau) - tiles yang membentuk Penrose lattice
3. **Center Model** (biru) - dual graph dengan:
   - Center sites = pusat setiap rhombus
   - Dual edges = koneksi jika dua rhombus berbagi sisi

## Command Line Arguments

```
--input PATH          Input file vertex model (default: vertex_model/data/penrose_lattice_data.npz)
--output PATH         Output image file (default: center_model/imgs/center_tiles_visualization.png)
--iteration N         Use specific iteration (e.g., 4, 5, 6)
--no-vertex           Hide vertex model background
--no-rhombi           Hide rhombi outlines
--no-centers          Hide center sites
--no-dual-edges       Hide dual edges
```

## Interpretasi

### Center Model yang Benar

Jika center model sudah benar, Anda akan melihat:
- ✅ Setiap rhombus (hijau) memiliki **satu center site** (kotak biru) di tengahnya
- ✅ Dual edges (biru putus-putus) menghubungkan center yang rhombi-nya **berbagi satu sisi**
- ✅ Jumlah center = jumlah rhombi dalam vertex model
- ✅ Struktur dual edges membentuk graph yang konsisten

### Verifikasi

Cek di output terminal:
```
Summary:
  Vertex model:  N = 596 vertices
  Center model:  N = 1000 centers, E = 1790 dual edges
```

Rasio yang wajar:
- N_centers / N_vertices ≈ 1.5 - 2.0 (tergantung iteration)
- Setiap center rata-rata punya ~3-4 koneksi dual

## Tips

- Untuk dataset kecil (iter 2-4): gunakan plot default untuk melihat detail
- Untuk dataset besar (iter 6-7): gunakan `--no-vertex` agar lebih jelas
- Zoom in di image viewer untuk melihat struktur detail

## Troubleshooting

### "File not found"
Pastikan sudah generate vertex model:
```bash
python vertex_model/penrose_tiling_fast.py --iteration 4
```

### Plot terlalu ramai
Gunakan iteration lebih kecil atau hide elemen:
```bash
python center_model/plot_center_tiles.py --iteration 3 --no-vertex
```

### Ingin lihat hanya dual graph
```bash
python center_model/plot_center_tiles.py --no-vertex --no-rhombi
```

## Referensi

- Lihat juga: `center_model.py` - script visualisasi alternatif
- Dokumentasi: `README_CENTER_TB.md` - tight binding analysis
