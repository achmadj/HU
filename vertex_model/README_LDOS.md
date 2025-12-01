# Local Density of States (LDOS) Implementation

## Overview
LDOS analysis untuk Penrose lattice telah ditambahkan pada `vertex_model/penrose_tight_binding.py` sesuai persamaan (42) dari paper.

## Persamaan LDOS

Sesuai persamaan (42):

$$\rho_z(\omega) = \frac{1}{L} \sum_{l=1}^{L} \frac{\sum_{l:(z_l=z)} |u_{lk}|^2}{\sum_{l=1}^{L} |u_{lk}|^2} \delta(\hbar\omega - \varepsilon_k)$$

Di mana:
- $\rho_z(\omega)$: Local density of states untuk coordination number $z$
- $z_l$: Coordination number dari site $l$
- $u_{lk}$: Komponen eigenvector untuk site $l$ pada state $k$
- $\varepsilon_k$: Energy eigenvalue untuk state $k$
- $\delta(\hbar\omega - \varepsilon_k)$: Delta function (diimplementasikan dengan Gaussian smoothing)

Total DOS:
$$\rho(\omega) = \sum_z \rho_z(\omega)$$

## Method Baru

### 1. `get_coordination_numbers()`
Menghitung coordination number (jumlah tetangga) untuk setiap site.

**Output**: `Dict[int, int]` - Peta site_id → coordination number (z)

### 2. `calculate_ldos(bins=100, sigma=0.05)`
Menghitung LDOS untuk setiap coordination number.

**Parameters**:
- `bins` (int): Jumlah bins untuk histogram energi
- `sigma` (float): Lebar Gaussian untuk smoothing delta function

**Output**: `Tuple[NDArray, Dict[int, NDArray]]`
- `energies`: Array energi
- `ldos_dict`: Dictionary z → ρ_z(ω)

**Implementation Details**:
1. Loop over semua eigenstates k
2. Untuk setiap z, hitung kontribusi: $\frac{\sum_{l:z_l=z} |u_{lk}|^2}{\sum_l |u_{lk}|^2}$
3. Tambahkan Gaussian-smoothed delta function di $\varepsilon_k$
4. Normalisasi dengan jumlah site dan bin width

### 3. `plot_ldos(bins=200, sigma=0.05, save_fig=True, filename='...')`
Plot LDOS untuk setiap coordination number.

**Features**:
- Plot stacked LDOS untuk setiap z dengan warna berbeda
- Plot total DOS sebagai sum dari semua LDOS
- Statistik: jumlah site, persentase, dan peak LDOS untuk setiap z

## Usage

### Basic Usage
```python
from vertex_model.penrose_tight_binding import PenroseTightBinding

# Load dan diagonalisasi
tb_model = PenroseTightBinding(epsilon_0=0.0, t=1.0)
tb_model.load_from_numpy('vertex_model/data/penrose_lattice_iter4.npz')
tb_model.build_hamiltonian()
tb_model.diagonalize()

# Calculate LDOS
energies, ldos_dict = tb_model.calculate_ldos(bins=200, sigma=0.05)

# Plot LDOS
tb_model.plot_ldos(bins=200, sigma=0.05)
```

### Test Script
Gunakan `test_ldos_vertex.py` untuk test dengan iteration kecil:

```bash
python test_ldos_vertex.py
```

## Output Files

- `vertex_model/imgs/penrose_ldos.png`: Plot LDOS (dari main script)
- `vertex_model/imgs/penrose_ldos_iter4.png`: Plot LDOS (dari test script)

## Interpretasi

### LDOS di E=0
- Penrose lattice memiliki macroscopically degenerate confined states di E=0
- States ini terlokalisasi di sites dengan coordination number z=3 dan z=5
- LDOS plot menunjukkan kontribusi relatif dari setiap coordination number
- Peak di ρ_z(ω=0) menunjukkan lokalisasi preferensial

### Expected Results
Berdasarkan paper:
- ~10% dari total states berada di E=0
- States ini strictly localized di z=3 dan z=5 sites
- Separated dari remainder states dengan gap ~0.172871

## Technical Notes

### Gaussian Smoothing
Delta function δ(ω - ε_k) digantikan dengan Gaussian:

$$\delta(\omega - \varepsilon_k) \approx \frac{1}{\sqrt{2\pi\sigma^2}} \exp\left(-\frac{(\omega - \varepsilon_k)^2}{2\sigma^2}\right)$$

Default σ=0.05 memberikan smoothing yang cukup tanpa blur berlebihan.

### Normalization
LDOS dinormalisasi sehingga:

$$\sum_z \int \rho_z(\omega) d\omega = 1$$

Setiap ρ_z weighted dengan fraksi site: $(n_z / N)$ di mana $n_z$ adalah jumlah site dengan coordination z.

## Integration dengan Main Script

LDOS plot ditambahkan ke main workflow:
1. Energy spectrum
2. DOS (histogram)
3. **LDOS (by coordination number)** ← NEW
4. Integrated DOS
5. Zero-energy wavefunction plots

## References

Persamaan implementasi mengikuti:
- Equation (42) dari attached paper
- Macroscopically degenerate confined states discussion (Section about Penrose lattice)
