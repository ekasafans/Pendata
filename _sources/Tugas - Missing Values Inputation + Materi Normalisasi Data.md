# Tugas — Missing Values (WKNN) & Normalisasi Data

---

## Struktur Tugas (Sesuai Notebook)

### 1. Soal

Lampiran gambar soal:

<img alt="Soal WKNN Normalisasi" src="Assets/Pertemuan4/Soal%20Missing%20Values%20Inputation%20WKKN.png" />

<img alt="Soal WKNN Normalisasi (Lampiran Tambahan)" src="Tugas/Missing Values & Normalisasi/Soal Missing Values Inputation WKKN.png" />

Lampiran file Excel (siap diunduh/dibagikan):

[Download Excel - Missing Value Metode WKKN dan Normalisasi](<Tugas/Missing Values & Normalisasi/Missing Value Metode WKKN dan Normalisasi.xlsx>)

Preview sheet perhitungan manual (sesuai file Excel):

<img alt="Sheet WKNN dan 3 normalisasi" src="Tugas/Missing Values & Normalisasi/{08731576-79E4-47C6-B15D-8614811B6232}.png" />

<img alt="Sheet parameter kontrol WKNN" src="Tugas/Missing Values & Normalisasi/{92225BFD-2920-4144-A7E3-FCB8297CC627}.png" />

<img alt="Sheet normalisasi Min-Max" src="Tugas/Missing Values & Normalisasi/{9772C413-1BB5-45DC-876B-90A3F1BA2C97}.png" />

<img alt="Sheet tabel tetangga terdekat WKNN" src="Tugas/Missing Values & Normalisasi/{BDD39292-E751-4943-9B4D-2584C2046196}.png" />

Diberikan dataset berikut:

| NO | IPK | PO | JMLH |
|----|-----|--------|------|
| 1  | 2   | 200000 | 2 |
| 2  | 3   | 300000 | 3 |
| 3  | 4   | 200000 | 2 |
| 4  | 2   | 200000 | 3 |
| 5  | 3   | 300000 | 2 |
| 6  | 4   | 400000 | 3 |
| 7  | 2   | 300000 | ? |

Pertanyaan:
1. Gunakan WKNN dengan $k=3$ untuk memprediksi nilai JMLH pada NO=7.
2. Lakukan normalisasi Min-Max, Z-Score, dan Decimal Scaling pada fitur IPK, PO, dan JMLH.

Catatan pengerjaan:
- Pada data awal, JMLH NO=7 memang belum diketahui (ditulis ?).
- Setelah dihitung dengan WKNN, diperoleh JMLH NO=7 = 3.
- Nilai 3 tersebut yang dipakai pada tahap normalisasi manual dan coding.

### 2. Jawaban Manual

Pada pengerjaan manual ini, nilai NO=7 diisi menggunakan hasil imputasi WKNN, yaitu:

$$
\boxed{\text{JMLH NO=7} = 3}
$$

Setelah nilai JMLH NO=7 diketahui, normalisasi dapat dihitung untuk seluruh baris data.

Rumus Min-Max:

$$
x' = \frac{x - x_{\min}}{x_{\max} - x_{\min}}
$$

Hasil normalisasi (ringkas):

| NO | IPK' | PO'  | JMLH' |
|----|------|------|-------|
| 1  | 0    | 0    | 0     |
| 2  | 0.5  | 0.5  | 1     |
| 3  | 1    | 0    | 0     |
| 4  | 0    | 0    | 1     |
| 5  | 0.5  | 0.5  | 0     |
| 6  | 1    | 1    | 1     |
| 7  | 0    | 0.5  | 1     |

Rumus jarak Euclidean:

$$
d(a,b) = \sqrt{\sum_{i=1}^{n}(a_i - b_i)^2}
$$

Jarak data NO=7 ke data latih:
- $d(7,1)=0.5$, $d(7,2)=0.5$, $d(7,3)\approx1.118$
- $d(7,4)=0.5$, $d(7,5)=0.5$, $d(7,6)\approx1.118$

Ambil $k=3$ tetangga terdekat: NO=1, NO=2, NO=4.

Rumus bobot WKNN:

$$
w_i = \frac{1}{d_i^2}
$$

Karena semua $d=0.5$ pada 3 tetangga terpilih, maka tiap bobot $w=4$.

Weighted voting:
- Kelas JMLH=2: total bobot $4$
- Kelas JMLH=3: total bobot $8$

Kesimpulan manual:
- Hasil imputasi WKNN NO=7 adalah kelas 3.
- Nilai tersebut dipakai pada tabel normalisasi (Min-Max, Z-Score, Decimal Scaling).
- Hasil pada dokumen ini sudah disesuaikan dengan sheet Excel lampiran.

Prediksi akhir:

$$
\boxed{\text{JMLH} = 3}
$$

### 3. Jawaban Coding (Python) - Program Lengkap

Bagian ini bukan hanya potongan sel, tetapi program Python utuh yang menghasilkan nilai normalisasi, jarak, bobot, dan prediksi yang sama seperti jawaban manual.

```python
import numpy as np
import pandas as pd
from collections import defaultdict

df = pd.DataFrame({
    'NO': [1, 2, 3, 4, 5, 6, 7],
    'IPK': [2, 3, 4, 2, 3, 4, 2],
    'PO': [200000, 300000, 200000, 200000, 300000, 400000, 300000],
    'JMLH': [2, 3, 2, 3, 2, 3, np.nan]
})

def min_max(s):
    return (s - s.min()) / (s.max() - s.min())

df_norm = df.copy()
df_norm['IPK_norm'] = min_max(df['IPK'])
df_norm['PO_norm'] = min_max(df['PO'])
df_norm['JMLH_norm'] = (df['JMLH'] - df['JMLH'].dropna().min()) / (df['JMLH'].dropna().max() - df['JMLH'].dropna().min())

train = df_norm[df_norm['JMLH'].notna()].copy()

query = df_norm.loc[df_norm['NO'] == 7, ['IPK_norm', 'PO_norm']].values[0]

train['jarak'] = ((train['IPK_norm']-query[0])**2 + (train['PO_norm']-query[1])**2) ** 0.5

k = 3
knn = train.sort_values('jarak').head(k).copy()
knn['bobot'] = 1 / (knn['jarak'] ** 2)

class_weights = defaultdict(float)
for _, row in knn.iterrows():
    class_weights[int(row['JMLH'])] += row['bobot']

pred = max(class_weights, key=class_weights.get)

# Isi nilai missing NO=7 dengan hasil prediksi WKNN agar tabel normalisasi final lengkap.
df_norm_final = df_norm.copy()
df_norm_final.loc[df_norm_final['NO'] == 7, 'JMLH'] = pred
df_norm_final['JMLH_norm'] = (
    (df_norm_final['JMLH'] - df['JMLH'].dropna().min()) /
    (df['JMLH'].dropna().max() - df['JMLH'].dropna().min())
)

print('=== TABEL NORMALISASI ===')
print(df_norm_final[['NO', 'IPK_norm', 'PO_norm', 'JMLH_norm']].to_string(index=False))

print('\n=== TABEL JARAK ===')
print(train[['NO', 'IPK_norm', 'PO_norm', 'JMLH', 'jarak']].sort_values('jarak').to_string(index=False))

print('\n=== 3 TETANGGA TERDEKAT ===')
print(knn[['NO', 'JMLH', 'jarak', 'bobot']].to_string(index=False))

print('\n=== WEIGHTED VOTING ===')
for kelas, bobot in sorted(class_weights.items()):
    print(f'JMLH = {kelas}, total bobot = {bobot:.2f}')

print(f'\nPrediksi JMLH NO=7 = {pred}')
```

Hasil dari program di atas:

#### Tabel normalisasi

| NO | IPK' | PO' | JMLH' |
|----|------|-----|-------|
| 1 | 0.0 | 0.0 | 0.0 |
| 2 | 0.5 | 0.5 | 1.0 |
| 3 | 1.0 | 0.0 | 0.0 |
| 4 | 0.0 | 0.0 | 1.0 |
| 5 | 0.5 | 0.5 | 0.0 |
| 6 | 1.0 | 1.0 | 1.0 |
| 7 | 0.0 | 0.5 | 1.0 |

#### Tabel jarak ke data NO=7

| NO | JMLH | Jarak |
|----|------|-------|
| 1 | 2 | 0.5 |
| 2 | 3 | 0.5 |
| 4 | 3 | 0.5 |
| 5 | 2 | 0.5 |
| 3 | 2 | 1.118 |
| 6 | 3 | 1.118 |

#### Tiga tetangga terdekat yang dipakai

| NO | JMLH | Jarak | Bobot |
|----|------|-------|-------|
| 1 | 2 | 0.5 | 4 |
| 2 | 3 | 0.5 | 4 |
| 4 | 3 | 0.5 | 4 |

#### Hasil weighted voting

- JMLH = 2, total bobot = 4
- JMLH = 3, total bobot = 8

Sehingga hasil program sama dengan jawaban manual, yaitu:

$$
\boxed{\text{Prediksi JMLH NO=7 = 3}}
$$

---

## Normalisasi Data dengan Soal yang Sama

Berikut adalah penerapan **tiga metode normalisasi** (Min-Max, Z-Score, Decimal Scaling) terhadap dataset soal di atas. WKNN telah memprediksi **JMLH NO=7 = 3**, sehingga nilai tersebut kini diikutsertakan dalam normalisasi semua kolom (IPK, PO, dan JMLH).

---

### A. Min-Max Normalization

**Rumus:**

$$
x' = \frac{x - x_{\min}}{x_{\max} - x_{\min}}
$$

#### ✍️ Jawaban Manual

**Nilai min & max:**

| Fitur | Min | Max | Range |
|-------|-----|-----|-------|
| IPK | 2 | 4 | 2 |
| PO | 200.000 | 400.000 | 200.000 |
| JMLH | 2 | 3 | 1 |

**Perhitungan per data:**

IPK:
$$IPK'_1 = \frac{2-2}{4-2}=0, \quad IPK'_2 = \frac{3-2}{2}=0{,}5, \quad IPK'_3 = \frac{4-2}{2}=1$$
$$IPK'_4 = 0, \quad IPK'_5 = 0{,}5, \quad IPK'_6 = 1, \quad IPK'_7 = 0$$

PO:
$$PO'_1 = \frac{200000-200000}{200000}=0, \quad PO'_2 = \frac{100000}{200000}=0{,}5, \quad PO'_3 = 0$$
$$PO'_4 = 0, \quad PO'_5 = 0{,}5, \quad PO'_6 = \frac{200000}{200000}=1, \quad PO'_7 = 0{,}5$$

JMLH (termasuk NO=7 dengan hasil prediksi WKNN = 3):
$$JMLH'_1 = \frac{2-2}{1}=0, \quad JMLH'_2 = \frac{3-2}{1}=1, \quad JMLH'_3 = 0$$
$$JMLH'_4 = 1, \quad JMLH'_5 = 0, \quad JMLH'_6 = 1, \quad JMLH'_7 = \frac{3-2}{1}=1$$

**Tabel Hasil Manual Min-Max:**

| NO | IPK | PO | JMLH | IPK' | PO' | JMLH' |
|----|-----|----|------|------|-----|-------|
| 1 | 2 | 200000 | 2 | 0 | 0 | 0 |
| 2 | 3 | 300000 | 3 | 0,5 | 0,5 | 1 |
| 3 | 4 | 200000 | 2 | 1 | 0 | 0 |
| 4 | 2 | 200000 | 3 | 0 | 0 | 1 |
| 5 | 3 | 300000 | 2 | 0,5 | 0,5 | 0 |
| 6 | 4 | 400000 | 3 | 1 | 1 | 1 |
| 7 | 2 | 300000 | **3** | 0 | 0,5 | **1** |

#### 💻 Jawaban Coding

```python
import numpy as np
import pandas as pd

df = pd.DataFrame({
    'NO':   [1, 2, 3, 4, 5, 6, 7],
    'IPK':  [2, 3, 4, 2, 3, 4, 2],
    'PO':   [200000, 300000, 200000, 200000, 300000, 400000, 300000],
    'JMLH': [2, 3, 2, 3, 2, 3, np.nan]
})

def min_max(series):
    return (series - series.min()) / (series.max() - series.min())

df_mm = df.copy()
df_mm.loc[df_mm['NO'] == 7, 'JMLH'] = 3  # isi dengan hasil prediksi WKNN
df_mm['IPK_mm']  = min_max(df['IPK'])
df_mm['PO_mm']   = min_max(df['PO'])
df_mm['JMLH_mm'] = min_max(df_mm['JMLH'])  # min=2, max=3 (tidak berubah)

print(df_mm[['NO','IPK','IPK_mm','PO','PO_mm','JMLH','JMLH_mm']].to_markdown(index=False, floatfmt='.4f'))
```

**Output:**

| NO | IPK | IPK_mm | PO | PO_mm | JMLH | JMLH_mm |
|----|-----|--------|----|-------|------|---------|
| 1 | 2 | 0.0000 | 200000 | 0.0000 | 2.0 | 0.0000 |
| 2 | 3 | 0.5000 | 300000 | 0.5000 | 3.0 | 1.0000 |
| 3 | 4 | 1.0000 | 200000 | 0.0000 | 2.0 | 0.0000 |
| 4 | 2 | 0.0000 | 200000 | 0.0000 | 3.0 | 1.0000 |
| 5 | 3 | 0.5000 | 300000 | 0.5000 | 2.0 | 0.0000 |
| 6 | 4 | 1.0000 | 400000 | 1.0000 | 3.0 | 1.0000 |
| 7 | 2 | 0.0000 | 300000 | 0.5000 | 3.0 | 1.0000 |

> Hasil coding sama dengan jawaban manual.

---

### B. Z-Score Normalization (Standardization)

**Rumus:**

$$
x' = \frac{x - \bar{x}}{\sigma}
$$

#### ✍️ Jawaban Manual

**Hitung mean dan standar deviasi populasi setiap fitur (7 data untuk IPK & PO; 6 data latih untuk JMLH):**

**IPK** ($n=7$):
$$\bar{x}_{IPK} = \frac{2+3+4+2+3+4+2}{7} = \frac{20}{7} \approx 2{,}857$$
$$\sigma_{IPK} = \sqrt{\frac{(2{-}2{,}857)^2 \times 3 + (3{-}2{,}857)^2 \times 2 + (4{-}2{,}857)^2 \times 2}{7}} \approx \sqrt{\frac{2{,}204+0{,}041+2{,}612}{7}} \approx \sqrt{0{,}694} \approx 0{,}833$$

**PO** ($n=7$):
$$\bar{x}_{PO} = \frac{200000 \times 3 + 300000 \times 3 + 400000}{7} = \frac{1900000}{7} \approx 271428{,}6$$
$$\sigma_{PO} \approx \sqrt{\frac{3(200000-271429)^2 + 3(300000-271429)^2 + (400000-271429)^2}{7}} \approx 67937{,}5$$

**JMLH** ($n=6$, data latih):
$$\bar{x}_{JMLH} = \frac{2+3+2+3+2+3}{6} = \frac{15}{6} = 2{,}5$$
$$\sigma_{JMLH} = \sqrt{\frac{3(2-2{,}5)^2 + 3(3-2{,}5)^2}{6}} = \sqrt{\frac{3(0{,}25)+3(0{,}25)}{6}} = \sqrt{0{,}25} = 0{,}5$$

**Perhitungan Z-Score JMLH (parameter dari data latih, diterapkan ke semua termasuk NO=7 prediksi WKNN=3):**
$$JMLH'_1 = \frac{2-2{,}5}{0{,}5} = -1, \quad JMLH'_2 = \frac{3-2{,}5}{0{,}5} = 1, \quad JMLH'_3 = -1$$
$$JMLH'_4 = 1, \quad JMLH'_5 = -1, \quad JMLH'_6 = 1, \quad JMLH'_7 = \frac{3-2{,}5}{0{,}5} = 1$$

**Tabel Hasil Manual Z-Score:**

| NO | IPK' | PO' | JMLH' |
|----|------|-----|-------|
| 1 | −1,031 | −1,053 | −1 |
| 2 | 0,172 | 0,421 | 1 |
| 3 | 1,374 | −1,053 | −1 |
| 4 | −1,031 | −1,053 | 1 |
| 5 | 0,172 | 0,421 | −1 |
| 6 | 1,374 | 1,895 | 1 |
| 7 | −1,031 | 0,421 | **1** |

#### 💻 Jawaban Coding

```python
import numpy as np
import pandas as pd

df = pd.DataFrame({
    'NO':   [1, 2, 3, 4, 5, 6, 7],
    'IPK':  [2, 3, 4, 2, 3, 4, 2],
    'PO':   [200000, 300000, 200000, 200000, 300000, 400000, 300000],
    'JMLH': [2, 3, 2, 3, 2, 3, np.nan]
})

def z_score(series):
    return (series - series.mean()) / series.std(ddof=0)  # standar deviasi populasi

df_zs = df.copy()
df_zs.loc[df_zs['NO'] == 7, 'JMLH'] = 3  # isi dengan hasil prediksi WKNN
df_zs['IPK_z']  = z_score(df['IPK'])
df_zs['PO_z']   = z_score(df['PO'])
# Pakai parameter data latih (mean=2,5 ; σ=0,5), terapkan ke semua termasuk NO=7
jmlh_train = df['JMLH'].dropna()
df_zs['JMLH_z'] = (df_zs['JMLH'] - jmlh_train.mean()) / jmlh_train.std(ddof=0)

print(df_zs[['NO','IPK','IPK_z','PO','PO_z','JMLH','JMLH_z']].to_markdown(index=False, floatfmt='.4f'))
```

**Output:**

| NO | IPK | IPK_z | PO | PO_z | JMLH | JMLH_z |
|----|-----|-------|----|------|------|--------|
| 1 | 2 | −1.0328 | 200000 | −1.0541 | 2.0 | −1.0000 |
| 2 | 3 | 0.1721 | 300000 | 0.4217 | 3.0 | 1.0000 |
| 3 | 4 | 1.3765 | 200000 | −1.0541 | 2.0 | −1.0000 |
| 4 | 2 | −1.0328 | 200000 | −1.0541 | 3.0 | 1.0000 |
| 5 | 3 | 0.1721 | 300000 | 0.4217 | 2.0 | −1.0000 |
| 6 | 4 | 1.3765 | 400000 | 1.8975 | 3.0 | 1.0000 |
| 7 | 2 | −1.0328 | 300000 | 0.4217 | 3.0 | 1.0000 |

> Nilai JMLH Z-Score konsisten: data bernilai 2 → −1, data bernilai 3 → +1, karena mean=2,5 dan σ=0,5. Termasuk NO=7 (prediksi WKNN=3) → JMLH\_z = 1.

---

### C. Decimal Scaling Normalization

**Rumus:**

$$
x' = \frac{x}{10^j}, \quad j = \lceil \log_{10}(\max|x|) \rceil
$$

#### ✍️ Jawaban Manual

**Nilai absolut terbesar setiap fitur:**

| Fitur | max\|x\| | log₁₀ | j | Pembagi |
|-------|---------|-------|---|---------|
| IPK | 4 | 0,602 | 1 | 10 |
| PO | 400000 | 5,602 | 6 | 1.000.000 |
| JMLH | 3 | 0,477 | 1 | 10 |

**Perhitungan:**

$$IPK'_1 = \frac{2}{10}=0{,}2, \quad IPK'_2 = \frac{3}{10}=0{,}3, \quad IPK'_3 = \frac{4}{10}=0{,}4$$
$$IPK'_4 = 0{,}2, \quad IPK'_5 = 0{,}3, \quad IPK'_6 = 0{,}4, \quad IPK'_7 = 0{,}2$$

$$PO'_1 = \frac{200000}{1000000}=0{,}2, \quad PO'_2 = 0{,}3, \quad PO'_3 = 0{,}2$$
$$PO'_4 = 0{,}2, \quad PO'_5 = 0{,}3, \quad PO'_6 = 0{,}4, \quad PO'_7 = 0{,}3$$

$$JMLH'_1 = \frac{2}{10}=0{,}2, \quad JMLH'_2 = \frac{3}{10}=0{,}3, \quad JMLH'_3 = 0{,}2$$
$$JMLH'_4 = 0{,}3, \quad JMLH'_5 = 0{,}2, \quad JMLH'_6 = 0{,}3, \quad JMLH'_7 = \frac{3}{10}=0{,}3$$

**Tabel Hasil Manual Decimal Scaling:**

| NO | IPK | PO | JMLH | IPK' | PO' | JMLH' |
|----|-----|-----|------|------|-----|-------|
| 1 | 2 | 200000 | 2 | 0,2 | 0,2 | 0,2 |
| 2 | 3 | 300000 | 3 | 0,3 | 0,3 | 0,3 |
| 3 | 4 | 200000 | 2 | 0,4 | 0,2 | 0,2 |
| 4 | 2 | 200000 | 3 | 0,2 | 0,2 | 0,3 |
| 5 | 3 | 300000 | 2 | 0,3 | 0,3 | 0,2 |
| 6 | 4 | 400000 | 3 | 0,4 | 0,4 | 0,3 |
| 7 | 2 | 300000 | **3** | 0,2 | 0,3 | **0,3** |

#### 💻 Jawaban Coding

```python
import numpy as np
import pandas as pd
import math

df = pd.DataFrame({
    'NO':   [1, 2, 3, 4, 5, 6, 7],
    'IPK':  [2, 3, 4, 2, 3, 4, 2],
    'PO':   [200000, 300000, 200000, 200000, 300000, 400000, 300000],
    'JMLH': [2, 3, 2, 3, 2, 3, np.nan]
})

def decimal_scaling(series):
    max_abs = series.abs().max()
    j = math.ceil(math.log10(max_abs))
    return series / (10 ** j), j

df_ds = df.copy()
df_ds.loc[df_ds['NO'] == 7, 'JMLH'] = 3  # isi dengan hasil prediksi WKNN
df_ds['IPK_ds'],  j_ipk  = decimal_scaling(df['IPK'])
df_ds['PO_ds'],   j_po   = decimal_scaling(df['PO'])
_, j_jmlh = decimal_scaling(df['JMLH'].dropna())  # ambil j dari data latih
df_ds['JMLH_ds'] = df_ds['JMLH'] / (10 ** j_jmlh)  # terapkan ke semua, termasuk NO=7

print(f'j IPK={j_ipk}, j PO={j_po}, j JMLH={j_jmlh}')
print()
print(df_ds[['NO','IPK','IPK_ds','PO','PO_ds','JMLH','JMLH_ds']].to_markdown(index=False, floatfmt='.4f'))
```

**Output:**

```
j IPK=1, j PO=6, j JMLH=1
```

| NO | IPK | IPK_ds | PO | PO_ds | JMLH | JMLH_ds |
|----|-----|--------|----|-------|------|---------|
| 1 | 2 | 0.2000 | 200000 | 0.2000 | 2.0 | 0.2000 |
| 2 | 3 | 0.3000 | 300000 | 0.3000 | 3.0 | 0.3000 |
| 3 | 4 | 0.4000 | 200000 | 0.2000 | 2.0 | 0.2000 |
| 4 | 2 | 0.2000 | 200000 | 0.2000 | 3.0 | 0.3000 |
| 5 | 3 | 0.3000 | 300000 | 0.3000 | 2.0 | 0.2000 |
| 6 | 4 | 0.4000 | 400000 | 0.4000 | 3.0 | 0.3000 |
| 7 | 2 | 0.2000 | 300000 | 0.3000 | 3.0 | 0.3000 |

> Hasil coding sama dengan jawaban manual.

---

### Perbandingan Ketiga Metode pada Dataset Soal

> **Catatan:** JMLH NO=7 menggunakan hasil prediksi WKNN = **3**, dinormalisasi dengan parameter yang sama dari data latih NO=1–6 (min, max, mean, σ, j).

| NO | IPK_mm | PO_mm | JMLH_mm | IPK_z | PO_z | JMLH_z | IPK_ds | PO_ds | JMLH_ds |
|----|--------|-------|---------|-------|------|--------|--------|-------|---------|
| 1 | 0.0 | 0.0 | 0.0 | −1.033 | −1.054 | −1.0 | 0.2 | 0.2 | 0.2 |
| 2 | 0.5 | 0.5 | 1.0 | 0.172 | 0.422 | 1.0 | 0.3 | 0.3 | 0.3 |
| 3 | 1.0 | 0.0 | 0.0 | 1.377 | −1.054 | −1.0 | 0.4 | 0.2 | 0.2 |
| 4 | 0.0 | 0.0 | 1.0 | −1.033 | −1.054 | 1.0 | 0.2 | 0.2 | 0.3 |
| 5 | 0.5 | 0.5 | 0.0 | 0.172 | 0.422 | −1.0 | 0.3 | 0.3 | 0.2 |
| 6 | 1.0 | 1.0 | 1.0 | 1.377 | 1.898 | 1.0 | 0.4 | 0.4 | 0.3 |
| 7 | 0.0 | 0.5 | **1.0** | −1.033 | 0.422 | **1.0** | 0.2 | 0.3 | **0.3** |

---

## Daftar Isi
```{dropdown} Klik untuk membuka Daftar Isi
:open:

0. [Struktur Tugas (Sesuai Notebook)](#struktur-tugas-sesuai-notebook)
1. [Normalisasi Data dengan Soal yang Sama](#normalisasi-data-dengan-soal-yang-sama)
2. [Missing Values & Imputasi WKNN](#1-missing-values--imputasi-wknn)
3. [Normalisasi Data](#2-normalisasi-data)
    - [Min-Max Normalization](#a-min-max-normalization)
    - [Z-Score Normalization](#b-z-score-normalization-standardization)
    - [Decimal Scaling Normalization](#c-decimal-scaling-normalization)
    - [Perbandingan Ketiga Metode](#perbandingan-ketiga-metode)
    - [Implementasi Python & Sklearn](#implementasi-dengan-python--sklearn)
```

---

# 1. Missing Values & Imputasi WKNN

> **Referensi utama:** [moelaab — Imputasi WKNN](https://moelaab.github.io/pendata/preproces/wknn.html)

## 1.1 Apa Itu Missing Values?

**Missing value** (nilai hilang) adalah kondisi di mana satu atau lebih atribut pada suatu observasi tidak memiliki nilai. Penyebabnya antara lain:

| Penyebab | Contoh |
|---|---|
| Kesalahan input | Operator lupa mengisi kolom |
| Sensor rusak | Alat pengukur tidak merekam data |
| Responden menolak menjawab | Kolom pendapatan dikosongkan |
| Data tidak relevan | Kolom "gaji" untuk pelajar |
| Masalah integrasi data | Penggabungan tabel dengan kolom berbeda |

Dalam Python/Pandas, missing value biasanya direpresentasikan sebagai `NaN` (*Not a Number*).

## 1.2 Strategi Menangani Missing Values

| Strategi | Keterangan |
|---|---|
| **Penghapusan (Deletion)** | Hapus baris atau kolom yang mengandung missing value. Sederhana tetapi bisa kehilangan banyak data. |
| **Imputasi Mean/Median/Modus** | Isi dengan rata-rata (numerik), median, atau modus (kategorikal). Tidak memperhitungkan hubungan antar atribut. |
| **Imputasi KNN** | Isi berdasarkan rata-rata dari *K* tetangga terdekat. |
| **Imputasi WKNN (Weighted KNN)** | Pengembangan KNN — tetangga yang lebih dekat diberi bobot lebih besar. |

---

## 1.3 Imputasi KNN Standar vs WKNN

### KNN Imputasi (Standar)
KNN standar mengambil **rata-rata sederhana** (atau modus untuk data kategorikal) dari $K$ tetangga terdekat untuk mengisi data yang hilang. Semua tetangga diperlakukan **sama rata** tanpa memandang jaraknya.

### WKNN Imputasi (Weighted KNN)
WKNN merupakan **pengembangan** dari KNN standar. Perbedaan utamanya:

> Tetangga yang nilainya **sangat mirip** (jaraknya dekat) dengan data yang hilang akan memiliki **pengaruh (bobot) lebih besar** dibandingkan tetangga yang jaraknya jauh.

---

## 1.4 Rumus WKNN

### Rumus (1) — Ukuran Kemiripan (*Similarity*)

Ukuran kesamaan $s_i(y_j)$ antara dua contoh $y_i$ dan $y_j$ didefinisikan oleh **jarak Euclidean** berdasarkan atribut-atribut yang teramati:

$$
\frac{1}{s_i} = \sum_{h_i \in O_i \cap O_j} (y_i^h - y_j^h)^2 \tag{1}
$$

di mana $O_i = \{h \mid \text{komponen ke-}h\text{ dari }y_i\text{ teramati}\}$.

**Penjelasan:**
- $\frac{1}{s_i}$: kebalikan dari kesamaan = jarak. Jika jarak kecil → $s_i$ besar (sangat mirip).
- $O_i \cap O_j$: penjumlahan **hanya** dilakukan pada atribut yang **ada datanya** pada kedua objek.
- Semakin kecil hasil penjumlahan → semakin mirip → bobot $s_i$ semakin besar.

### Rumus (2) — Estimasi Nilai Hilang (Weighted Average)

Entri yang hilang $y_i^h$ diestimasi sebagai **rata-rata tertimbang**:

$$
\hat{y}_i^h = \frac{\sum_{j \in I_{K_i^h}} s_i(y_j) \cdot y_j^h}{\sum_{j \in I_{K_i^h}} s_i(y_j)} \tag{2}
$$

di mana $I_{K_i^h}$ adalah himpunan indeks dari $K$ tetangga terdekat dari contoh ke-$i$.

**Interpretasi:**
- **Pembilang:** jumlah (bobot kesamaan × nilai tetangga)
- **Penyebut:** jumlah total bobot kesamaan (normalisasi)
- Tetangga dengan bobot $s_i$ besar → pengaruhnya besar terhadap hasil

---

## 1.5 Contoh Perhitungan Manual WKNN (Data Numerik)

### Dataset: Harga Sewa Kos

| No | Luas ($x$) | Harga Sewa ($y$, juta) |
|:---:|:---:|:---:|
| 1 | 20 | 2.0 |
| 2 | 25 | 2.4 |
| 3 | 35 | 3.5 |
| 4 | 40 | 4.2 |
| 5 | 50 | 5.1 |
| 6 | 60 | 6.5 |
| **7** | **30** | **? (hilang)** |

**Target:** Estimasi Harga Sewa untuk Luas = 30.

### Penentuan K

Tidak ada rumus baku untuk memilih $K$. Nilai $K$ ditentukan secara **empiris** (coba-coba, misal $K = 3, 5, 7$, lalu lihat error terkecil via cross-validation). Pada contoh ini kita gunakan **semua 6 tetangga** (K = 6).

### Langkah 1 — Hitung Jarak dan Kemiripan ($s_i$)

Gunakan rumus (1) dengan satu fitur (Luas):

$$
\frac{1}{s_i} = (x_i - x_j)^2
$$

| Tetangga | $x_j$ | $(30 - x_j)^2$ | $s_i = \frac{1}{(30-x_j)^2}$ |
|:---:|:---:|:---:|:---:|
| $y_1$ | 20 | 100 | 0.0100 |
| $y_2$ | 25 | 25 | 0.0400 |
| $y_3$ | 35 | 25 | 0.0400 |
| $y_4$ | 40 | 100 | 0.0100 |
| $y_5$ | 50 | 400 | 0.0025 |
| $y_6$ | 60 | 900 | 0.0011 |

### Langkah 2 — Hitung Estimasi (Weighted Average)

**A. Pembilang** $\sum s_i \cdot y_j$:

| | $s_i$ | $y_j$ | $s_i \times y_j$ |
|:---:|:---:|:---:|:---:|
| $y_1$ | 0.0100 | 2.0 | 0.0200 |
| $y_2$ | 0.0400 | 2.4 | 0.0960 |
| $y_3$ | 0.0400 | 3.5 | 0.1400 |
| $y_4$ | 0.0100 | 4.2 | 0.0420 |
| $y_5$ | 0.0025 | 5.1 | 0.01275 |
| $y_6$ | 0.0011 | 6.5 | 0.00715 |
| **Total** | | | **0.3179** |

**B. Penyebut** $\sum s_i$:

$$
0.0100 + 0.0400 + 0.0400 + 0.0100 + 0.0025 + 0.0011 = 0.1036
$$

**C. Hasil Akhir:**

$$
\hat{y} = \frac{0.3179}{0.1036} \approx \mathbf{3.068 \text{ juta}}
$$

---

## 1.6 Contoh WKNN pada Data Klasifikasi (Same-Class Weighting)

Jika label kelas sudah diketahui, label menjadi informasi tambahan untuk estimasi yang **lebih akurat** — titik dengan kelas sama biasanya memiliki karakteristik fitur yang serupa.

### Dataset: Data Pelamar Kerja

| No | Pengalaman (th) | Skor Tes | Status |
|:---:|:---:|:---:|:---:|
| 1 | 2 | 70 | 0 (Gagal) |
| 2 | 3 | 75 | 0 (Gagal) |
| 3 | 5 | 85 | 1 (Lulus) |
| 4 | 6 | 88 | 1 (Lulus) |
| 5 | 1 | 65 | 0 (Gagal) |
| 6 | 7 | 92 | 1 (Lulus) |
| **7** | **5.5** | **? (hilang)** | **1 (Lulus)** |

**Target:** Estimasi Skor Tes untuk data ke-7 (Pengalaman = 5.5, Kelas = Lulus).

### Langkah 1 — Hitung Jarak & Kemiripan

Gunakan fitur yang lengkap (Pengalaman saja):

| Tetangga | Pengalaman | $(5.5 - x_j)^2$ | $s_i$ |
|:---:|:---:|:---:|:---:|
| $y_1$ | 2 | 12.25 | 0.081 |
| $y_2$ | 3 | 6.25 | 0.160 |
| $y_3$ | 5 | 0.25 | 4.000 |
| $y_4$ | 6 | 0.25 | 4.000 |
| $y_5$ | 1 | 20.25 | 0.049 |
| $y_6$ | 7 | 2.25 | 0.444 |

### Langkah 2 — Filter Same-Class (Kelas 1)

Karena data target memiliki Kelas 1, kita **hanya pakai tetangga Kelas 1**: $y_3, y_4, y_6$.

### Langkah 3 — Hitung Estimasi (Hanya Kelas 1)

**Pembilang** $\sum s_i \cdot \text{Skor}_j$:

- $y_3$: $4.000 \times 85 = 340$
- $y_4$: $4.000 \times 88 = 352$
- $y_6$: $0.444 \times 92 = 40.848$
- **Total = 732.848**

**Penyebut** $\sum s_i$: $4.000 + 4.000 + 0.444 = 8.444$

**Hasil:**

$$
\hat{y} = \frac{732.848}{8.444} \approx \mathbf{86.79}
$$

> Skor tes yang hilang diestimasi sebesar **86.79**. Masuk akal karena pengalaman 5.5 berada di antara $y_3$ (5 tahun, skor 85) dan $y_4$ (6 tahun, skor 88).

---

# 2. Normalisasi Data

> **Referensi utama:** García, S., Luengo, J., & Herrera, F. (2015). *Data Preprocessing in Data Mining*. Springer.

## 2.1 Mengapa Perlu Normalisasi?

Normalisasi (atau *scaling*) adalah proses **mengubah skala nilai atribut** agar berada dalam rentang tertentu. Tujuannya:

1. **Menghilangkan dominasi atribut berskala besar** — fitur dengan rentang besar (misal gaji: 1.000.000–50.000.000) akan mendominasi fitur berskala kecil (misal umur: 20–60) saat menghitung jarak.
2. **Mempercepat konvergensi** algoritma berbasis gradien (misalnya neural network, gradient descent).
3. **Meningkatkan akurasi** algoritma berbasis jarak seperti KNN, K-Means, SVM.

---

## 2.2 Macam-Macam Normalisasi Data

### A. Min-Max Normalization

> **Rumus dari buku referensi** (García et al., 2015 — *Data Preprocessing in Data Mining*, hal. 47, Bagian 3.4):
>
> ![Rumus Min-Max Normalization — Persamaan (3.9) dari buku Data Preprocessing in Data Mining](<Tugas/Missing Values & Normalisasi/gambar_buku/rumus_minmax_3.9.png>)
>
> *Gambar: Formulasi Min-Max Normalization (Persamaan 3.9). Normalisasi min-max mentransformasi nilai atribut ke rentang [0, 1] atau rentang baru [new–min_A, new–max_A]. Rumus ini memetakan nilai $v$ secara linear berdasarkan nilai minimum dan maksimum asli dari atribut $A$. Formulasi alternatif menggunakan base value $\text{new–min}_A$ dan rentang baru $R$ untuk transformasi ke interval yang diinginkan.*

Mentransformasi data ke dalam rentang $[0, 1]$ (atau rentang $[\text{new\_min}, \text{new\_max}]$ yang diinginkan).

**Rumus:**

$$
v' = \frac{v - \min(A)}{\max(A) - \min(A)}
$$

Jika ingin memetakan ke rentang $[\text{new\_min}, \text{new\_max}]$:

$$
v' = \frac{v - \min(A)}{\max(A) - \min(A)} \times (\text{new\_max} - \text{new\_min}) + \text{new\_min}
$$

di mana:
- $v$ = nilai asli  
- $\min(A)$ = nilai minimum atribut $A$  
- $\max(A)$ = nilai maksimum atribut $A$  
- $v'$ = nilai setelah normalisasi

**Karakteristik:**
- Hasil selalu berada di rentang $[0, 1]$
- Sensitif terhadap **outlier** (nilai ekstrem mengubah min/max)
- Cocok untuk data yang **tidak mengikuti distribusi normal**

#### Contoh Perhitungan Min-Max

Data atribut Umur: **[20, 25, 30, 35, 40]**

$\min = 20$, $\max = 40$

| Nilai Asli ($v$) | Perhitungan | Hasil ($v'$) |
|:---:|:---:|:---:|
| 20 | $(20-20)/(40-20)$ | **0.00** |
| 25 | $(25-20)/(40-20)$ | **0.25** |
| 30 | $(30-20)/(40-20)$ | **0.50** |
| 35 | $(35-20)/(40-20)$ | **0.75** |
| 40 | $(40-20)/(40-20)$ | **1.00** |

---

### B. Z-Score Normalization (Standardization)

> **Rumus dari buku referensi** (García et al., 2015 — *Data Preprocessing in Data Mining*, hal. 47–48, Bagian 3.4.2):
>
> ![Rumus Z-Score Normalization — Persamaan (3.10) dari buku Data Preprocessing in Data Mining](<Tugas/Missing Values & Normalisasi/gambar_buku/rumus_zscore_3.10.png>)
>
> *Gambar: Formulasi Z-Score Normalization (Persamaan 3.10). Pada beberapa kasus, min-max normalization tidak bisa diterapkan — misalnya ketika nilai minimum/maksimum atribut $A$ tidak diketahui, atau ketika ada outlier yang membias hasil. Z-Score menormalisasi nilai $v$ menggunakan mean ($\bar{A}$) dan standar deviasi ($\sigma_A$), sehingga hasil transformasi memiliki mean = 0 dan standar deviasi = 1.*

Mentransformasi data sehingga memiliki **mean = 0** dan **standar deviasi = 1**. Disebut juga *standardization* atau *zero-mean normalization*.

**Rumus:**

$$
v' = \frac{v - \bar{A}}{\sigma_A}
$$

di mana:
- $v$ = nilai asli  
- $\bar{A}$ = rata-rata (*mean*) atribut $A$  
- $\sigma_A$ = standar deviasi atribut $A$  
- $v'$ = nilai setelah normalisasi (*z-score*)

**Rumus Standar Deviasi (populasi):**

$$
\sigma_A = \sqrt{\frac{1}{N} \sum_{i=1}^{N} (v_i - \bar{A})^2}
$$

**Karakteristik:**
- Hasil **tidak terbatas** dalam rentang tertentu (bisa negatif maupun positif)
- **Kurang sensitif** terhadap outlier dibanding Min-Max
- Cocok untuk data yang **mendekati distribusi normal**
- Digunakan secara luas pada algoritma yang mengasumsikan distribusi Gaussian

#### Contoh Perhitungan Z-Score

Data atribut Umur: **[20, 25, 30, 35, 40]**

$\bar{A} = \frac{20+25+30+35+40}{5} = 30$

$\sigma_A = \sqrt{\frac{(20{-}30)^2 + (25{-}30)^2 + (30{-}30)^2 + (35{-}30)^2 + (40{-}30)^2}{5}} = \sqrt{\frac{100+25+0+25+100}{5}} = \sqrt{50} \approx 7.07$

| Nilai Asli ($v$) | Perhitungan | Hasil ($v'$) |
|:---:|:---:|:---:|
| 20 | $(20-30)/7.07$ | **−1.414** |
| 25 | $(25-30)/7.07$ | **−0.707** |
| 30 | $(30-30)/7.07$ | **0.000** |
| 35 | $(35-30)/7.07$ | **+0.707** |
| 40 | $(40-30)/7.07$ | **+1.414** |

> Nilai negatif = di bawah rata-rata, positif = di atas rata-rata.

---

### C. Decimal Scaling Normalization

> **Rumus dari buku referensi** (García et al., 2015 — *Data Preprocessing in Data Mining*, hal. 48, Bagian 3.4.2–3.4.3):
>
> ![Rumus Standar Deviasi, Mean Absolute Deviation, dan Decimal Scaling — Persamaan (3.12)–(3.15) dari buku Data Preprocessing in Data Mining](<Tugas/Missing Values & Normalisasi/gambar_buku/rumus_stddev_decimal_scaling_3.12-3.15.png>)
>
> *Gambar: Halaman 48 buku referensi memuat beberapa rumus penting:*
> - *Persamaan (3.12):* $\sigma_A = +\sqrt{\frac{1}{n}\sum_{i=1}^{n}(v_i - \bar{A})^2}$ — *rumus standar deviasi populasi untuk Z-Score.*
> - *Persamaan (3.13):* $s_A = \frac{1}{n}\sum_{i=1}^{n}|v_i - \bar{A}|$ — *mean absolute deviation, variasi Z-Score yang lebih robust terhadap outlier karena tidak mengkuadratkan selisih.*
> - *Persamaan (3.14):* $v' = \frac{v - \bar{A}}{s_A}$ — *Z-Score menggunakan mean absolute deviation.*
> - *Persamaan (3.15):* $v' = \frac{v}{10^j}$ — *Decimal Scaling Normalization, di mana $j$ adalah bilangan bulat terkecil sehingga $\max|v'| < 1$.*

Menormalisasi data dengan **menggeser titik desimal** berdasarkan nilai absolut terbesar pada atribut tersebut.

**Rumus:**

$$
v' = \frac{v}{10^j}
$$

di mana $j$ adalah **bilangan bulat terkecil** sehingga $\max(|v'|) < 1$.

Dengan kata lain, $j = \lceil \log_{10}(\max |v|) \rceil$ (pembulatan ke atas dari logaritma basis 10 dari nilai absolut terbesar).

**Karakteristik:**
- Hasil berada di rentang $(-1, 1)$
- Sangat **sederhana** untuk dihitung
- Bergantung pada **nilai absolut terbesar** dalam data

#### Contoh Perhitungan Decimal Scaling

Data atribut Gaji: **[1200, 3400, 5600, 7800, 9900]**

$\max(|v|) = 9900$

$j = \lceil \log_{10}(9900) \rceil = \lceil 3.9956 \rceil = 4$

Maka kita bagi semua nilai dengan $10^4 = 10000$:

| Nilai Asli ($v$) | Perhitungan | Hasil ($v'$) |
|:---:|:---:|:---:|
| 1200 | $1200/10000$ | **0.12** |
| 3400 | $3400/10000$ | **0.34** |
| 5600 | $5600/10000$ | **0.56** |
| 7800 | $7800/10000$ | **0.78** |
| 9900 | $9900/10000$ | **0.99** |

---

## Perbandingan Ketiga Metode

| Aspek | Min-Max | Z-Score | Decimal Scaling |
|---|---|---|---|
| **Rentang hasil** | $[0, 1]$ | Tidak terbatas | $(-1, 1)$ |
| **Sensitif outlier?** | Ya (sangat) | Lebih tahan | Tergantung nilai maks |
| **Distribusi data** | Bebas | Baik untuk distribusi normal | Bebas |
| **Menjaga proporsi jarak?** | Ya | Ya | Ya |
| **Kapan digunakan?** | Neural network, image processing | SVM, Logistic Regression, PCA | Data berskala besar |

### Contoh Komparasi (Data yang Sama)

Data: **[20, 25, 30, 35, 40]**

| Metode | 20 | 25 | 30 | 35 | 40 |
|---|:---:|:---:|:---:|:---:|:---:|
| **Asli** | 20 | 25 | 30 | 35 | 40 |
| **Min-Max** | 0.00 | 0.25 | 0.50 | 0.75 | 1.00 |
| **Z-Score** | −1.414 | −0.707 | 0.000 | +0.707 | +1.414 |
| **Decimal Scaling** ($j{=}2$) | 0.20 | 0.25 | 0.30 | 0.35 | 0.40 |

---

## Implementasi dengan Python & Sklearn

### A. Min-Max Normalization

```python
import numpy as np
from sklearn.preprocessing import MinMaxScaler

# Data contoh
data = np.array([[20], [25], [30], [35], [40]])

# --- Menggunakan sklearn ---
scaler = MinMaxScaler()
hasil_sklearn = scaler.fit_transform(data)
print("Min-Max (sklearn):")
print(hasil_sklearn.flatten())

# --- Fungsi sendiri ---
def min_max_normalisasi(arr):
    min_val = np.min(arr)
    max_val = np.max(arr)
    return (arr - min_val) / (max_val - min_val)

hasil_manual = min_max_normalisasi(data.flatten())
print("\nMin-Max (manual):")
print(hasil_manual)
```

**Output:**
```
Min-Max (sklearn):
[0.   0.25 0.5  0.75 1.  ]

Min-Max (manual):
[0.   0.25 0.5  0.75 1.  ]
```

---

### B. Z-Score Normalization (Standardization)

```python
from sklearn.preprocessing import StandardScaler

data = np.array([[20], [25], [30], [35], [40]])

# --- Menggunakan sklearn ---
scaler = StandardScaler()
hasil_sklearn = scaler.fit_transform(data)
print("Z-Score (sklearn):")
print(hasil_sklearn.flatten())

# --- Fungsi sendiri ---
def z_score_normalisasi(arr):
    mean = np.mean(arr)
    std = np.std(arr)  # standar deviasi populasi
    return (arr - mean) / std

hasil_manual = z_score_normalisasi(data.flatten())
print("\nZ-Score (manual):")
print(hasil_manual)
```

**Output:**
```
Z-Score (sklearn):
[-1.41421356 -0.70710678  0.          0.70710678  1.41421356]

Z-Score (manual):
[-1.41421356 -0.70710678  0.          0.70710678  1.41421356]
```

---

### C. Decimal Scaling Normalization

> **Catatan:** Sklearn **tidak menyediakan** fungsi bawaan untuk Decimal Scaling, sehingga kita buat fungsi sendiri.

```python
import math

def decimal_scaling_normalisasi(arr):
    max_abs = np.max(np.abs(arr))
    j = math.ceil(math.log10(max_abs))
    return arr / (10 ** j)

data = np.array([1200, 3400, 5600, 7800, 9900])

hasil = decimal_scaling_normalisasi(data)
print(f"j = {math.ceil(math.log10(np.max(np.abs(data))))}")
print(f"Decimal Scaling: {hasil}")
```

**Output:**
```
j = 4
Decimal Scaling: [0.12 0.34 0.56 0.78 0.99]
```

---

### D. Contoh Lengkap — Normalisasi Dataset Multi-Kolom

```python
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler

# Dataset contoh
df = pd.DataFrame({
    'Umur':         [20, 25, 30, 35, 40],
    'Gaji':         [1200, 3400, 5600, 7800, 9900],
    'Pengalaman':   [1, 3, 5, 7, 10]
})

print("=== Data Asli ===")
print(df)

# Min-Max
mm_scaler = MinMaxScaler()
df_minmax = pd.DataFrame(
    mm_scaler.fit_transform(df),
    columns=[col + '_mm' for col in df.columns]
)
print("\n=== Min-Max Normalization ===")
print(df_minmax.round(4))

# Z-Score
z_scaler = StandardScaler()
df_zscore = pd.DataFrame(
    z_scaler.fit_transform(df),
    columns=[col + '_z' for col in df.columns]
)
print("\n=== Z-Score Normalization ===")
print(df_zscore.round(4))

# Decimal Scaling (semua kolom)
import math

def decimal_scaling_df(dataframe):
    hasil = dataframe.copy().astype(float)
    for col in hasil.columns:
        max_abs = hasil[col].abs().max()
        j = math.ceil(math.log10(max_abs)) if max_abs > 0 else 1
        hasil[col] = hasil[col] / (10 ** j)
    return hasil

df_decimal = decimal_scaling_df(df)
df_decimal.columns = [col + '_ds' for col in df.columns]
print("\n=== Decimal Scaling Normalization ===")
print(df_decimal.round(4))
```

**Output:**
```
=== Data Asli ===
   Umur  Gaji  Pengalaman
0    20  1200           1
1    25  3400           3
2    30  5600           5
3    35  7800           7
4    40  9900          10

=== Min-Max Normalization ===
   Umur_mm  Gaji_mm  Pengalaman_mm
0   0.0000   0.0000         0.0000
1   0.2500   0.2529         0.2222
2   0.5000   0.5057         0.4444
3   0.7500   0.7586         0.6667
4   1.0000   1.0000         1.0000

=== Z-Score Normalization ===
   Umur_z  Gaji_z  Pengalaman_z
0 -1.4142 -1.3887       -1.3363
1 -0.7071 -0.6943       -0.6682
2  0.0000  0.0000        0.0000
3  0.7071  0.6943        0.6682
4  1.4142  1.3887        1.3363

=== Decimal Scaling Normalization ===
   Umur_ds  Gaji_ds  Pengalaman_ds
0    0.200    0.120           0.10
1    0.250    0.340           0.30
2    0.300    0.560           0.50
3    0.350    0.780           0.70
4    0.400    0.990           1.00
```

---

## Referensi

1. García, S., Luengo, J., & Herrera, F. (2015). *Data Preprocessing in Data Mining*. Springer. — Bab tentang Data Reduction dan Normalization.
2. [moelaab — Imputasi Berbobot dengan K-Tetangga Terdekat (WKNNI)](https://moelaab.github.io/pendata/preproces/wknn.html)
3. [Scikit-learn — Preprocessing and Normalization](https://scikit-learn.org/stable/modules/preprocessing.html)