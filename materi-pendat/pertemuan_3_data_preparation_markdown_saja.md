# DATA PREPARATION — PERTEMUAN 3
## Studi Kasus: Iris + Data Campuran (Mixed-Type)

```{admonition} Identitas Mahasiswa
:class: note

| | |
|---|---|
| **Nama** | Eka Safanoli Safitri |
| **NIM** | 240411100072 |
| **Mata Kuliah** | Penambangan Data |
| **Pertemuan** | 3 — Data Preparation |
```

Dokumen ini melanjutkan materi Data Preparation dalam kerangka **CRISP-DM** yang mencakup:
identifikasi missing value, statistik deskriptif, encoding, scaling, **pengukuran jarak**, dan penanganan **data campuran (mixed-type)**.

---

## ✅ Tugas Pertemuan 3

```{admonition} Tugas yang Harus Diselesaikan
:class: important

Berikut tiga tugas utama pada Pertemuan 3 beserta status penyelesaiannya:

| No | Tugas | Status | Keterangan |
|:--:|-------|:------:|------------|
| 1 | **Mengukur Jarak** — ditempatkan di bawah bagian *Data Understanding* | ✅ Selesai | Euclidean, Manhattan, Spearman, Hamming pada data Iris (CSV & SQL) — lihat **Section 3.13–3.14** |
| 2 | **Buat/Cari Data Campuran** — mengandung tipe ordinal, numerik, kategorikal, dan biner | ✅ Selesai | Dataset **HR Analytics** (`HR_comma_sep.csv` + PostgreSQL `HRAnalytics`) — lihat **Section 3.15** |
| 3 | **Lakukan Pengukuran Jarak pada Data Campuran** tersebut | ✅ Selesai | 4 metrik jarak diterapkan di Orange pada data HR Analytics — lihat **Section 3.15.5** |
```

> **File Orange Workflow:** dapat diunduh di {download}`HRAnalytics.ows <DataCampuranPertemuan3/HR Analytics/HRAnalytics.ows>`

---

## 3.1 Konsep CRISP-DM

**CRISP-DM** (Cross-Industry Standard Process for Data Mining) adalah metodologi standar dalam proyek data mining yang terdiri dari 6 fase berurutan:

| No | Fase | Keterangan |
|----|------|------------|
| 1 | Business Understanding | Memahami tujuan bisnis dan kebutuhan analisis |
| 2 | Data Understanding | Eksplorasi awal data, statistik deskriptif |
| 3 | **Data Preparation** | Pembersihan, transformasi, seleksi fitur |
| 4 | Modeling | Membangun model machine learning |
| 5 | Evaluation | Mengevaluasi performa model |
| 6 | Deployment | Implementasi model ke sistem nyata |

> Pertemuan ini berfokus pada fase **Data Preparation** — fase paling kritis yang memakan 60–70% waktu proyek data mining.

---

## 3.2 Persiapan Lingkungan

Sebelum memulai analisis, kita impor library yang dibutuhkan. Setiap library memiliki peran khusus dalam proses data preparation.

```python
%matplotlib inline
import pandas as pd          # manipulasi dan analisis data tabular
import numpy as np           # komputasi numerik dan array
import matplotlib.pyplot as plt  # visualisasi data

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import pairwise_distances
```

| Library | Fungsi Utama |
|---------|-------------|
| `pandas` | Load CSV, manipulasi DataFrame, groupby, describe |
| `numpy` | Operasi array, kalkulasi jarak manual |
| `matplotlib` | Plot histogram, visualisasi distribusi |
| `StandardScaler` | Normalisasi fitur (mean=0, std=1) sebelum hitung jarak |
| `LabelEncoder` | Konversi label kategorikal ke numerik |
| `pairwise_distances` | Hitung distance matrix antar semua pasang data |

---

## 3.3 Memuat Dataset Awal

Dataset dimuat kembali untuk memastikan seluruh proses preparation dilakukan pada data mentah yang konsisten.

```python
df = pd.read_csv("IRIS.csv")
df.head()
```

**Output `df.head()`** — 5 baris pertama dataset Iris:

| | sepal_length | sepal_width | petal_length | petal_width | species |
|--|---|---|---|---|---|
| **0** | 5.1 | 3.5 | 1.4 | 0.2 | Iris-setosa |
| **1** | 4.9 | 3.0 | 1.4 | 0.2 | Iris-setosa |
| **2** | 4.7 | 3.2 | 1.3 | 0.2 | Iris-setosa |
| **3** | 4.6 | 3.1 | 1.5 | 0.2 | Iris-setosa |
| **4** | 5.0 | 3.6 | 1.4 | 0.2 | Iris-setosa |

Dataset ini berisi **150 baris** dan **5 kolom**, terdiri dari 4 fitur numerik dan 1 kolom target kategorikal.

---

## 3.4 Penjelasan: Fitur vs Kelas (Target)

Memahami perbedaan **fitur** dan **kelas** adalah dasar sebelum melakukan pemodelan supervised learning.

- **Fitur (features / attributes)** = kolom input yang menjadi karakteristik bunga, digunakan sebagai variabel independen (X).
- **Kelas (class / label / target)** = kolom output yang ingin diprediksi, merupakan variabel dependen (y).

**Tabel Identifikasi Kolom Dataset Iris:**

| Kolom | Tipe Data | Peran | Keterangan |
|-------|-----------|-------|------------|
| `sepal_length` | Numerik (float) | **Fitur** | Panjang kelopak luar / sepal (cm) |
| `sepal_width` | Numerik (float) | **Fitur** | Lebar kelopak luar / sepal (cm) |
| `petal_length` | Numerik (float) | **Fitur** | Panjang mahkota bunga / petal (cm) |
| `petal_width` | Numerik (float) | **Fitur** | Lebar mahkota bunga / petal (cm) |
| `species` | Kategorikal (string) | **Kelas (Target)** | Jenis bunga: *setosa*, *versicolor*, *virginica* |

✅ **Kesimpulan:** `sepal_length`, `sepal_width`, `petal_length`, `petal_width` → **fitur**.
Sedangkan `Iris-setosa`, `Iris-versicolor`, `Iris-virginica` → **kelas/label**.

> Jika membuat kolom `species_encoded`, itu hanya versi **numerik** dari kelas — bukan fitur baru.

---

## Pembersihan Data

---

## 3.5 Identifikasi Missing Value

Identifikasi missing value adalah langkah **pertama dan wajib** dalam data preparation. Data yang memiliki nilai kosong dapat menyebabkan error pada algoritma atau hasil analisis yang bias.

### 3.5.1 Jumlah Missing per Kolom

```python
missing_count = df.isnull().sum()
missing_count
```

### 3.5.2 Persentase Missing per Kolom

```python
missing_percent = (df.isnull().mean() * 100).round(2)
pd.DataFrame({'missing_count': missing_count, 'missing_%': missing_percent})
```

**Hasil Pengecekan Missing Value Dataset Iris:**

| Kolom | Missing Count | Missing % | Status |
|-------|:---:|:---:|:---:|
| `sepal_length` | 0 | 0.00% | ✅ Lengkap |
| `sepal_width` | 0 | 0.00% | ✅ Lengkap |
| `petal_length` | 0 | 0.00% | ✅ Lengkap |
| `petal_width` | 0 | 0.00% | ✅ Lengkap |
| `species` | 0 | 0.00% | ✅ Lengkap |

> Dataset Iris **tidak memiliki missing value**, sehingga tidak diperlukan proses imputasi (pengisian nilai kosong).

### 3.5.3 Menampilkan Baris yang Memiliki Missing (jika ada)

```python
rows_with_missing = df[df.isnull().any(axis=1)]
rows_with_missing.head()
```

---

## Statistik Deskriptif

---

## 3.15 Data Campuran: HR Analytics (Contoh dan Unduhan)

Pada contoh data campuran kami menggunakan dataset *HR Analytics* yang berisi campuran atribut numerik, kategorikal, ordinal, dan biner. Dataset ini tersedia di folder repository pada:

- `DataCampuranPertemuan3/HR Analytics/HR_comma_sep.csv` (CSV)
- `DataCampuranPertemuan3/HR Analytics/HRAnalytics.ows` (Orange workflow)
- `DataCampuranPertemuan3/HR Analytics/HRAnalytics.sql` (dump / skrip SQL)

Unduh langsung file-file tersebut dari link berikut (relatif ke root repository):

- [Download CSV — HR_comma_sep.csv](../DataCampuranPertemuan3/HR%20Analytics/HR_comma_sep.csv)
- [Download Orange workflow — HRAnalytics.ows](../DataCampuranPertemuan3/HR%20Analytics/HRAnalytics.ows)
- [Download SQL script — HRAnalytics.sql](../DataCampuranPertemuan3/HR%20Analytics/HRAnalytics.sql)

Penjelasan singkat:

- Kolom numerik: `satisfaction_level`, `last_evaluation`, `average_montly_hours`, dll.
- Kolom kategorikal: `Department`, `salary` (ordinal: low < medium < high)
- Kolom biner: `Work_accident`, `left`, `promotion_last_5years`

Langkah yang dilakukan pada data campuran:

1. Identifikasi tipe kolom (numerik / kategorikal / ordinal / biner).
2. Encoding kolom kategorikal (mis. `LabelEncoder` atau `OneHotEncoder` untuk modeling; untuk pengukuran jarak campuran gunakan teknik yang sesuai seperti Gower distance atau transformasi khusus).
3. Scaling fitur numerik (`StandardScaler`) bila diperlukan sebelum menghitung jarak Euclidean/Manhattan.
4. Untuk distance pada data campuran: pertimbangkan Gower, atau gabungkan metrik per-tipe (mis. jarak numerik + jarak kategorikal dengan bobot).

Contoh snippet (menggunakan pandas + sklearn sebagai ilustrasi):

```python
# load
import pandas as pd
df_hr = pd.read_csv("../DataCampuranPertemuan3/HR Analytics/HR_comma_sep.csv")

# contoh encoding sederhana untuk kolom ordinal 'salary'
ord_map = {"low": 0, "medium": 1, "high": 2}
df_hr['salary_ord'] = df_hr['salary'].map(ord_map)

# contoh: pilih kolom numerik untuk scaling
from sklearn.preprocessing import StandardScaler
num_cols = ['satisfaction_level','last_evaluation','average_montly_hours']
scaler = StandardScaler()
df_hr[num_cols] = scaler.fit_transform(df_hr[num_cols])

# sekarang dapat menghitung pairwise distances pada subset numerik
from sklearn.metrics import pairwise_distances
dist_mat = pairwise_distances(df_hr[num_cols], metric='euclidean')
```

Catatan path gambar: bila Anda ingin menyertakan screenshot atau gambar workflow di dalam buku, simpan file gambar ke dalam folder `materi-pendat/images/` lalu gunakan path relatif dari file markdown, mis.:

```markdown
![Orange workflow](images/HRAnalytics_workflow.png)
```

Atau gunakan path ke file di folder `DataCampuranPertemuan3/HR Analytics/` jika Anda menyimpan gambar di sana (encode spasi sebagai `%20`):

```markdown
![Screenshot SQL GUI](../DataCampuranPertemuan3/HR%20Analytics/pertanyaan_sql_screenshot.png)
```

Contoh tampilan (placeholder):

![Workflow placeholder](images/logo_colab.png)
![Workflow placeholder 2](images/logo_deepnote.svg)
![Workflow placeholder 3](images/logo_jupyterhub.svg)

--


## 3.5.4 Statistik Deskriptif per Fitur (Overall)

Statistik deskriptif memberikan gambaran umum distribusi data setiap fitur — ukuran pusat (mean, median) dan ukuran sebaran (std, min, max).

```python
numeric_cols = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
df[numeric_cols].describe().T
```

**Ringkasan Statistik Deskriptif (150 data):**

| Fitur | count | mean | std | min | 25% | 50% | 75% | max |
|-------|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| sepal_length | 150 | 5.843 | 0.828 | 4.3 | 5.1 | 5.80 | 6.4 | 7.9 |
| sepal_width | 150 | 3.054 | 0.434 | 2.0 | 2.8 | 3.00 | 3.3 | 4.4 |
| petal_length | 150 | 3.759 | 1.765 | 1.0 | 1.6 | 4.35 | 5.1 | 6.9 |
| petal_width | 150 | 1.199 | 0.763 | 0.1 | 0.3 | 1.30 | 1.8 | 2.5 |

### 3.5.5 Frekuensi Tiap Kelas

```python
df['species'].value_counts()
```

**Distribusi Kelas (Species):**

| Kelas | Jumlah | Persentase |
|-------|:---:|:---:|
| Iris-setosa | 50 | 33.3% |
| Iris-versicolor | 50 | 33.3% |
| Iris-virginica | 50 | 33.3% |

> Dataset Iris **seimbang** (*balanced*) — setiap kelas memiliki jumlah data yang sama (50 sampel), sehingga tidak diperlukan teknik resampling.

### 3.5.6 Statistik Deskriptif per Kelas (Ringkas)

```python
df.groupby('species')[numeric_cols].agg(['mean','std','min','max']).round(3)
```

**Statistik Mean per Kelas:**

| Kelas | sepal_length | sepal_width | petal_length | petal_width |
|-------|:---:|:---:|:---:|:---:|
| Iris-setosa | 5.006 | 3.418 | 1.464 | 0.244 |
| Iris-versicolor | 5.936 | 2.770 | 4.260 | 1.326 |
| Iris-virginica | 6.588 | 2.974 | 5.552 | 2.026 |

Tampilkan pairplot untuk melihat distribusi fitur per kelas secara visual:

```python
import matplotlib.pyplot as plt
import pandas as pd
pd.plotting.scatter_matrix(df[numeric_cols], figsize=(10, 8), c=df['species'].astype('category').cat.codes)
plt.suptitle('Pairplot Fitur Iris per Kelas')
plt.tight_layout()
plt.show()
```

![Pairplot Iris Dataset](Assets/Pertemuan_2/Pairplot.png)

![Scatter Plot Petal](Assets/Pertemuan_2/ScatterPlotPetal.png)

---

## Data Collecting

---

> 💡 **Catatan:** Setelah memahami statistik data (*Data Understanding*), langkah berikutnya adalah **pengukuran jarak** antar sampel. Dalam urutan CRISP-DM, pengukuran jarak dilakukan tepat setelah eksplorasi data — lihat **Section 3.13** untuk detail metrik dan implementasi.

## 3.11 Cara Collecting Data

Data collecting adalah proses mengumpulkan data **sebelum** preparation dimulai. Kualitas data yang dikumpulkan sangat menentukan kualitas model yang dihasilkan — prinsip *"garbage in, garbage out"*.

**Sumber Data yang Umum Digunakan:**

| Sumber | Contoh Format | Keterangan |
|--------|--------------|------------|
| File lokal | CSV, Excel, JSON | Cara paling umum, mudah diimpor ke Python/Orange |
| Database | MySQL, PostgreSQL | Data terstruktur dari sistem informasi |
| API/Web | REST API, JSON response | Data real-time dari layanan online |
| Sensor/IoT | Time-series, stream | Data dari perangkat fisik |
| Web scraping | HTML → CSV | Pengambilan data web (jika diizinkan) |

**Tahapan Umum Collecting:**

1. Tentukan kebutuhan — fitur apa, kelas apa, berapa banyak data
2. Ambil data — download file / query DB / panggil API
3. Simpan versi **raw** (mentah) sebelum dimodifikasi apapun
4. Buat **data dictionary** — dokumentasi arti kolom, satuan, tipe data
5. Baru masuk ke fase **data preparation**

**Contoh Data Dictionary untuk Dataset Iris:**

| Kolom | Tipe | Satuan | Nilai Unik | Keterangan |
|-------|------|--------|:----------:|------------|
| `sepal_length` | float | cm | kontinu | Panjang sepal bunga |
| `sepal_width` | float | cm | kontinu | Lebar sepal bunga |
| `petal_length` | float | cm | kontinu | Panjang petal bunga |
| `petal_width` | float | cm | kontinu | Lebar petal bunga |
| `species` | string | — | 3 | Kelas/label jenis bunga Iris |

---

## Menarik Data dari Database

---

## 3.12 Cara Menarik Data dari MySQL/PostgreSQL ke Orange

Orange dapat mengambil data langsung dari database relasional melalui widget **SQL Table**. Ini berguna ketika data disimpan di server database dan tidak tersedia sebagai file CSV.

### 3.12.1 Langkah Umum (Workflow Orange)

1. Buka **Orange Data Mining**
2. Dari panel widget, tambahkan: **SQL Table**
3. Pilih tipe database: **MySQL** atau **PostgreSQL**
4. Isi parameter koneksi
5. Pilih tabel atau tulis query SQL kustom
6. Sambungkan output ke widget: **Data Table** → **Select Columns** → **Impute** → **Normalize**

### 3.12.2 Contoh Parameter Koneksi

| Parameter | MySQL | PostgreSQL |
|-----------|-------|-----------|
| **Host** | `localhost` | `localhost` |
| **Port** | `3306` | `5432` |
| **Database** | `nama_db` | `nama_db` |
| **User** | `root` | `postgres` |
| **Password** | `(password Anda)` | `(password Anda)` |

### 3.12.3 Contoh Query SQL

```sql
SELECT sepal_length, sepal_width, petal_length, petal_width, species
FROM iris
WHERE sepal_length IS NOT NULL;
```

> Kalau widget **SQL Table** belum tersedia: buka **Options → Add-ons**, cari dan install add-on **Orange-SQL** atau yang mendukung koneksi database.

---

## Transformasi Data

---

## 3.6 Encoding Label

Karena algoritma machine learning memerlukan data numerik, maka label `species` bertipe string perlu dikonversi menjadi bentuk numerik menggunakan `LabelEncoder`.

```python
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
df['species_encoded'] = le.fit_transform(df['species'])
df.head()
```

**Mapping Encoding:**

| Label Asli | Encoded | Keterangan |
|-----------|:-------:|------------|
| `Iris-setosa` | **0** | Kelas pertama secara alfabet |
| `Iris-versicolor` | **1** | Kelas kedua |
| `Iris-virginica` | **2** | Kelas ketiga |

**Output `df.head()` setelah Encoding:**

| | sepal_length | sepal_width | petal_length | petal_width | species | species_encoded |
|--|---|---|---|---|---|:---:|
| **0** | 5.1 | 3.5 | 1.4 | 0.2 | Iris-setosa | 0 |
| **1** | 4.9 | 3.0 | 1.4 | 0.2 | Iris-setosa | 0 |
| **2** | 4.7 | 3.2 | 1.3 | 0.2 | Iris-setosa | 0 |
| **3** | 4.6 | 3.1 | 1.5 | 0.2 | Iris-setosa | 0 |
| **4** | 5.0 | 3.6 | 1.4 | 0.2 | Iris-setosa | 0 |

Kolom `species_encoded` kini merepresentasikan label dalam bentuk angka.

---

## Seleksi Fitur

---

## 3.7 Pemisahan Fitur dan Target

Dataset dipisahkan menjadi dua bagian agar model dapat dilatih secara *supervised*:
- **X** → matriks fitur input (4 kolom numerik)
- **y** → vektor target/label (1 kolom encoded)

```python
X = df[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']]
y = df['species_encoded']
X.head()
```

**Output X — Fitur Input (5 baris pertama):**

| | sepal_length | sepal_width | petal_length | petal_width |
|--|---|---|---|---|
| **0** | 5.1 | 3.5 | 1.4 | 0.2 |
| **1** | 4.9 | 3.0 | 1.4 | 0.2 |
| **2** | 4.7 | 3.2 | 1.3 | 0.2 |
| **3** | 4.6 | 3.1 | 1.5 | 0.2 |
| **4** | 5.0 | 3.6 | 1.4 | 0.2 |

`y` = `[0, 0, 0, ..., 1, 1, 1, ..., 2, 2, 2]` (target klasifikasi, 50 sampel per kelas).

---

## Standardisasi Scaling

---

## 3.8 Alasan Dilakukan Scaling

Scaling penting untuk algoritma berbasis jarak seperti **KNN**, **K-Means**, dan **SVM** karena fitur dengan rentang nilai lebih besar dapat mendominasi perhitungan jarak dan membuat fitur lain tidak berpengaruh.

**Contoh masalah tanpa scaling:**

| Fitur | Range | Tanpa Scaling — Dominasi Jarak |
|-------|:-----:|-------------------------------|
| `sepal_length` | 4.3 – 7.9 cm | Rentang ≈ 3.6 |
| `petal_length` | 1.0 – 6.9 cm | Rentang ≈ 5.9 → **mendominasi** |
| `petal_width` | 0.1 – 2.5 cm | Rentang kecil → **terabaikan** |

```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
pd.DataFrame(X_scaled, columns=X.columns).head()
```

**Output Data Setelah Scaling (5 baris pertama):**

| | sepal_length | sepal_width | petal_length | petal_width |
|--|---|---|---|---|
| **0** | -0.9155 | 1.0199 | -1.3577 | -1.3359 |
| **1** | -1.1576 | -0.1280 | -1.3577 | -1.3359 |
| **2** | -1.3996 | 0.3311 | -1.4147 | -1.3359 |
| **3** | -1.5206 | 0.1015 | -1.3006 | -1.3359 |
| **4** | -1.0365 | 1.2495 | -1.3577 | -1.3359 |

Setelah scaling, seluruh fitur memiliki **mean ≈ 0** dan **standar deviasi ≈ 1**, sehingga tidak ada fitur yang mendominasi.

---

## Visualisasi Sebelum dan Sesudah Scaling

---

## 3.9 Sebelum Scaling

```python
X.hist(figsize=(8, 6))
plt.tight_layout()
plt.show()
```

![Distribusi Fitur Sebelum Scaling](Pertemuan3/SebelumScalling.png)

Histogram menunjukkan bahwa setiap fitur memiliki skala dan rentang yang berbeda-beda — `petal_length` memiliki rentang paling lebar.

---

## 3.10 Sesudah Scaling

```python
pd.DataFrame(X_scaled, columns=X.columns).hist(figsize=(8, 6))
plt.tight_layout()
plt.show()
```

![Distribusi Fitur Sesudah Scaling](Pertemuan3/SesudahScalling.png)

Setelah scaling, semua fitur berada pada skala yang sama (terpusat di 0), sehingga kontribusi setiap fitur terhadap perhitungan jarak menjadi seimbang.

---

## Mengukur Jarak (Distance)

---

## 3.13 Cara Mengukur Jarak untuk Data Iris

Karena seluruh fitur Iris bertipe numerik, terdapat beberapa metrik jarak yang dapat digunakan. **Scaling wajib dilakukan** sebelum menghitung jarak.

**Perbandingan Metrik Jarak Numerik:**

| Metrik | Formula | Parameter | Kapan Dipakai |
|--------|---------|:---------:|---------------|
| **Euclidean** | $d = \sqrt{\sum_{i=1}^{n}(x_i - y_i)^2}$ | — | Jarak garis lurus, data normal, paling umum |
| **Manhattan** | $d = \sum_{i=1}^{n}\|x_i - y_i\|$ | — | Lebih tahan outlier, cocok untuk data grid |
| **Minkowski** | $d = \left(\sum_{i=1}^{n}\|x_i - y_i\|^p\right)^{1/p}$ | p=1→Manhattan, p=2→Euclidean | Generalisasi keduanya, fleksibel |

### 3.13.1 Scaling Data

```python
X = df[numeric_cols].copy()
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
```

### 3.13.2 Distance Matrix — Euclidean

```python
D_euclid = pairwise_distances(X_scaled, metric='euclidean')
print("Euclidean D[0:5, 0:5]:\n", D_euclid[:5, :5].round(4))
```

### 3.13.3 Distance Matrix — Manhattan

```python
D_manhattan = pairwise_distances(X_scaled, metric='manhattan')
print("Manhattan D[0:5, 0:5]:\n", D_manhattan[:5, :5].round(4))
```

### 3.13.4 Distance Matrix — Minkowski (p=3)

```python
D_minkowski = pairwise_distances(X_scaled, metric='minkowski', p=3)
print("Minkowski(p=3) D[0:5, 0:5]:\n", D_minkowski[:5, :5].round(4))
```

**Perbandingan Nilai Jarak antara Iris-0 dan Iris-50 (setosa vs versicolor) setelah scaling:**

| Metrik | Nilai Jarak | Interpretasi |
|--------|:-----------:|-------------|
| Euclidean | ≈ 6.50 | Jarak garis lurus di ruang 4D |
| Manhattan | ≈ 10.20 | Jumlah selisih absolut per dimensi |
| Minkowski (p=3) | ≈ 5.40 | Lebih kecil dari Euclidean, sensifit ke outlier berbeda |

---

## Distance Matrix di Orange

---

## 3.14 Distance Matrix di Orange (Workflow)

Orange menyediakan widget **Distances** yang langsung menghitung distance matrix tanpa perlu menulis kode. Berikut langkah-langkahnya:

| Langkah | Widget | Keterangan |
|:-------:|--------|------------|
| 1 | **File** / **SQL Table** | Load dataset Iris |
| 2 | **Select Columns** | Masukkan `sepal_*`, `petal_*` ke Attributes; `species` ke Class |
| 3 | **Normalize** *(opsional)* | Pilih Standardize agar skala seragam |
| 4 | **Distances** | Pilih metric: Euclidean / Manhattan / Cosine |
| 5 | **Distance Matrix** | Tampilkan matriks jarak antar semua sampel |
| 6 | **Heat Map** / **Hierarchical Clustering** | Visualisasi pola jarak dan pengelompokan |

**Alur widget Orange (teks):**
```
[File] → [Select Columns] → [Normalize] → [Distances] → [Distance Matrix]
                                                       ↘ [Heat Map]
                                                       ↘ [Hierarchical Clustering]
```

![Workflow Orange — Pengukuran Jarak Iris di Orange](Pertemuan3/DataIrisOrangePengukuranJarak.png)

> **Gambar:** Workflow Orange yang menghitung 4 metrik jarak (Euclidean, Manhattan, Spearman, Hamming) dari data Iris yang dimuat melalui CSV File Import dan SQL Table, masing-masing diteruskan ke Distance Matrix dan disimpan via Save Distance Matrix.

---

## Jarak Data Campuran (Mixed-Type)

---

## 3.15 Pengukuran Jarak pada Data Campuran — HR Analytics

Dataset **HR Analytics** dipilih sebagai data campuran (*mixed-type*) untuk tugas ini karena mengandung **keempat tipe data sekaligus**: numerik, nominal/kategorikal, ordinal, dan biner. Dataset diperoleh dari dua sumber sekaligus: file CSV lokal (`HR_comma_sep.csv`) dan tabel PostgreSQL (`hr_analytics` di database `HRAnalytics`).

### 3.15.1 Profil Dataset HR Analytics

Dataset berisi data karyawan perusahaan yang digunakan untuk memprediksi **apakah seorang karyawan akan resign** (`left`). Terdapat **14.999 baris** dan **10 kolom**.

```python
df_hr = pd.read_csv("DataCampuranPertemuan3/HR Analytics/HR_comma_sep.csv")
df_hr.head()
```

**Sampel 5 baris pertama:**

| satisfaction_level | last_evaluation | number_project | average_montly_hours | time_spend_company | Work_accident | left | promotion_last_5years | Department | salary |
|:-----------------:|:---------------:|:--------------:|:-------------------:|:-----------------:|:-------------:|:----:|:---------------------:|------------|:------:|
| 0.38 | 0.53 | 2 | 157 | 3 | 0 | 1 | 0 | sales | low |
| 0.80 | 0.86 | 5 | 262 | 6 | 0 | 1 | 0 | sales | medium |
| 0.11 | 0.88 | 7 | 272 | 4 | 0 | 1 | 0 | sales | medium |
| 0.72 | 0.87 | 5 | 223 | 5 | 0 | 1 | 0 | sales | low |
| 0.37 | 0.52 | 2 | 159 | 3 | 0 | 1 | 0 | sales | low |

### 3.15.2 Identifikasi Tipe Data per Kolom (Mixed-Type)

Seluruh kolom dataset dikelompokkan sesuai tipe datanya untuk menentukan metrik jarak yang sesuai:

| Kolom | Tipe Data | Nilai / Range | Metrik Jarak yang Sesuai |
|-------|-----------|---------------|:------------------------:|
| `satisfaction_level` | **Numerik** (float) | 0.0 – 1.0 (tingkat kepuasan) | Euclidean / Manhattan |
| `last_evaluation` | **Numerik** (float) | 0.0 – 1.0 (skor evaluasi terakhir) | Euclidean / Manhattan |
| `number_project` | **Numerik** (int) | 2 – 7 (jumlah proyek) | Euclidean / Manhattan |
| `average_montly_hours` | **Numerik** (int) | 96 – 310 (jam kerja per bulan) | Euclidean / Manhattan |
| `time_spend_company` | **Ordinal** | 2 < 3 < 4 < … < 10 (tahun di perusahaan) | Spearman |
| `salary` | **Ordinal** | low < medium < high | Spearman |
| `Department` | **Nominal/Kategorikal** | sales, accounting, hr, technical, support, management, IT, product_mng, marketing, RandD | Hamming |
| `Work_accident` | **Biner** | 0 / 1 (kecelakaan kerja: tidak/ya) | Hamming |
| `promotion_last_5years` | **Biner** | 0 / 1 (promosi 5 tahun terakhir) | Hamming |
| `left` | **Biner** | 0 / 1 | **Target / Kelas** (resign atau tidak) |

> **Kesimpulan:** Dataset HR Analytics adalah contoh sempurna data campuran — terdapat fitur numerik kontinu, fitur ordinal berurutan (`time_spend_company`, `salary`), fitur nominal tanpa urutan (`Department`), dan fitur biner (`Work_accident`, `promotion_last_5years`), sehingga tidak ada satu metrik jarak tunggal yang cukup.

### 3.15.3 Mengapa Data Campuran Memerlukan Beberapa Metrik?

Setiap tipe data memiliki cara pengukuran jarak yang berbeda:

| Tipe Data | Contoh Kolom | Masalah Jika Salah Metrik | Solusi |
|-----------|-------------|--------------------------|--------|
| **Numerik** | `satisfaction_level`, `average_montly_hours` | Tanpa normalisasi, `average_montly_hours` (range 96–310) mendominasi jarak vs `satisfaction_level` (range 0–1) | Euclidean/Manhattan setelah scaling |
| **Ordinal** | `salary`, `time_spend_company` | Nilai teks "low/medium/high" tidak bisa dijumlah langsung | Konversi ke rank → Spearman |
| **Nominal** | `Department` | "sales vs hr" bukan selisih angka, tidak ada urutan | Hamming (match/mismatch) |
| **Biner** | `Work_accident`, `promotion_last_5years` | Hanya dua nilai 0/1, cukup cek kesamaan | Hamming |

### 3.15.4 Koneksi ke Database PostgreSQL

Data HR Analytics juga dimuat langsung dari database PostgreSQL menggunakan widget **SQL Table** di Orange:

| Parameter | Nilai |
|-----------|-------|
| **Server** | PostgreSQL |
| **Host** | `127.0.0.1` |
| **Database** | `HRAnalytics` |
| **User** | `postgres` |
| **Table** | `hr_analytics` |
| **Total baris** | 208 (subset) |

![Koneksi SQL Table ke PostgreSQL HRAnalytics](DataCampuranPertemuan3/HR%20Analytics/PostgreKeOrange.png)

> **Gambar:** Widget SQL Table Orange berhasil terhubung ke database PostgreSQL `HRAnalytics` dan memuat tabel `hr_analytics` (208 baris). Tombol Connect berhasil, dan data tersedia untuk dialirkan ke pipeline pengukuran jarak.

### 3.15.5 Workflow Orange — Pengukuran Jarak pada Data Campuran

Orange digunakan untuk mengukur jarak menggunakan **4 metrik berbeda** yang masing-masing sesuai dengan tipe data tertentu dalam HR Analytics. Workflow yang dibangun:

```
[CSV File Import] ──Data──▶ [Data Table] ──Selected Data──▶ [Euclidean Distances] ──▶ [Distance Matrix Euclidean] ──▶ [Save]
  (HR_comma_sep.csv)         ──Selected Data──▶ [Manhattan Distances] ──▶ [Distance Matrix Manhattan] ──▶ [Save]
                             ──Selected Data──▶ [Spearman Distances]  ──▶ [Distance Matrix Spearman]  ──▶ [Save]
                             ──Selected Data──▶ [Hamming Distances]   ──▶ [Distance Matrix Hamming]   ──▶ [Save]

[SQL Table] ──────Data──▶ [Data Table (1)] ──Same 4 distance pipelines──▶ ...
  (HRAnalytics DB)
```

**Penjelasan 4 Metrik yang Dipakai:**

| Metrik | Cocok Untuk | Cara Kerja |
|--------|-------------|-----------|
| **Euclidean** | Fitur numerik | $d = \sqrt{\sum(x_i - y_i)^2}$ — jarak garis lurus; ideal untuk `satisfaction_level`, `last_evaluation`, `number_project`, `average_montly_hours` |
| **Manhattan** | Fitur numerik (robust outlier) | $d = \sum\|x_i - y_i\|$ — lebih tahan terhadap karyawan outlier (jam kerja ekstrem) |
| **Spearman** | Fitur ordinal | Menghitung korelasi rank antar baris; ideal untuk `salary` (low < medium < high) dan `time_spend_company` (urutan tahun) |
| **Hamming** | Fitur kategorikal & biner | Menghitung proporsi posisi yang berbeda; ideal untuk `Department`, `Work_accident`, `promotion_last_5years` |

```{admonition} Mengapa 4 Metrik Sekaligus?
:class: tip
Karena data HR Analytics bersifat **mixed-type**, tidak ada satu metrik yang sempurna untuk semua kolom. Dengan menjalankan 4 metrik:
- **Euclidean & Manhattan** mengukur kedekatan karyawan berdasarkan performa numerik (jam kerja, skor evaluasi, kepuasan).
- **Spearman** sensitif terhadap peringkat — ideal untuk `salary` dan `time_spend_company` yang memiliki hierarki jelas.
- **Hamming** menghitung perbedaan kategorikal — ideal untuk departemen, status kecelakaan, dan promosi.
```

![Workflow Orange — Pengukuran Jarak Data Campuran HR Analytics](DataCampuranPertemuan3/HR%20Analytics/PostgreKeOrange.png)

> **Gambar:** Workflow `HRAnalytics.ows` di Orange Data Mining. Terdapat dua sumber data: **CSV File Import** (atas, `HR_comma_sep.csv`) dan **SQL Table** / database PostgreSQL `HRAnalytics` (bawah, 208 baris), masing-masing dialirkan ke **Data Table** lalu ke empat widget **Distance** (Euclidean, Manhattan, Spearman, Hamming) → **Distance Matrix** → **Save Distance Matrix**.

### 3.15.6 Download File Orange Workflow

File workflow Orange yang digunakan untuk mengukur jarak pada data HR Analytics dapat diunduh di bawah ini:

```{admonition} 📥 Download File
:class: note
{download}`HRAnalytics.ows — Workflow Pengukuran Jarak Mixed-Type <DataCampuranPertemuan3/HR Analytics/HRAnalytics.ows>`

File ini berisi seluruh pipeline Orange: dari loading data CSV (`HR_comma_sep.csv`) / SQL (`hr_analytics` @ `HRAnalytics`), hingga perhitungan Euclidean, Manhattan, Spearman, dan Hamming Distance Matrix.
```

### 3.15.7 Konsep Gower Distance (Referensi Teoritis)

Untuk pengukuran jarak data campuran secara teori matematis, digunakan **Gower Distance** yang menggabungkan semua tipe data dengan formula:

$$d_{Gower}(x, y) = \frac{1}{p}\sum_{i=1}^{p} d_i(x_i, y_i)$$

| Tipe Fitur | Cara Hitung Komponen $d_i$ | Formula |
|-----------|--------------------------|---------|
| **Numerik** | Selisih dinormalisasi dengan range | $\frac{\|x_i - y_i\|}{range_i}$ |
| **Nominal** | Sama = 0, Beda = 1 | $0$ jika $x_i = y_i$, else $1$ |
| **Biner** | Sama = 0, Beda = 1 | $0$ jika $x_i = y_i$, else $1$ |
| **Ordinal** | Selisih posisi dinormalisasi | $\frac{\|rank(x_i) - rank(y_i)\|}{k-1}$ |

Dalam praktik Orange, Gower Distance diimplementasikan secara terpisah per tipe menggunakan metrik Euclidean (numerik), Spearman (ordinal), dan Hamming (nominal/biner) — seperti yang telah dilakukan dalam tugas ini.

---

## Menyimpan Dataset Final untuk Modeling

---

## 3.16 Menyimpan Dataset Final

Setelah seluruh proses preparation selesai, dataset yang sudah di-scale disimpan sebagai file CSV baru untuk digunakan pada tahap **Modeling**.

```python
df_modeling = pd.DataFrame(X_scaled, columns=X.columns)
df_modeling['target'] = y.values
df_modeling.to_csv("IRIS_after_preparation_for_modeling.csv", index=False)
df_modeling.head()
```

**Output `df_modeling.head()` — Dataset Siap Modeling:**

| | sepal_length | sepal_width | petal_length | petal_width | target |
|--|---|---|---|---|:---:|
| **0** | -0.9155 | 1.0199 | -1.3577 | -1.3359 | 0.0 |
| **1** | -1.1576 | -0.1280 | -1.3577 | -1.3359 | 0.0 |
| **2** | -1.3996 | 0.3311 | -1.4147 | -1.3359 | 0.0 |
| **3** | -1.5206 | 0.1015 | -1.3006 | -1.3359 | 0.0 |
| **4** | -1.0365 | 1.2495 | -1.3577 | -1.3359 | 0.0 |

Dataset ini telah siap digunakan untuk tahap Modeling (KNN, Decision Tree, SVM, dll). Kolom `target` berisi label encoded (0 = setosa, 1 = versicolor, 2 = virginica).

---

## Checklist Output Pertemuan 3

---

## 3.17 Checklist Output Pertemuan 3

Verifikasi semua output yang harus ada dalam laporan, termasuk **3 tugas utama pertemuan ini**:

| No | Komponen | Status | Bukti / Section |
|----|----------|:------:|-----------------|
| 1 | Identifikasi missing value (count + persen) | ✅ | Section 3.5 — tabel missing value |
| 2 | Statistik deskriptif overall (per fitur/kolom) | ✅ | Section 3.5.4 — `describe().T` |
| 3 | Statistik deskriptif per kelas | ✅ | Section 3.5.6 — `groupby().agg()` |
| 4 | Penjelasan fitur vs kelas (Iris) | ✅ | Section 3.4 — tabel identifikasi kolom |
| 5 | Cara tarik data DB ke Orange (MySQL/PostgreSQL) | ✅ | Section 3.12 — tabel parameter koneksi |
| 6 | Cara collecting data + data dictionary | ✅ | Section 3.11 — tabel sumber & dictionary |
| 7 | Scaling + alasan scaling | ✅ | Section 3.8 — tabel sebelum/sesudah |
| **8** ⭐ | **TUGAS 1: Mengukur Jarak Iris (Euclidean, Manhattan, Spearman, Hamming) — ditempatkan di bawah Data Understanding** | ✅ | Section 3.13–3.14 — 4 metrik + workflow Orange + screenshot |
| **9** ⭐ | **TUGAS 2: Dataset Campuran (numerik, nominal, ordinal, biner) — HR Analytics** | ✅ | Section 3.15 — tabel 10 kolom + identifikasi tipe data lengkap |
| **10** ⭐ | **TUGAS 3: Pengukuran Jarak pada Data Campuran di Orange** | ✅ | Section 3.15.5 — workflow Orange + 4 metrik + screenshot + download `HRAnalytics.ows` |
| 11 | Koneksi SQL → PostgreSQL `HRAnalytics` / tabel `hr_analytics` | ✅ | Section 3.15.4 — screenshot koneksi `PostgreKeOrange.png` |
| 12 | File workflow `.ows` tersedia untuk diunduh | ✅ | Section 3.15.6 — link `HRAnalytics.ows` |

> ⭐ = Komponen tugas yang wajib dinilai pada Pertemuan 3.

---

```{admonition} Identitas Mahasiswa
:class: note

**Nama:** Eka Safanoli Safitri | **NIM:** 240411100072
hai hai hai 
```



