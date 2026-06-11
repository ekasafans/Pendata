# Naive Bayes Classifier
### Pengantar Data Mining — Probabilistic Classification

---

## Apa itu Naive Bayes?

Naive Bayes adalah algoritma klasifikasi berbasis **probabilitas** yang menggunakan **Teorema Bayes** sebagai fondasinya. Disebut *"naive"* (naif) karena algoritma ini mengasumsikan bahwa setiap fitur/atribut **tidak saling mempengaruhi** satu sama lain — padahal di dunia nyata hal ini jarang benar-benar terjadi.

Meski terkesan sederhana, Naive Bayes terbukti sangat efektif untuk:

- Klasifikasi teks & spam filter
- Diagnosa medis
- Sistem rekomendasi
- Deteksi penipuan (fraud detection)

---

## Fondasi: Teorema Bayes

Seluruh mekanisme Naive Bayes bertumpu pada satu rumus:

$$\boxed{P(C \mid X) = \frac{P(X \mid C) \cdot P(C)}{P(X)}}$$

Artinya tiap komponen:

| Komponen | Nama | Penjelasan |
|----------|------|------------|
| $P(C \mid X)$ | **Posterior** | Probabilitas kelas C, setelah melihat data X |
| $P(X \mid C)$ | **Likelihood** | Seberapa mungkin data X muncul di kelas C |
| $P(C)$ | **Prior** | Probabilitas awal kelas C sebelum melihat data |
| $P(X)$ | **Evidence** | Probabilitas data X (konstan, bisa diabaikan) |

Karena $P(X)$ konstan untuk semua kelas, yang perlu dimaksimalkan hanyalah:

$$P(C_i \mid X) \propto P(X \mid C_i) \cdot P(C_i)$$

---

## Asumsi Conditional Independence

Di sinilah letak kata "naive" — semua atribut diasumsikan **bebas satu sama lain** given kelas C:

$$P(X \mid C_i) = P(x_1 \mid C_i) \times P(x_2 \mid C_i) \times \cdots \times P(x_n \mid C_i) = \prod_{k=1}^{n} P(x_k \mid C_i)$$

**Implikasinya:** kita tidak perlu menghitung joint probability semua kombinasi atribut — cukup hitung masing-masing secara terpisah lalu kalikan.

---

## Dua Jenis Naive Bayes Berdasarkan Tipe Atribut

### 1. Atribut Kategorikal → `CategoricalNB`

Hitung frekuensi kemunculan nilai $x_k$ di dalam kelas $C_i$:

$$P(x_k \mid C_i) = \frac{\text{count}(x_k \in C_i)}{|C_i|}$$

### 2. Atribut Kontinu → `GaussianNB`

Asumsikan distribusi Gaussian (normal) dengan mean $\mu$ dan std $\sigma$ dari data training:

$$P(x_k \mid C_i) = \frac{1}{\sqrt{2\pi}\,\sigma_{C_i}} \exp\!\left(-\frac{(x_k - \mu_{C_i})^2}{2\sigma_{C_i}^2}\right)$$

---

## Studi Kasus 1 — Dataset `buys_computer`

Dataset dari kuliah: 14 record, 4 atribut kategorikal, label biner.

```
age        income    student   credit_rating   buys_computer
---------------------------------------------------------------
<=30       high      no        fair            no
<=30       high      no        excellent       no
31..40     high      no        fair            yes
>40        medium    no        fair            yes
>40        low       yes       fair            yes
>40        low       yes       excellent       no
31..40     low       yes       excellent       yes
<=30       medium    no        fair            no
<=30       low       yes       fair            yes
>40        medium    yes       fair            yes
<=30       medium    yes       excellent       yes
31..40     medium    no        excellent       yes
31..40     high      yes       fair            yes
>40        medium    no        excellent       no
```

**Pertanyaan:** Apakah seseorang dengan profil berikut akan membeli komputer?
```
X = { age=<=30, income=medium, student=yes, credit_rating=fair }
```

### Langkah 1 — Hitung Prior

```
Total data (D) = 14

yes → 9 record    P(yes) = 9/14 = 0.643
no  → 5 record    P(no)  = 5/14 = 0.357
```

### Langkah 2 — Hitung Likelihood tiap Atribut

```
─── age = <=30 ───────────────────────────────
  Di kelas YES : 2 dari 9 record    → P(age=<=30 | yes) = 2/9 = 0.222
  Di kelas NO  : 3 dari 5 record    → P(age=<=30 | no)  = 3/5 = 0.600

─── income = medium ──────────────────────────
  Di kelas YES : 4 dari 9 record    → P(medium | yes) = 4/9 = 0.444
  Di kelas NO  : 2 dari 5 record    → P(medium | no)  = 2/5 = 0.400

─── student = yes ────────────────────────────
  Di kelas YES : 6 dari 9 record    → P(student=yes | yes) = 6/9 = 0.667
  Di kelas NO  : 1 dari 5 record    → P(student=yes | no)  = 1/5 = 0.200

─── credit_rating = fair ─────────────────────
  Di kelas YES : 6 dari 9 record    → P(fair | yes) = 6/9 = 0.667
  Di kelas NO  : 2 dari 5 record    → P(fair | no)  = 2/5 = 0.400
```

### Langkah 3 — Kalikan & Bandingkan

```
P(X | yes) × P(yes) = 0.222 × 0.444 × 0.667 × 0.667 × 0.643
                    = 0.044 × 0.643
                    = 0.028  ✅

P(X | no)  × P(no)  = 0.600 × 0.400 × 0.200 × 0.400 × 0.357
                    = 0.019 × 0.357
                    = 0.007
```

**Kesimpulan:** $0.028 > 0.007$ → **Prediksi: `buys_computer = YES`** 🎯

---

## Studi Kasus 2 — Weather/Play Dataset (Bayesian Belief Network)

Dataset ini digunakan untuk mendemonstrasikan **Bayesian Belief Network (BBN)** — versi Naive Bayes yang memperbolehkan sebagian variabel saling bergantung.

```
Struktur BBN:
         ┌────────┐
         │  play  │
         └───┬────┘
       ┌─────┼──────┐
       ▼     ▼      ▼
  [outlook] [temp] [windy]
       └─────┬──────┘
             ▼
         [humidity]
```

Node `outlook` → parent dari `windy` dan `humidity`, sehingga probabilitasnya **bersyarat pada outlook**, bukan bebas.

### Dengan Laplacian Correction (α = 1)

```
P(play=yes) = (9+1)/(14+2) = 10/16 = 0.625
P(play=no)  = (5+1)/(14+2) =  6/16 = 0.375
```

Tabel conditional outlook:

| play | sunny | overcast | rainy |
|------|-------|----------|-------|
| yes  | (2+1)/(9+3) = **0.250** | (4+1)/(9+3) = **0.417** | (3+1)/(9+3) = **0.333** |
| no   | (3+1)/(5+3) = **0.500** | (0+1)/(5+3) = **0.125** | (2+1)/(5+3) = **0.375** |

### Klasifikasi X = (Sunny, Cool, High humidity, Windy=True)

```
P(yes | X) = α × 0.625 × 0.25 × 0.4 × 0.2 × 0.5 = α × 0.00625
P(no  | X) = α × 0.375 × 0.5 × 0.167 × 0.333 × 0.4 = α × 0.00417

α = 1 / (0.00625 + 0.00417) = 95.969

P(play=yes | X) = 95.969 × 0.00625 = 0.60
P(play=no  | X) = 95.969 × 0.00417 = 0.40

→ PREDIKSI: play = YES (probabilitas 60%)
```

---

## Problem: Zero Probability & Solusinya

### Masalah

Bayangkan di training set **tidak ada satu pun** record dengan `income=low` di kelas tertentu:

```
P(income=low | C) = 0/N = 0
```

Akibatnya seluruh produk likelihood menjadi **nol** — walau atribut lain sangat mendukung kelas tersebut. Satu nol merusak segalanya.

### Solusi: Laplacian Correction

Tambahkan konstanta $\alpha$ (biasanya 1) ke semua count:

$$P(x_k \mid C_i) = \frac{\text{count}(x_k \in C_i) + \alpha}{|C_i| + \alpha \cdot V}$$

di mana $V$ = jumlah nilai unik atribut tersebut.

```
Contoh: 1000 record, 3 nilai income (low, medium, high)

                   Tanpa Laplace    Dengan Laplace (α=1)
income = low    →  0/1000 = 0.000   (0+1)/(1000+3) = 0.001  ✅
income = medium →  990/1000 = 0.990 (990+1)/(1000+3) = 0.987
income = high   →  10/1000 = 0.010  (10+1)/(1000+3) = 0.011
```

Di sklearn, parameter ini dikontrol dengan `alpha=1.0` pada `CategoricalNB`.

---

## Varian Naive Bayes di sklearn

| Kelas sklearn | Tipe Data | Kapan Digunakan |
|---------------|-----------|-----------------|
| `GaussianNB` | Kontinu | Fitur numerik (tinggi, berat, suhu) |
| `CategoricalNB` | Kategorikal | Fitur diskrit dengan kategori (low/med/high) |
| `MultinomialNB` | Count/frekuensi | Klasifikasi teks (bag-of-words, TF-IDF) |
| `BernoulliNB` | Biner (0/1) | Fitur biner (ada/tidak ada kata) |
| `ComplementNB` | Count | Imbalanced text classification |

---

## Kelebihan dan Keterbatasan

**Kelebihan:**
- Sangat cepat dilatih — kompleksitas $O(n \cdot d)$
- Efektif untuk dataset kecil maupun besar
- Bekerja baik meski asumsi independence dilanggar
- Robust terhadap fitur yang tidak relevan

**Keterbatasan:**
- Asumsi independence jarang terpenuhi di data nyata
- Tidak bisa menangkap interaksi antar fitur
- Estimasi probabilitas kurang akurat (tapi arah prediksi biasanya benar)
- Sensitif terhadap zero-frequency jika tanpa Laplace correction

---

## Implementasi Python — sklearn

> 🔗 Dokumentasi resmi: https://scikit-learn.org/stable/api/sklearn.naive_bayes.html

```python
"""
╔══════════════════════════════════════════════════════════════╗
║         PROYEK: KLASIFIKASI NAIVE BAYES                     ║
║         Menggunakan scikit-learn (sklearn)                   ║
║         Data Mining | May 2026                              ║
╚══════════════════════════════════════════════════════════════╝

Dataset yang digunakan:
  1. buys_computer dataset  (dari materi kuliah - manual)
  2. Iris dataset           (dari sklearn - sebagai bonus)

Referensi sklearn:
  https://scikit-learn.org/stable/api/sklearn.naive_bayes.html
"""

# ─────────────────────────────────────────────
# IMPORT LIBRARY
# ─────────────────────────────────────────────
import numpy as np
import pandas as pd
from sklearn.naive_bayes import GaussianNB, CategoricalNB
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score
)
from sklearn.preprocessing import LabelEncoder
from sklearn.datasets import load_iris
import warnings
warnings.filterwarnings('ignore')


# ══════════════════════════════════════════════
# BAGIAN 1: Dataset buys_computer (dari kuliah)
# ══════════════════════════════════════════════

def load_buys_computer_dataset():
    """
    Dataset dari slide kuliah (14 record).
    Atribut: age, income, student, credit_rating
    Label  : buys_computer (yes/no)
    """
    data = {
        'age':           ['<=30','<=30','31..40','>40','>40','>40','31..40',
                          '<=30','<=30','>40','<=30','31..40','31..40','>40'],
        'income':        ['high','high','high','medium','low','low','low',
                          'medium','low','medium','medium','medium','high','medium'],
        'student':       ['no','no','no','no','yes','yes','yes',
                          'no','yes','yes','yes','no','yes','no'],
        'credit_rating': ['fair','excellent','fair','fair','fair','excellent',
                          'excellent','fair','fair','fair','excellent','excellent',
                          'fair','excellent'],
        'buys_computer': ['no','no','yes','yes','yes','no','yes',
                          'no','yes','yes','yes','yes','yes','no']
    }
    return pd.DataFrame(data)


def encode_categorical(df, target_col='buys_computer'):
    """Label encode semua kolom kategorikal."""
    encoders = {}
    df_encoded = df.copy()
    for col in df.columns:
        le = LabelEncoder()
        df_encoded[col] = le.fit_transform(df[col])
        encoders[col] = le
    return df_encoded, encoders


def bagian1_buys_computer():
    print("=" * 60)
    print("📦 BAGIAN 1: Dataset buys_computer (Manual / Kuliah)")
    print("=" * 60)

    # Load dan tampilkan dataset
    df = load_buys_computer_dataset()
    print("\n📋 Dataset:")
    print(df.to_string(index=False))

    # Encode
    df_enc, encoders = encode_categorical(df)

    X = df_enc.drop('buys_computer', axis=1)
    y = df_enc['buys_computer']

    # ── Training dengan SEMUA data (karena kecil) ──
    model = CategoricalNB(alpha=1.0)   # alpha=1 = Laplacian Correction
    model.fit(X, y)

    # ── Prediksi sampel dari slide kuliah ──
    # X = (age<=30, income=medium, student=yes, credit_rating=fair)
    sample_raw = {
        'age': '<=30',
        'income': 'medium',
        'student': 'yes',
        'credit_rating': 'fair'
    }
    sample_encoded = []
    for col in X.columns:
        val = encoders[col].transform([sample_raw[col]])[0]
        sample_encoded.append(val)

    sample_arr = np.array([sample_encoded])
    pred = model.predict(sample_arr)
    pred_proba = model.predict_proba(sample_arr)
    label = encoders['buys_computer'].inverse_transform(pred)[0]
    classes = encoders['buys_computer'].classes_

    print(f"\n🔍 Sampel X dari kuliah: {sample_raw}")
    print(f"\n📊 Probabilitas:")
    for cls, prob in zip(classes, pred_proba[0]):
        marker = "← PREDIKSI" if cls == label else ""
        print(f"   P(buys_computer={cls}|X) = {prob:.4f}  {marker}")
    print(f"\n✅ Hasil Prediksi: buys_computer = '{label.upper()}'")

    # ── Accuracy dengan cross-validation (5-fold) ──
    scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')
    print(f"\n📈 Cross-Validation Accuracy (5-fold): {scores.mean():.4f} ± {scores.std():.4f}")
    print()


# ══════════════════════════════════════════════
# BAGIAN 2: Dataset Iris (sklearn built-in)
# ══════════════════════════════════════════════

def bagian2_iris():
    print("=" * 60)
    print("🌸 BAGIAN 2: Dataset Iris (GaussianNB - Atribut Kontinu)")
    print("=" * 60)

    # Load dataset
    iris = load_iris()
    X = pd.DataFrame(iris.data, columns=iris.feature_names)
    y = pd.Series(iris.target)
    class_names = iris.target_names

    print(f"\n📋 Info Dataset:")
    print(f"   Jumlah sampel  : {len(X)}")
    print(f"   Jumlah fitur   : {X.shape[1]}")
    print(f"   Kelas          : {list(class_names)}")
    print(f"\n📊 Statistik Fitur:")
    print(X.describe().round(3).to_string())

    # Split train/test (80/20)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"\n✂️  Train: {len(X_train)} sampel | Test: {len(X_test)} sampel")

    # Train GaussianNB (untuk atribut kontinu)
    model = GaussianNB()
    model.fit(X_train, y_train)

    # Evaluasi
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    print(f"\n🎯 Accuracy: {acc:.4f} ({acc*100:.2f}%)")
    print("\n📄 Classification Report:")
    print(classification_report(y_test, y_pred, target_names=class_names))

    print("🔲 Confusion Matrix:")
    cm = confusion_matrix(y_test, y_pred)
    cm_df = pd.DataFrame(cm, index=class_names, columns=class_names)
    print(cm_df.to_string())

    # Cross-validation
    cv_scores = cross_val_score(model, X, y, cv=10, scoring='accuracy')
    print(f"\n📈 Cross-Validation Accuracy (10-fold): {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

    # Prior probability per kelas
    print(f"\n📌 Prior Probability (P(Cᵢ)) dari training:")
    for i, cls in enumerate(class_names):
        prior = model.class_prior_[i]
        print(f"   P({cls}) = {prior:.4f}")

    # Prediksi satu sampel baru
    sample_new = np.array([[5.1, 3.5, 1.4, 0.2]])
    pred_class = model.predict(sample_new)[0]
    pred_proba = model.predict_proba(sample_new)[0]
    print(f"\n🔍 Prediksi sampel baru {list(sample_new[0])}:")
    for cls, prob in zip(class_names, pred_proba):
        marker = "← PREDIKSI" if class_names[pred_class] == cls else ""
        print(f"   P({cls}|X) = {prob:.6f}  {marker}")
    print()


# ══════════════════════════════════════════════
# BAGIAN 3: Penjelasan Laplacian Correction
# ══════════════════════════════════════════════

def bagian3_laplacian_demo():
    print("=" * 60)
    print("🔧 BAGIAN 3: Demo Laplacian Correction")
    print("=" * 60)

    print("""
Masalah:
  Jika satu nilai atribut tidak pernah muncul di training set
  untuk suatu kelas, maka P(xₖ|Cᵢ) = 0, dan seluruh
  P(X|Cᵢ) = 0 → klasifikasi rusak!

Solusi: Laplacian Correction (alpha=1 di CategoricalNB)
  Tambahkan 1 ke setiap count

Contoh (1000 record):
  income=low    : 0 record
  income=medium : 990 record
  income=high   : 10 record

Tanpa Laplace:
  P(income=low)    = 0/1000  = 0.000   ← MASALAH!

Dengan Laplace (tambah 1 per nilai, ada 3 nilai income):
  P(income=low)    = (0+1)/(1000+3) = 0.001
  P(income=medium) = (990+1)/(1000+3) = 0.987
  P(income=high)   = (10+1)/(1000+3) = 0.011
""")

    # Demo dengan CategoricalNB alpha
    print("Demo alpha=0 (tanpa Laplace) vs alpha=1 (dengan Laplace):")
    df = load_buys_computer_dataset()
    df_enc, encoders = encode_categorical(df)
    X = df_enc.drop('buys_computer', axis=1)
    y = df_enc['buys_computer']

    for alpha in [0.0, 0.5, 1.0]:
        model = CategoricalNB(alpha=alpha)
        scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')
        print(f"   alpha={alpha:.1f} → CV Accuracy: {scores.mean():.4f} ± {scores.std():.4f}")
    print()


# ══════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════

if __name__ == "__main__":
    print("""
╔══════════════════════════════════════════════════════════════╗
║   🧠 NAIVE BAYES CLASSIFIER — Python Sklearn Implementation  ║
╚══════════════════════════════════════════════════════════════╝
""")
    bagian1_buys_computer()
    bagian2_iris()
    bagian3_laplacian_demo()
    print("=" * 60)
    print("✅ Selesai! Semua klasifikasi berhasil dijalankan.")
    print("=" * 60)
```

---