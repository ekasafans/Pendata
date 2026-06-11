# Penambangan Data - A
## Repository ini dibuat untuk Mata Kuliah Penambangan Data Kelas A di Semester 4 Program Studi Teknik Informatika
---
### Nama : Eka Safanoli Safitri
### Nim : 240411100072

## Panduan Penggunaan Jupyter Book

### 1. Persiapan Lingkungan (Virtual Environment)
Pastikan Anda sudah memiliki dan mengaktifkan virtual environment sebelum menjalankan perintah-perintah di bawah ini.

**Cara Mengaktifkan (PowerShell):**
```powershell
# Jika Anda berada di folder Pendata:
..\.venv\Scripts\Activate.ps1

# Jika terminal menolak karena kebijakan keamanan, jalankan command ini dulu:
# Set-ExecutionPolicy -ExecutionPolicy Bypass -Scope Process -Force
```

### 2. Instalasi Jupyter Book (Versi Klasik/v1)
Untuk mengikuti tutorial yang menggunakan perintah `create` (standar lama), kita perlu menggunakan Jupyter Book versi klasik (di bawah versi 2.0). Versi terbaru (2.x) menggunakan perintah `init` yang berbeda.

```bash
pip install "jupyter-book<2.0.0"
```

### 3. Membuat Struktur Buku Baru
Untuk membuat kerangka buku baru (misalnya folder `materi-pendat`):

```bash
jupyter-book create materi-pendat
```

### 4. Membangun Buku (Build)
Setelah membuat konten, bangun buku menjadi format HTML agar bisa dilihat di browser.

```bash
jupyter-book build materi-pendat
```
Hasil build akan berada di folder `materi-pendat/_build/html/index.html`.

### 5. Panduan Git (Push ke GitHub)
Berikut adalah langkah-langkah untuk menyimpan perubahan ke repository GitHub.

**Cek Status Perubahan:**
Melihat file mana saja yang berubah.
```bash
git status
```

**Menambahkan Semua Perubahan:**
Menyiapkan semua file yang berubah untuk disimpan.
```bash
git add .
```

**Membuat Commit (Menyimpan Perubahan):**
Memberikan pesan catatan tentang apa yang Anda ubah.
```bash
git commit -m "Deskripsi perubahan Anda di sini (misal: update materi bab 1)"
```

**Mengirim ke GitHub (Push):**
Mengirim commit lokal Anda ke server GitHub (remote origin).
```bash
git push origin main
# Catatan: Sesuaikan 'main' dengan nama branch utama Anda (bisa 'master' atau 'main')
```
