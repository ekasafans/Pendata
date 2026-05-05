# Jupyter Book (v1) — Build & Deploy Instructions

Ringkasan langkah untuk membuat environment, membangun Jupyter Book (versi klasik v1.x), dan deploy ke GitHub Pages.

1. Aktifkan virtual environment (PowerShell):

   ..\\.venv\\Scripts\\Activate.ps1

   Jika kebijakan menolak, jalankan (PowerShell admin session jika perlu):

   Set-ExecutionPolicy -ExecutionPolicy Bypass -Scope Process -Force

2. Membuat environment (conda, optional):

   conda env create -f environment.yml
   conda activate pendata-jb

3. Atau gunakan venv (Linux/macOS):

   ./build_and_deploy.sh

   Untuk langsung melakukan deploy setelah build gunakan:

   DEPLOY_GHP=1 ./build_and_deploy.sh

4. Cara manual (setelah install):

   pip install -r requirements-jb.txt
   jupyter-book build materi-pendat

   Untuk deploy (lokal) menggunakan ghp-import:

   ghp-import -n -p -f materi-pendat/_build/html

5. Git & Push ke GitHub (simpan perubahan sumber sebelum deploy otomatis):

   git add .
   git commit -m "Build: update materi-pendat"
   git push origin main

Catatan:
- Workflow GitHub Actions otomatis: file `.github/workflows/jupyter-book-gh-pages.yml` akan membangun buku dan meng-deploy ke branch `gh-pages` pada setiap push ke `main`.
- Tidak ada file yang dihapus oleh perubahan ini; hanya file tambahan yang dibuat untuk membantu build dan deploy.
