#!/usr/bin/env bash
set -euo pipefail

# Simple helper to create a venv, install requirements, build the book,
# and optionally deploy to GitHub Pages using ghp-import.

VENV_DIR=".venv-jb"
REQ_FILE="requirements-jb.txt"
BOOK_DIR="materi-pendat"

echo "Using venv: ${VENV_DIR}"
if [ ! -d "${VENV_DIR}" ]; then
  python3 -m venv "${VENV_DIR}"
fi
source "${VENV_DIR}/bin/activate"
pip install --upgrade pip
pip install -r "${REQ_FILE}"

echo "Building Jupyter Book in ${BOOK_DIR}"
jupyter-book build "${BOOK_DIR}"

if [ "${DEPLOY_GHP:-0}" = "1" ]; then
  echo "Deploying to GitHub Pages with ghp-import"
  # Push built site to gh-pages branch (uses current git remote origin)
  ghp-import -n -p -f "${BOOK_DIR}/_build/html"
else
  echo "Build complete. To deploy run:" 
  echo "  DEPLOY_GHP=1 ./build_and_deploy.sh"
  echo "Or run the ghp-import command manually:" 
  echo "  ghp-import -n -p -f ${BOOK_DIR}/_build/html"
fi
