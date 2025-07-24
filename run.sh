#!/bin/bash

echo "=== Memulai pipeline analisis sentimen ==="

# Cek Python
echo "Python yang digunakan:"
which python
python --version

# Membuat struktur folder jika belum ada
mkdir -p data/processed
mkdir -p data/raw

# Menjalankan pipeline utama
echo "Menjalankan main.py ..."
python src/main.py

echo "=== Pipeline selesai dijalankan ==="
