## Judul Kasus

ANALISIS SENTIMEN ULASAN PRODUK APLIKASI E-COMMERCE LAYANAN TRANSPORTASI INDONESIA

---
## Anggota Kelompok & NIM
- Revan Yuda Febriansyah		714220070
- Bargana Kukuh Raditya		714220013
- Ruth Diana Purnamasari Sagala	714220042
- Gilang Andhika Buwana		714220046
  
---

## Deskripsi Kasus
# Sentiment Analysis: Gojek Twitter Reviews

Proyek ini bertujuan untuk melakukan analisis sentimen pada ulasan pengguna aplikasi Gojek dari Twitter menggunakan berbagai pendekatan machine learning klasik, dengan pipeline yang modular dan dapat direproduksi, mengidentifikasi sentimen positif, negatif, atau netral.
Analisis ini diharapkan dapat memberikan wawasan yang berguna dalam pengambilan keputusan yang strategis untuk pengembangan produk yang serupa di masa mendatang

---
## Sumber Dataset
dataset diambil dari hasil scraping ulasan pengguna terhadap produk gojek melalui platform X (Twitter)

---
## Langkah Preprocessing
1. Cleaning: Menghapus karakter spesial, angka, dan tanda baca yang tidak diperlukan.
2. Case Folding: Mengubah seluruh teks ke huruf kecil.
3. Tokenization: Memisahkan kalimat menjadi kata-kata.
4. Stopword Removal: Menghapus kata-kata umum yang tidak memiliki makna signifikan.
5. Stemming: Mengembalikan kata ke bentuk dasarnya menggunakan Sastrawi.
6. Labeling Sentimen: Memberi label sentimen (positif, negatif, netral) secara manual atau semi-otomatis.

---

## Algoritma yang Digunakan
Naive Bayes (NB)
Digunakan untuk klasifikasi sentimen berdasarkan probabilitas kemunculan kata.
dengan model :
- Naive Bayes + TF-IDF
- Naive Bayes + BoW
- Naive Bayes + Word2Vec
- Naive Bayes + TF-IDF+Chi2

---
## Evaluasi & Hasil
Evaluasi dilakukan menggunakan beberapa metrik performa model:

Akurasi
Precision
Recall
F1-Score
Confusion Matrix
Hasil evaluasi menunjukkan bahwa:
Model Naive Bayes + BoW menunjukkan performa terbaik dengan akurasi 70% (0.70)

---

## 📁 Struktur Folder
```
tugas-besar-datamining-kelompok4/
│
├── data/ # Dataset
│ ├── raw/ # Data mentah hasil scraping
│ └── processed/ # Data setelah preprocessing
│
├── notebook/ # Eksplorasi interaktif
│ ├── eda_template.ipynb
│ ├── preprocessing_template.ipynb
│ └── modeling_template.ipynb
│
├── report/ # Laporan akhir
│ ├── laporan-akhir_template.pdf
│ ├── lampiran_template.docx
│ └── struktur-lampiran.md
│
├── src/ # Source code
│ ├── data_loader.py
│ ├── model.py
│ ├── utils.py
│ ├── main.py
│ └── main_notebook.ipynb
│
├── run.sh # Bash script untuk menjalankan pipeline
├── requirements.txt # Daftar library yang dibutuhkan
└── README.md # Dokumentasi proyek ini
```
---

## 📌 Tahapan Analisis

1. **Data Collection**
   - Data dikumpulkan dari Twitter dengan kata kunci "gojek".
   - Dataset disimpan di `data/raw/`.

2. **Exploratory Data Analysis (EDA)**
   - Visualisasi kata paling umum dan distribusi sentimen.
   - EDA dilakukan untuk data mentah dan data yang telah dilabeli.

3. **Preprocessing**
   - Lowercasing, stopword removal, stemming, dan cleaning karakter non-alfabet.
   - Dataset bersih disimpan di `data/processed/gojek_cleaned.csv`.

4. **Sentiment Labeling**
   - Menggunakan model pretrained `mdhugol/indonesia-bert-sentiment-classification` dari HuggingFace Transformers.

5. **Modeling**
   - Menggunakan model Naive Bayes dengan beberapa metode representasi fitur:
     - Naive Bayes + TF-IDF
     - Naive Bayes + Bag-of-Words
     - Naive Bayes + Word2Vec
     - Naive Bayes + TF-IDF + Chi-Squared Feature Selection

6. **Evaluation**
   - Menggunakan Confusion Matrix dan Classification Report (precision, recall, f1-score).
   - Visualisasi akurasi dan distribusi prediksi.

---

## 🚀 Cara Menjalankan Pipeline

1. Pastikan dependencies sudah terinstall:
   ```bash
   pip install -r requirements.txt
   ```

2. Jalankan pipeline:
    ```
    ./run.sh
    ```

---

## 🧩 Requirements
Python ≥ 3.8
Dependensi Python disimpan di requirements.txt, meliputi:

- pandas, numpy, scikit-learn

- matplotlib, seaborn, wordcloud

- nltk, gensim, transformers, torch

---

## 📜 Lisensi
Proyek ini dibuat untuk keperluan pembelajaran dan tugas akademik. Gunakan dengan bijak.
