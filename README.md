# 📊 Sentiment Analysis: Gojek Twitter Reviews

Proyek ini bertujuan untuk melakukan analisis sentimen pada ulasan pengguna aplikasi Gojek dari Twitter menggunakan berbagai pendekatan machine learning klasik, dengan pipeline yang modular dan dapat direproduksi.

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