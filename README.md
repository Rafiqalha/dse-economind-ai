# EconoMind AI

Sistem prediksi keputusan harga jual berbasis Machine Learning.

Project IT Incubation DSE Teknik Informatika UIN Maulana Malik Ibrahim Malang.

## Fitur

Model memprediksi keputusan harga menjadi:

- NAIK
- STABIL
- TURUN

berdasarkan:

- Biaya Produksi per Unit
- Persentase Inflasi Daerah
- Jumlah Kompetitor Aktif
- Tren Permintaan Konsumen
- Kelangkaan Bahan Baku
- Sentimen Media Sosial

## Tech Stack

- Python
- NumPy
- Model Random Forest dari scikit-learn
- Joblib
- Streamlit

Inferensi dijalankan melalui `portable_forest.py`, sehingga aplikasi tidak
perlu memuat DLL pandas, SciPy, atau scikit-learn yang dapat diblokir oleh
Windows Application Control.

## Menjalankan Aplikasi

```bash
python -m pip install -r requirements.txt
python -m streamlit run app.py
```
