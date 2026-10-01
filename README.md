# Assignment 2 — Learning Strategy pada MNIST

**Nur Ilham Iskandar** · NIM 572104 · Magister Kecerdasan Artifisial, FMIPA UGM
Mata kuliah: Neural Network · Dosen: Afiahayati

Repository ini berisi:
1. Replikasi kode klasifikasi MNIST dari slide *Learning Strategy* (halaman 14–23).
2. Eksperimen untuk mengatasi overfitting: L1 regularization, L2 regularization, Dropout, Early stopping, dan kombinasi L1 + Dropout.

## Isi repository

| File | Keterangan |
|---|---|
| `Assignment2_MNIST_LearningStrategy.ipynb` | Notebook utama, sudah dijalankan (output, `model.summary()`, dan grafik tersimpan) |
| `run_experiments.py` | Versi script dari eksperimen yang sama |
| `figures/` | Grafik loss & accuracy tiap skenario dan `results.json` |
| `requirements.txt` | Library yang dibutuhkan |

## Cara menjalankan

```bash
pip install -r requirements.txt
jupyter notebook Assignment2_MNIST_LearningStrategy.ipynb
# atau
python run_experiments.py
```

Notebook juga bisa dibuka langsung di Google Colab.

## Setup eksperimen

- Arsitektur sama dengan slide: Flatten (784) → Dense 64 ReLU → Dense 10 softmax, optimizer Adam, batch 100.
- Split data: 55.000 train / 5.000 validation / 10.000 test. Test set tidak dipakai selama training.
- 30 epoch (bukan 10) supaya overfitting pada baseline terlihat jelas.
- Seed 42 untuk semua skenario.

## Hasil

**Bagian 1 (replikasi slide):** akurasi test 97,27% (slide: 97,61%), 50.890 trainable params.

**Bagian 2:**

| Skenario | Akurasi train | Akurasi test | Gap (poin) | Cross-entropy test | Epoch |
|---|---|---|---|---|---|
| Baseline | 99,74% | 97,45% | 2,29 | 0,1105 | 30 |
| 2a L1 (0,0001) | 98,60% | **97,64%** | 0,96 | **0,0766** | 30 |
| 2b L2 (0,001) | 97,42% | 96,98% | **0,44** | 0,1140 | 30 |
| 2c Dropout (0,3) | 99,24% | 97,46% | 1,78 | 0,0915 | 30 |
| 2d Early stopping (patience 3) | 98,28% | 97,24% | 1,04 | 0,0965 | 11 |
| 2e L1 + Dropout | 98,09% | 97,39% | 0,70 | 0,0874 | 30 |

Cross-entropy test dihitung tanpa penalti regularization, sehingga bisa dibandingkan antar skenario.

## Kesimpulan

- Semua teknik mengurangi overfitting (gap turun dari 2,29 menjadi 0,44–1,78 poin).
- L1 (1e-4) memberi generalisasi terbaik: akurasi test tertinggi dan cross-entropy test terendah.
- L2 dengan rate 1e-3 terlalu kuat sehingga model underfitting.
- Early stopping paling hemat: model sama, 19 epoch lebih sedikit.
