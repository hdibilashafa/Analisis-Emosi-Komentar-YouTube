# Analisis Emosi Komentar YouTube: Dalang Demo dan Mencegah Darurat Militer

![Python](https://img.shields.io/badge/Python-3-blue)
![Google Colab](https://img.shields.io/badge/Google-Colab-F9AB00)
![scikit-learn](https://img.shields.io/badge/scikit--learn-SVM-F7931E)
![IndoBERT](https://img.shields.io/badge/IndoBERT-Transformers-yellow)
![NLP](https://img.shields.io/badge/NLP-Emotion_Analysis-green)

Analisis emosi pada **11.271 komentar YouTube** dari video Ferry Irwandi berjudul *"Dalang Demo dan Mencegah Darurat Militer"*. Proyek ini membandingkan dua pendekatan:

1. **Lexicon + SVM**: komentar dilabeli emosinya dengan kamus emosi (NRC Emotion Lexicon versi Indonesia), lalu diklasifikasikan dengan SVM
2. **IndoBERT**: model dilatih (fine-tuning) pada dataset emosi Twitter berbahasa Indonesia, lalu dipakai memprediksi emosi komentar YouTube

> Analisis ini bersifat akademik dan tidak mencerminkan opini penulis terhadap topik maupun isi komentar.

---

## Dataset

| Data | Jumlah | Keterangan |
|---|---|---|
| `youtube_comments.csv` | 11.271 komentar | Kolom: `publishedAt`, `textDisplay`, `likeCount` |
| `Twitter_Emotion_Dataset.csv` | 4.401 tweet | Data latih IndoBERT, 5 kelas: anger, happy, sadness, fear, love |
| `indonesian_emotion_lexicon.csv` | 14.154 kata | Lexicon emosi (anger, anticipation, disgust, fear, joy, sadness, surprise, trust, plus polaritas negative/positive) |
| `stopword/stoplist.txt` | 768 kata | Daftar stopword kustom |
| `stopword/singkatan-lib.csv` | 175 pasang | Kamus normalisasi singkatan (misal `akn` menjadi `akan`) |

---

## Pipeline

```
Komentar YouTube
      │
      ├── Preprocessing (HTML, URL, mention, emoji, huruf berulang, singkatan, stopword)
      │
      ├── Pendekatan 1: Lexicon → label 8 emosi + neutral → TF-IDF → SVM
      │
      └── Pendekatan 2: IndoBERT (fine-tuning pada Twitter) → prediksi 5 emosi
```

### Preprocessing

Hapus HTML entity dan tag, URL, `@mention`, `#hashtag`, emoji, angka, dan tanda baca. Lalu ubah ke huruf kecil, rapikan huruf berulang (`baguuus` menjadi `bagus`), normalisasi singkatan, dan hapus stopword.

### Pendekatan 1: Lexicon dan SVM

| Tahap | Detail |
|---|---|
| Pelabelan | Setiap kata dicocokkan dengan lexicon, skor emosi dijumlahkan, emosi dengan skor tertinggi dipilih (voting). Tanpa kata cocok menjadi `neutral` |
| Emosi dasar | 8 emosi Plutchik: anger, anticipation, disgust, fear, joy, sadness, surprise, trust, plus `neutral` |
| Fitur | TF-IDF, maksimal 3.000 fitur, unigram dan bigram, `min_df=2` |
| Model | `LinearSVC(class_weight='balanced')` |
| Split | 80% latih, 20% uji (stratified, `random_state=42`) |

**Distribusi label hasil lexicon:**

| Emosi | Jumlah | Persentase |
|---|---|---|
| neutral | 5.240 | 46,5% |
| anger | 1.375 | 12,2% |
| trust | 1.298 | 11,5% |
| fear | 1.261 | 11,2% |
| anticipation | 860 | 7,6% |
| disgust | 526 | 4,7% |
| joy | 363 | 3,2% |
| sadness | 288 | 2,6% |
| surprise | 60 | 0,5% |

**Hasil SVM** (data uji 2.193 komentar): akurasi **0,85**, macro F1 **0,75**.

| Emosi | Precision | Recall | F1 |
|---|---|---|---|
| neutral | 0,92 | 0,96 | 0,94 |
| fear | 0,86 | 0,83 | 0,85 |
| disgust | 0,82 | 0,86 | 0,84 |
| trust | 0,86 | 0,80 | 0,83 |
| joy | 0,82 | 0,77 | 0,79 |
| anger | 0,79 | 0,74 | 0,77 |
| anticipation | 0,71 | 0,69 | 0,70 |
| sadness | 0,63 | 0,66 | 0,64 |
| surprise | 0,40 | 0,33 | 0,36 |

### Pendekatan 2: IndoBERT

| Tahap | Detail |
|---|---|
| Model | `indobenchmark/indobert-base-p1` |
| Data latih | Twitter Emotion Dataset (3.520 latih, 881 uji) |
| Hyperparameter | 3 epoch, batch size 16, warmup 100 langkah, weight decay 0,01, max length 128 |
| Perangkat | GPU Tesla T4 (Google Colab), waktu training ± 6 menit |
| Prediksi | 10.984 komentar YouTube yang sudah bersih |

**Hasil pada data uji Twitter** (881 tweet): akurasi **0,69**, macro F1 **0,70**.

| Emosi | Precision | Recall | F1 |
|---|---|---|---|
| love | 0,80 | 0,83 | 0,82 |
| fear | 0,79 | 0,70 | 0,74 |
| anger | 0,74 | 0,70 | 0,72 |
| happy | 0,71 | 0,63 | 0,67 |
| sadness | 0,52 | 0,64 | 0,57 |

---

## Struktur Repository

```
.
├── bert_emotion_twitter_to_youtube.ipynb   # IndoBERT: training + prediksi + visualisasi
├── script/
│   ├── Pelabelan_Emosi_Menggunakan_Lexicon.ipynb   # pelabelan emosi dengan lexicon
│   ├── Analisis_Emosi_Menggunakan_SVM.ipynb        # klasifikasi SVM
│   └── 100_sample/                                  # versi sampel 100 komentar (uji cepat)
├── stopword/
│   ├── stoplist.txt
│   └── singkatan-lib.csv
├── output/
│   ├── youtube_comment_emotion_labeled.csv          # hasil pelabelan lexicon (11.271 baris)
│   └── youtube_comment_emotion_labeled_100.csv      # versi sampel 100 baris
├── youtube_comments.csv
├── Twitter_Emotion_Dataset.csv
└── indonesian_emotion_lexicon.csv
```

---

## Cara Menjalankan (Google Colab)

Semua notebook dibuat untuk **Google Colab**, jadi tidak perlu install apa pun di komputer.

### Urutan untuk pendekatan Lexicon dan SVM

1. Buka `script/Pelabelan_Emosi_Menggunakan_Lexicon.ipynb` di Colab
2. Upload ke sesi Colab: `youtube_comments.csv`, `indonesian_emotion_lexicon.csv`, `stoplist.txt`, `singkatan-lib.csv`
3. Jalankan semua sel (**Runtime → Run all**). Hasilnya tersimpan sebagai `youtube_comment_emotion_labeled.csv`
4. Buka `script/Analisis_Emosi_Menggunakan_SVM.ipynb`, upload `youtube_comment_emotion_labeled.csv` ke sesi Colab, lalu jalankan semua sel

### Untuk IndoBERT

1. Buka `bert_emotion_twitter_to_youtube.ipynb` di Colab
2. Ganti runtime ke **GPU** (**Runtime → Change runtime type → T4 GPU**)
3. Jalankan sel upload, lalu pilih: `Twitter_Emotion_Dataset.csv`, `stoplist.txt`, `singkatan-lib.csv`, `youtube_comments.csv`
4. Jalankan sisa sel. Hasil prediksi tersimpan sebagai `youtube_comments_bert_labeled.csv`

---

## Tech Stack

Python · Google Colab · pandas · scikit-learn (TF-IDF, LinearSVC) · Hugging Face Transformers · PyTorch · IndoBERT · Sastrawi · NLTK · Matplotlib · Seaborn · WordCloud

---

## Keterbatasan

- **Label lexicon adalah label otomatis**, bukan label manusia. Akurasi SVM 0,85 menunjukkan seberapa baik SVM meniru aturan lexicon, bukan seberapa benar emosi sebenarnya
- **Pelabelan lexicon sederhana**: hanya mencocokkan kata, sehingga sarkasme, kalimat negasi, dan bahasa gaul sering terlewat. Hampir separuh komentar (46,5%) berlabel `neutral` karena tidak ada kata yang cocok
- **Kelas tidak seimbang**: emosi `surprise` hanya 60 komentar, sehingga performanya paling rendah
- **IndoBERT dilatih pada domain Twitter** dan hanya punya 5 emosi tanpa kelas `neutral`, sehingga setiap komentar YouTube dipaksa masuk ke salah satu dari 5 emosi
- **Hasil kedua pendekatan tidak bisa dibandingkan langsung** karena set label dan data ujinya berbeda
- Data hanya berasal dari **satu video** dan satu periode pengambilan, sehingga tidak mewakili seluruh opini publik
