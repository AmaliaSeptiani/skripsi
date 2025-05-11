import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from tqdm.auto import tqdm
import joblib
import nltk
from nltk.tokenize import word_tokenize

# Unduh resource NLTK (hanya perlu sekali)
nltk.download('punkt')

# Baca dataset
input_file = 'D:/SKRIPSI/labelling_fix.xlsx'  # Pastikan file ini punya kolom: 'teks_normalisasi' & 'VADER_Sentiment'
df = pd.read_excel(input_file)

# Pastikan kolom yang dibutuhkan ada
if 'text_akhir' not in df.columns or 'VADER Sentiment (Binary)' not in df.columns:
    raise ValueError("Dataset harus memiliki kolom 'text_akhir' dan 'VADER Sentiment (Binary)'")

# Bersihkan data NaN
df = df.dropna(subset=['text_akhir'])  # Buang baris jika 'text_akhir' kosong
df['text_akhir'] = df['text_akhir'].astype(str)  # Pastikan kolom teks bertipe string

# Inisialisasi TF-IDF Vectorizer
tfidf_vectorizer = TfidfVectorizer(
    ngram_range=(1, 1),
    max_features=1000,
    min_df=7,
    max_df=0.7)

# Hitung TF-IDF dengan progress bar
print("🔍 Menghitung TF-IDF...")
with tqdm(total=1, desc="TF-IDF Processing", unit="step") as pbar:
    tfidf_matrix = tfidf_vectorizer.fit_transform(df['text_akhir'])
    pbar.update(1)

# Konversi ke DataFrame
tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=tfidf_vectorizer.get_feature_names_out())

# Tambahkan kolom label di paling kanan
label = df[['VADER Sentiment (Binary)']].reset_index(drop=True)
output_df = pd.concat([tfidf_df, label], axis=1)

# Simpan ke file Excel
# output_file = 'D:/SKRIPSI/tfidf_fix.xlsx'
# output_df.to_excel(output_file, index=False)

# # Simpan model TF-IDF ke file .pkl
# joblib.dump(tfidf_vectorizer, 'D:/SKRIPSI/tfidf_model_fix.pkl')

# print(f"✅ TF-IDF selesai! Disimpan di: {output_file}")
