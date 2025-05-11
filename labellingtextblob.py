import pandas as pd
from textblob import TextBlob
from deep_translator import GoogleTranslator

# Fungsi untuk menerjemahkan komentar
def translate_comments(comments):
    translated_comments = []
    translator = GoogleTranslator(source='auto', target='en')
    for comment in comments:
        if pd.isna(comment) or not comment.strip():  # Cek komentar kosong atau NaN
            translated_comments.append("No comment")
        else:
            try:
                # Pecah komentar panjang menjadi bagian yang lebih kecil
                comment_parts = [comment[i:i+1000] for i in range(0, len(comment), 1000)]
                translated = ""
                for part in comment_parts:
                    translated += translator.translate(part) + " "
                translated_comments.append(translated.strip())  # Gabungkan hasil terjemahan
            except Exception as e:
                print(f"Error translating comment: {comment}, Error: {e}")
                translated_comments.append(comment)  # Jika gagal, gunakan komentar asli
    return translated_comments

# Fungsi analisis sentimen menggunakan TextBlob (binary: 1 untuk positif & 0 untuk negatif)
def textblob_sentiment_binary(comment):
    analysis = TextBlob(comment)
    score = analysis.sentiment.polarity
    return 1 if score >= 0.05 else 0  # Menggunakan 1 untuk positif dan 0 untuk negatif

# Membaca file Excel
input_file = 'D:/SKRIPSI/dataset_cleaned_without_emoticon.xlsx'
df = pd.read_excel(input_file)

# Asumsikan kolom komentar bernama 'teks_normalisasi'
comments = df['teks_normalisasi'].tolist()

# Terjemahkan komentar ke bahasa Inggris
translated_comments = translate_comments(comments)

# Lakukan analisis sentimen menggunakan TextBlob (1 untuk positif, 0 untuk negatif)
textblob_labels = [textblob_sentiment_binary(comment) for comment in translated_comments]

# Tambahkan hasil terjemahan dan label sentimen ke DataFrame
df['Translated Comment'] = translated_comments
df['TextBlob Sentiment (Binary)'] = textblob_labels

# Simpan hasil ke file Excel baru
output_file = 'D:/SKRIPSI/labelling_textblob_wo_0.5.xlsx'
df.to_excel(output_file, index=False)

print(f"Hasil analisis sentimen TextBlob (1 untuk positif dan 0 untuk negatif) telah disimpan ke {output_file}")
