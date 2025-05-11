import pandas as pd

# Path ke file lexicon positif dan negatif
positive_lexicon_path = "D:/SKRIPSI/InSet-master/positive.tsv"
negative_lexicon_path = "D:/SKRIPSI/InSet-master/negative.tsv"

# Load data dari file TSV
positive_words = pd.read_csv(positive_lexicon_path, sep='\t', header=0, names=['word', 'weight'], encoding='utf-8')
negative_words = pd.read_csv(negative_lexicon_path, sep='\t', header=0, names=['word', 'weight'], encoding='utf-8')

# Gabungkan lexicon positif dan negatif ke dalam satu dictionary
lexicon_dict = {row['word']: row['weight'] for _, row in pd.concat([positive_words, negative_words]).iterrows()}

# Fungsi untuk analisis sentimen satu kalimat dan menampilkan transparansi skor
def analisis_skoring_komentar(comment):
    print(f"\nKalimat: {comment}")
    words = comment.lower().split()
    total_score = 0
    print("\nDetail kontribusi kata:")
    for word in words:
        score = lexicon_dict.get(word, 0)
        total_score += score
        print(f"  {word} : {score}")

    print(f"\nTotal Skor: {total_score}")
    if total_score > 0:
        print("Sentimen: Positive")
    elif total_score < 0:
        print("Sentimen: Negative")
    else:
        print("Sentimen: Neutral")

# Uji coba dengan satu kalimat
kalimat_uji = "jatuh perhatian sayang "
analisis_skoring_komentar(kalimat_uji)
