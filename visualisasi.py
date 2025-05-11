import pandas as pd
from collections import Counter
import matplotlib.pyplot as plt
import nltk
from nltk.tokenize import word_tokenize
import re

# Download resource NLTK
nltk.download('punkt')

# Baca file Excel
df = pd.read_excel('dataset.xlsx')

# Gabungkan semua komentar jadi satu teks
all_text = ' '.join(df['Comments'].dropna().astype(str).tolist())

# Tokenisasi dan filter hanya kata (alfabet)
tokens = word_tokenize(all_text.lower())
words_only = [word for word in tokens if re.match(r'^[a-zA-Z]+$', word)]

# Hitung frekuensi kata
word_counts = Counter(words_only)

# Ambil 10 kata terbanyak
top_10 = word_counts.most_common (150)
words, freqs = zip(*top_10)

# Plot barchart
plt.figure(figsize=(10, 6))
plt.bar(words, freqs, color='steelblue')
plt.title('10 Kata dengan Frekuensi Terbanyak')
plt.xlabel('Kata')
plt.ylabel('Frekuensi')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
