from nltk.sentiment.vader import SentimentIntensityAnalyzer
from deep_translator import GoogleTranslator
import nltk

# Download lexicon VADER
nltk.download('vader_lexicon')

# Inisialisasi analyzer dan translator
analyzer = SentimentIntensityAnalyzer()

# Kalimat dalam Bahasa Indonesia
kalimat = [
     "rakyat msih banyak yang kurang . arti indonesia sedikit belum merdeka . apalagi maju .",
    "waow ikn keren , salam sukses berkah sehat dari tvade kakak",
    "saya nanti akan libur sama keluarga ke ikn"
]

# Analisis tiap kalimat
for teks in kalimat:
    # Terjemahkan kalimat ke Bahasa Inggris menggunakan DeepL
    translated_text = GoogleTranslator(source='id', target='en').translate(teks)
    
    # Analisis sentimen pada kalimat terjemahan
    skor = analyzer.polarity_scores(translated_text)
    compound = skor['compound']
    
    # Aturan klasifikasi
    if compound >= 0.05:
        label = 'Positif'
    elif compound <= -0.05:
        label = 'Negatif'
    else:
        label = 'Netral'
    
    # Output
    print(f"Kalimat Asli: {teks}")
    print(f"Terjemahan (Bahasa Inggris): {translated_text}")
    print(f"Skor Sentimen: {skor}")
    print(f"Label Sentimen: {label}")
    print("-" * 40)
