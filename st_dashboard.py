import streamlit as st
import pandas as pd
import joblib
from preprocessing import (
    cleaningText, casefoldingText, tokenizingText, stemmingText, toSentence, fix_slangwords
)
import warnings


# Filter warning
warnings.filterwarnings("ignore", category=UserWarning, message="X does not have valid feature names")

# Load model TF-IDF dan SVM
tfidf_vectorizer = joblib.load('tfidf_model_trained.pkl')
svm_model = joblib.load('model_svm_trained.pkl')

# Fungsi load kamus dari CSV
def load_slangwords_from_csv(csv_file_path):
    try:
        df_kamus = pd.read_csv(csv_file_path, encoding='ISO-8859-1')
        if 'word' not in df_kamus.columns or 'correct_word' not in df_kamus.columns:
            raise ValueError("File tidak memiliki kolom 'word' dan 'correct_word'.")
        return dict(zip(df_kamus['word'], df_kamus['correct_word']))
    except Exception as e:
        st.error(f"Gagal load kamus: {e}")
        return {}

# Load base slangwords dan custom dari CSV
csv_kamus_path = 'new_kamusalay.csv'
custom_slangwords = load_slangwords_from_csv(csv_kamus_path)

# Base slangwords (dipersingkat di sini, tapi gunakan yang lengkap di implementasi final)
base_slangwords = {
    "dg":"dengan", "dr":"dari", "tdk":"tidak", "euy":"ini", "jlebb":"sakit", "jleb":"sakit",
    "adaitu":"ada itu", "wajarkrna":"wajar karena", "krna":"karena", "buzzerrp":"buzzer", 
    "aresl":"area","adatxlbh":"adat lebih", "lbh":"lebih", "sedihmelihat":"sedih melihat",
    "kejahatansetiap":"kejahatan setiap","matakarena": "mata karena","katakatanya":"kata katanya",
    "bismillahmenurut":"bismillah menurut", "uara":"suara","narasisemoga":"narasi semoga",
    "nyesss":"sakit", "kontenkonten":"konten konten","indonesiasemoga":"indonesia semoga",
    "mantappp":"mantap", "waduhhh":"waduh","kalimantanmemang":"kalimantan memang",
    "sulitapalagi":"sulit apalagi","jawaenakair":"jawa enak air","yaqin":"yakin,",
    "banggasalutterharumerindingkita":"bangga salut terharu merinding kita",
    "alhamdulillahsmg":"alhamdulillah semoga","alhamdulillahsemoga":"alhamdulillah semoga",
    "merindingbetapa":"merinding betapa","airsemua":"air semua","kerennnnnsemoga":"keren semoga",
    "terharuuuuuuuuu":"terharu","terharuu":"terharu","manteppppp":"mantap","pakdhesemoga":"pakdhe semoga",
    "sewotkawatir":"marah khawatir","sewot":"marah","kawatir":"khawatir","embungsmg":"embung semoga",
    "tingal":"tinggal","batmankenapa":"batman kenapa","inisaya":"ini saya","ktahanan":"ketahanan",
    "panganbikin":"pangan bikin","pmerintah":"pemerintah","pindahktp":"pindah ktp","mendoakn":"mendoakan",
    "yahbeda":"yah berbeda", "propinsisemoga":"provinsi semoga","propinsi":"provinsi", "gw":"saya",
    "aku":"saya", "y":"ya","rcn":"rencana","narasi2":"narasi narasi","bawak":"bawa","p":"bapak",
    "pak":"bapak","hrs":"harus", "alloh":"allah","dn":"dan","dgn":"dengan","alllah":"allah",
    "kerennn":"keren", "smg":"semoga","bp":"bapak","sdh":"sudah","terwujut":"terwujud","ideide":"ide ide",
    "thn":"tahun","jk":"jika","jln":"jalan","jkw":"jokowi","jama":"zaman","jaman":"zaman","byk":"banyak",
    "kereeen":"keren","klo":"kalau","negatip":"negatif","positip":"positif","dri":"dari",
    "ngak":"tidak","gimna":"gimana","ad":"ada","n":"dan","sll":"selalu","aya":"ada","naon":"apa",
    "orng":"orang","slll":"selalu","brasa":"berasa","sajah":"saja","prof":"profesor","bpk":"bapak",
    "indononesia":"indonesia","ahkir":"akhir","jokowidodosehat":"jokowidodo sehat","sht":"sehat",
    "salud":"salut","dirimu tapi":"diri kamu tapi","dirimu":"diri kamu","jokowiberani":"jokowi berani",
    "iknbaru":"ikn baru","dikalimantansaya":"di kalimantan saya","mangkrakutang":"mangkrak hutang",
    "utang":"hutang","ibukotabaru":"ibu kota baru","kecurangansetiap":"kecurangan setiap",
    "rayapadahal":"raya padahal","dikalbarsedih":"di kalbar sedih","no":"nomor","saudaraa":"saudara",
    "narasidengan":"narasi dengan","iniakan":"ini akan","the real":"nyata",
    # Negations
    "ga": "tidak", "gak": "tidak", "engga": "tidak", "kagak": "tidak", 
    "gk": "tidak", "tdk": "tidak", "gpp": "tidak apa-apa", "gapapa": "tidak apa-apa",
    "ngga": "tidak", "nggak": "tidak", "gaada": "tidak ada", "gada": "tidak ada",
    
    # Pronouns
    "yg": "yang", "org": "orang", "ko": "kamu", "loe": "kamu", "lu": "kamu", 
    "elo": "kamu", "gw": "saya", "gue": "saya", "ane": "saya", "sya": "saya",
    "akuu": "aku", "akku": "aku", "aq": "aku", "sy": "saya", "ane": "saya",
    "kmu": "kamu", "km": "kamu", "kalo": "kalau", "kl": "kalau",
    
    # Prepositions/conjunctions
    "dgn": "dengan", "sm": "dengan", "klo": "kalau", "klu": "kalau",
    "krn": "karena", "karna": "karena", "krna": "karena", "dri": "dari",
    "dr": "dari", "d": "di", "ke": "kepada", "pd": "pada", "utk": "untuk",
    "buat": "untuk", "biar": "agar", "supaya": "agar",
    
    # Adverbs
    "udah": "sudah", "udh": "sudah", "udeh": "sudah", "skrg": "sekarang",
    "skrng": "sekarang", "nnti": "nanti", "ntar": "nanti", "kmrn": "kemarin",
    "sblm": "sebelum", "td": "tadi", "bs": "bisa", "dpt": "dapat", "bnyk": "banyak",
    "sdh": "sudah", "blm": "belum", "blum": "belum", "lg": "lagi", "lgi": "lagi",
    
    # Question words
    "gmn": "bagaimana", "gmna": "bagaimana", "gimana": "bagaimana",
    "knp": "kenapa", "kpn": "kapan", "pa": "apa", "ap": "apa", "ape": "apa",
    "siapa": "siapa", "sp": "siapa", "mana": "di mana", "mna": "di mana",
    
    # Common expressions
    "nih": "ini", "aja": "saja", "jg": "juga", "jga": "juga", "jgn": "jangan",
    "mah": "sih", "deh": "sih", "dong": "dong", "tau": "tahu", "tw": "tahu", 
    "kq": "kok", "koq": "kok", "apk": "aplikasi", "app": "aplikasi", "yg": "yang",
    "bgt": "banget", "bngt": "banget", "ampe": "sampai", "sampe": "sampai",
    
    # Positive expressions
    "thx": "terima kasih", "trims": "terima kasih", "makasih": "terima kasih",
    "mksh": "terima kasih", "ok": "oke", "okey": "oke", "sip": "oke", 
    "wkwk": "tertawa", "wkwkwk": "tertawa", "haha": "tertawa", "kocak": "lucu",
    "ngakak": "tertawa", "mantap": "bagus", "keren": "bagus", "gacor": "lancar",
    "cuy": "bro", "coy": "bro", "gan": "bro", "sis": "sister", "bro": "brother",
    
    # Negative expressions (non-vulgar)
    "anjir": "astaga", "anjrit": "astaga", "wtf": "astaga", "astaghfirullah": "astaga",
    "parah": "berlebihan", "ekstrim": "berlebihan", "capek": "lelah", "cape": "lelah",
    "bngung": "bingung", "puyeng": "bingung", "ribet": "rumit", "ruwet": "rumit",
    "lemot": "lambat", "lelet": "lambat", "eror": "error", "err": "error",
    "gaje": "tidak jelas", "gajelas": "tidak jelas", "gabut": "tidak ada kerjaan", "ga bagus":"jelek", "ga jelas":"jelek",
    
    # Vulgar words (neutralized)
    "goblok": "bodoh", "bego": "bodoh", "dungu": "bodoh", "tolol": "bodoh",
    "bangsat": "sialan", "kampret": "sialan", "brengsek": "sialan",
    "asu": "anjing", "anjing": "anjing", "jancok": "sialan", "jancuk": "sialan",
    "cok": "sialan", "bajingan": "tidak baik", "kontol": "alat kelamin laki-laki",
    
    # Technical terms
    "judol": "judi online", "coretax": "sistem pajak", "kominfo": "kementerian komunikasi",
    "djp": "direktorat jenderal pajak", "kpp": "kantor pelayanan pajak",
    "npwp": "nomor pokok wajib pajak", "spt": "surat pemberitahuan", "ppn": "pajak pertambahan nilai",
    "e-faktur": "faktur elektronik", "e-filing": "pelaporan elektronik", "freez": "rusak", "lag":"rusak",
    
    # Internet slang
    "cmiiw": "koreksi jika saya salah", "afk": "jauh dari keyboard",
    "btw": "ngomong-ngomong", "omg": "ya Tuhan", "lol": "tertawa", "rofl": "tertawa terbahak-bahak",
    "dm": "pesan langsung", "pm": "pesan pribadi", "cc": "carbon copy", "bcc": "blind carbon copy",
    
    # Common typos
    "emang": "memang", "emg": "memang", "emng": "memang", "bener": "benar",
    "bner": "benar", "bnr": "benar", "kyk": "kaya", "ky": "kaya", "kek": "kaya",
    "kayak": "seperti", "kaya": "seperti", "kayanya": "sepertinya",
    
    # Reduplicated words
    "cepet": "cepat", "cepetan": "cepatan", "pelan": "lambat", "pelan-pelan": "perlahan",
    "santai": "santai", "santuy": "santai", "asyik": "asyik", "asyikk": "asyik",
    
    # English loanwords
    "save": "simpan", "savenya": "simpannya", "upload": "unggah", "download": "unduh",
    "update": "pembaruan", "upgrade": "tingkatkan", "install": "pasang", "uninstall": "copot", 
    "ngebug": "rusak", "freze": "rusak", "freeze": "rusak", "bug": "rusak",
    
    # New slang (2023-2024)
    "kepo": "penasaran", "gabut": "tidak ada kerjaan", "mager": "malas gerak",
    "ygy": "ya guys ya", "sukasuka": "suka-suka", "random": "acak", "geming": "gaming",
    "halu": "halusinasi", "gercep": "gerak cepat", "pansos": "panjat sosial",
    "baper": "bawa perasaan", "gabungan": "gabung", "nyok": "ayo", "yuk": "ayo",
}
all_slangwords = {**base_slangwords, **custom_slangwords}

# Fungsi Preprocessing
def preprocess_input(text):
    cleaned = cleaningText(text)
    casefolded = casefoldingText(cleaned)
    slang_fixed = fix_slangwords(casefolded, all_slangwords)
    tokenized = tokenizingText(slang_fixed)
    stemmed = stemmingText(tokenized)
    final_text = toSentence(stemmed)
    return final_text

# Fungsi prediksi
def prediksi_sentimen(kalimat):
    kalimat_bersih = preprocess_input(kalimat)
    tfidf_input = tfidf_vectorizer.transform([kalimat_bersih]).toarray()
    prediksi = svm_model.predict(tfidf_input)[0]
    return "Positif" if prediksi == 1 else "Negatif", kalimat_bersih

# === Streamlit UI === #
st.set_page_config(page_title="Analisis Sentimen", page_icon="🧠", layout="centered")

st.title("📊 Dashboard Analisis Sentimen dengan SVM + TF-IDF")
st.write("Masukkan kalimat yang ingin Anda analisis:")

user_input = st.text_area("Teks input", height=150)

if st.button("Prediksi Sentimen"):
    if user_input.strip() == "":
        st.warning("Masukkan kalimat terlebih dahulu!")
    else:
        label, processed_text = prediksi_sentimen(user_input)
        st.success(f"Hasil Prediksi Sentimen: **{label}**")
        with st.expander("🔍 Lihat Hasil Preprocessing"):
            st.code(processed_text, language="text")
