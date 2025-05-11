import pandas as pd
import re
import string
from nltk.tokenize import word_tokenize
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

# Load data dari file Excel
def load_data(input_file):
    return pd.read_excel(input_file)

# Fungsi-fungsi untuk preprocessing
def cleaningText(text):
    text = str(text)
    text = re.sub(r'@[A-Za-z0-9]+', '', text)
    text = re.sub(r'#[A-Za-z0-9]+', '', text)
    text = re.sub(r'RT[\s]', '', text)
    text = re.sub(r"http\S+", '', text)
    text = re.sub(r'[0-9]+', '', text)  
    text = re.sub(r'[^\w\s]', '', text) 
    text = text.replace('\n', ' ') 
    text = text.strip()
    return text

def casefoldingText(text):
    return text.lower()

def tokenizingText(text):
    return word_tokenize(text)

factory = StemmerFactory()
stemmer = factory.create_stemmer()

def stemmingText(tokens):
    return [stemmer.stem(word) for word in tokens]

def toSentence(tokens):
    return ' '.join(tokens)

# Fungsi fix slang words
def fix_slangwords(text, slang_dict):
    return ' '.join([slang_dict.get(word, word) for word in text.split()])

# Fungsi load slangwords dari CSV
def load_slangwords_from_csv(csv_file_path):
    try:
        # Baca file Excel dengan header yang sesuai
        df_kamus = pd.read_csv(csv_file_path, encoding='ISO-8859-1')
        
        # Pastikan kolom yang ada adalah 'word' dan 'correct word'
        if 'word' not in df_kamus.columns or 'correct_word' not in df_kamus.columns:
            raise ValueError("File Excel tidak memiliki kolom 'word' dan 'correct_word'.")
        
        # Mengubah menjadi kamus
        kamus = dict(zip(df_kamus['word'], df_kamus['correct_word']))
        return kamus
    except Exception as e:
        print(f"Gagal load kamus dari Excel: {e}")
        return {}

# Load slangwords dari file lokal CSV
csv_kamus_path = 'new_kamusalay.csv'  # Ganti dengan path kamus lokal CSV

# Load slangwords custom
custom_slangwords = load_slangwords_from_csv(csv_kamus_path)

# Base slangwords
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
    "baper": "bawa perasaan", "gabungan": "gabung", "nyok": "ayo", "yuk": "ayo",}  # Lanjutkan kamus seperti di atas

# Gabungkan base slangwords dengan custom slangwords
all_slangwords = {**base_slangwords, **custom_slangwords}

def process_text(df):
    df['text_clean'] = df['Comments'].apply(cleaningText)
    df['text_casefolding'] = df['text_clean'].apply(casefoldingText)
    df['text_slangwords'] = df['text_casefolding'].apply(lambda x: fix_slangwords(x, all_slangwords))
    df['text_tokenizing'] = df['text_slangwords'].apply(tokenizingText)
    df['text_stemming'] = df['text_tokenizing'].apply(stemmingText)
    df['text_akhir'] = df['text_stemming'].apply(toSentence)
    return df

# Simpan hasil ke file Excel
def save_to_excel(df, output_file):
    df.to_excel(output_file, index=False)

if __name__ == "__main__":
    # Input dan Output file Excel
    input_file = 'dataset.xlsx'  # Ganti dengan path file Excel input
    output_file = 'dataset_fix.xlsx'  # Ganti dengan path file Excel output

    # Load data
    df = load_data(input_file)

    # Proses text
    df_processed = process_text(df)

    # Simpan hasil
    save_to_excel(df_processed, output_file)
    print(f"Data telah diproses dan disimpan ke {output_file}")
