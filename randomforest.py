import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from imblearn.over_sampling import SMOTE
import numpy as np
import matplotlib.pyplot as plt
import joblib
from tqdm.auto import tqdm
import nltk

# Unduh tokenizer NLTK (sekali saja)
nltk.download('punkt')

# === Step 1: Baca dataset ===
input_file = 'D:/SKRIPSI/labelling_fix.xlsx'
df = pd.read_excel(input_file)

# Validasi kolom
if 'text_akhir' not in df.columns or 'VADER Sentiment (Binary)' not in df.columns:
    raise ValueError("Dataset harus memiliki kolom 'text_akhir' dan 'VADER Sentiment (Binary)'")

# Bersihkan data
df = df.dropna(subset=['text_akhir'])
df['text_akhir'] = df['text_akhir'].astype(str)

# === Step 2: Split data sebelum TF-IDF ===
X_text = df['text_akhir']
y = df['VADER Sentiment (Binary)']

X_train_text, X_test_text, y_train, y_test = train_test_split(
    X_text, y, test_size=0.2, random_state=42, stratify=y)

# === Step 3: TF-IDF hanya dari data latih ===
print("🔍 Menghitung TF-IDF hanya dari data latih...")
tfidf_vectorizer = TfidfVectorizer(
    ngram_range=(1, 1),
    max_features=1000,
    min_df=7,
    max_df=0.7
)

with tqdm(total=1, desc="TF-IDF Processing", unit="step") as pbar:
    X_train_tfidf = tfidf_vectorizer.fit_transform(X_train_text)
    X_test_tfidf = tfidf_vectorizer.transform(X_test_text)
    pbar.update(1)

# === Step 4: SMOTE pada data latih ===
print("📊 Distribusi Label Sebelum SMOTE:")
print(pd.Series(y_train).value_counts(), '\n')

smote = SMOTE(random_state=42)
X_train_bal, y_train_bal = smote.fit_resample(X_train_tfidf, y_train)

print("📊 Distribusi Label Setelah SMOTE:")
print(pd.Series(y_train_bal).value_counts(), '\n')

# === Step 5: Latih model Random Forest ===
rf_model = RandomForestClassifier(
    n_estimators=300,
    random_state=42,
    class_weight='balanced'
)
rf_model.fit(X_train_bal, y_train_bal)

# === Step 6: Evaluasi ===
y_pred = rf_model.predict(X_test_tfidf)

print("\n📋 Classification Report (Testing):\n")
print(classification_report(y_test, y_pred, zero_division=0))

accuracy_train = accuracy_score(y_train_bal, rf_model.predict(X_train_bal))
print(f"📈 Akurasi Training: {accuracy_train:.4f}")

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=rf_model.classes_)
disp.plot(cmap='Blues', values_format='d')
plt.title("Confusion Matrix - Random Forest")
plt.grid(False)
plt.show()

# === Optional: Simpan model dan vectorizer ===
# joblib.dump(rf_model, 'D:/SKRIPSI/model_rf_fix.pkl')
# joblib.dump(tfidf_vectorizer, 'D:/SKRIPSI/tfidf_model_fix.pkl')

# === Optional: Simpan feature importance ke Excel ===
feature_importance = rf_model.feature_importances_
feature_names = tfidf_vectorizer.get_feature_names_out()
importance_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': feature_importance
}).sort_values(by='Importance', ascending=False)

# importance_df.to_excel('D:/SKRIPSI/rf_feature_importance.xlsx', index=False)
