import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from imblearn.over_sampling import SMOTE
import numpy as np
import matplotlib.pyplot as plt
import joblib
import nltk
from tqdm.auto import tqdm

nltk.download('punkt')

# === Step 1: Load Dataset ===
input_file = 'D:/SKRIPSI/labelling_fix.xlsx'
df = pd.read_excel(input_file)

if 'text_akhir' not in df.columns or 'VADER Sentiment (Binary)' not in df.columns:
    raise ValueError("Dataset harus memiliki kolom 'text_akhir' dan 'VADER Sentiment (Binary)'")

df = df.dropna(subset=['text_akhir'])
df['text_akhir'] = df['text_akhir'].astype(str)

# === Step 2: Split Data ===
X_text = df['text_akhir']
y = df['VADER Sentiment (Binary)']
X_train_text, X_test_text, y_train, y_test = train_test_split(X_text, y, test_size=0.2, random_state=42, stratify=y)

# === Step 3: TF-IDF (fit hanya ke data training) ===
print("🔍 Menghitung TF-IDF (fit ke data training)...")
tfidf_vectorizer = TfidfVectorizer(ngram_range=(1, 1), max_features=1000, min_df=7, max_df=0.7)

with tqdm(total=1, desc="TF-IDF Processing", unit="step") as pbar:
    X_train_tfidf = tfidf_vectorizer.fit_transform(X_train_text)
    X_test_tfidf = tfidf_vectorizer.transform(X_test_text)
    pbar.update(1)

# Simpan hasil TF-IDF dari training saja
tfidf_df_train = pd.DataFrame(X_train_tfidf.toarray(), columns=tfidf_vectorizer.get_feature_names_out())
tfidf_df_train['Label'] = y_train.reset_index(drop=True)
tfidf_df_train.to_excel('D:/SKRIPSI/tfidf_train_only.xlsx', index=False)
print("✅ TF-IDF training disimpan.")

# === Step 4: SMOTE ===
print("📊 Distribusi Label Sebelum SMOTE:")
print(pd.Series(y_train).value_counts(), '\n')

smote = SMOTE(random_state=42)
X_train_bal, y_train_bal = smote.fit_resample(X_train_tfidf, y_train)

print("📊 Distribusi Label Setelah SMOTE:")
print(pd.Series(y_train_bal).value_counts(), '\n')

# === Step 5: Train SVM ===
svm_model = SVC(kernel='linear', C=0.3, random_state=42)
svm_model.fit(X_train_bal, y_train_bal)

# === Step 6: Evaluasi di data testing ===
y_pred = svm_model.predict(X_test_tfidf)

print("\n📋 Classification Report (Testing):\n")
print(classification_report(y_test, y_pred, zero_division=0))

accuracy_train = accuracy_score(y_train_bal, svm_model.predict(X_train_bal))
print(f"📈 Akurasi Training: {accuracy_train:.4f}")

bias = svm_model.intercept_[0]
print(f"📉 Bias (Intercept): {bias:.4f}")

weights = svm_model.coef_.toarray()[0]
margin = 2 / np.linalg.norm(weights)
print(f"📏 Margin: {margin:.4f}")

# === Step 7: Simpan bobot fitur ===
feature_weights_df = pd.DataFrame({
    'Feature': tfidf_vectorizer.get_feature_names_out(),
    'Weight': weights
})
feature_weights_df = feature_weights_df.sort_values(by='Weight', ascending=False)
feature_weights_df.to_excel('D:/SKRIPSI/svm_feature_weights.xlsx', index=False)
print("✅ Bobot fitur SVM disimpan.")

# === Step 8: Simpan Model & Vectorizer ===
joblib.dump(svm_model, 'D:/SKRIPSI/model_svm_trained.pkl')
joblib.dump(tfidf_vectorizer, 'D:/SKRIPSI/tfidf_model_trained.pkl')
print("✅ Model dan TF-IDF vectorizer disimpan.")

# === Step 9: Confusion Matrix ===
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=svm_model.classes_)
disp.plot(cmap='Blues', values_format='d')
plt.title("Confusion Matrix - Data Testing")
plt.grid(False)
plt.show()
