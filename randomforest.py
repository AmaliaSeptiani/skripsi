import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from imblearn.over_sampling import SMOTE

# Baca data hasil TF-IDF
df = pd.read_excel('D:/SKRIPSI/tfidf_fix.xlsx')

# Pisahkan fitur dan label
X = df.drop(columns=['VADER Sentiment (Binary)'])
y = df['VADER Sentiment (Binary)']

# Pastikan nama kolom berupa string
X.columns = X.columns.astype(str)

# Split data: 80% latih, 20% uji
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Balancing data latih dengan SMOTE
smote = SMOTE(random_state=42)
X_train_bal, y_train_bal = smote.fit_resample(X_train, y_train)

# Inisialisasi dan latih model Random Forest
rf_model = RandomForestClassifier(n_estimators=300, random_state=42)
rf_model.fit(X_train_bal, y_train_bal)

# Prediksi data training dan testing
y_pred_train = rf_model.predict(X_train_bal)
y_pred = rf_model.predict(X_test)

# Evaluasi model
accuracy_train = accuracy_score(y_train_bal, y_pred_train)
accuracy_test = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, zero_division=0)
recall = recall_score(y_test, y_pred, zero_division=0)
f1 = f1_score(y_test, y_pred, zero_division=0)

# Cetak hasil evaluasi
print("📊 Hasil Evaluasi Random Forest:")
print(f"📈 Akurasi Training : {accuracy_train:.4f}")
print("\n📋 Classification Report:\n")
print(classification_report(y_test, y_pred, zero_division=0))
