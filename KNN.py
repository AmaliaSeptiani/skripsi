import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler

# Baca data hasil TF-IDF
df = pd.read_excel('D:/SKRIPSI/tfidf_fix.xlsx')

# Pisahkan fitur dan label
X = df.drop(columns=['VADER Sentiment (Binary)'])
y = df['VADER Sentiment (Binary)']

# Pastikan semua nama kolom berupa string
X.columns = X.columns.astype(str)

# Split data: 80% latih, 20% uji
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Balancing data latih dengan SMOTE
smote = SMOTE(random_state=42)
X_train_bal, y_train_bal = smote.fit_resample(X_train, y_train)

# Standardisasi data (WAJIB untuk KNN)
scaler = StandardScaler()
X_train_bal = scaler.fit_transform(X_train_bal)
X_test = scaler.transform(X_test)

# Inisialisasi dan latih model KNN
knn_model = KNeighborsClassifier(n_neighbors=2) 
knn_model.fit(X_train_bal, y_train_bal)

# Prediksi data uji
y_pred = knn_model.predict(X_test)

# Hitung akurasi untuk data latih
y_train_pred = knn_model.predict(X_train_bal)
accuracy_train = accuracy_score(y_train_bal, y_train_pred)

# Evaluasi model
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, zero_division=0)
recall = recall_score(y_test, y_pred, zero_division=0)
f1 = f1_score(y_test, y_pred, zero_division=0)

# Cetak hasil evaluasi
print("📊 Hasil Evaluasi KNN:")
print(f"✅ Akurasi Training : {accuracy_train:.4f}")
print(f"✅ Akurasi Testing  : {accuracy:.4f}")
print(f"✅ Presisi          : {precision:.4f}")
print(f"✅ Recall           : {recall:.4f}")
print(f"✅ F1-Score         : {f1:.4f}")
print("\n📋 Classification Report:\n")
print(classification_report(y_test, y_pred, zero_division=0))
