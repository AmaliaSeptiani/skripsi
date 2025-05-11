import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from imblearn.over_sampling import SMOTE
import joblib
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix

# Baca data hasil TF-IDF
df = pd.read_excel('D:/SKRIPSI/tfidf_fix.xlsx')

# Pisahkan fitur dan label
X = df.drop(columns=['VADER Sentiment (Binary)'])
y = df['VADER Sentiment (Binary)']

# Ubah semua nama kolom menjadi string
X.columns = X.columns.astype(str)

# Split data: 80% latih, 20% uji
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)
print("📊 Distribusi Label Data Testing:")
print(y_test.value_counts(), '\n')

# 🔍 Tampilkan distribusi label sebelum SMOTE
print("📊 Distribusi Label Sebelum SMOTE:")
print(y_train.value_counts(), '\n')

# Balancing data latih dengan SMOTE
smote = SMOTE(random_state=42)
X_train_bal, y_train_bal = smote.fit_resample(X_train, y_train)

# 🔍 Tampilkan distribusi label setelah SMOTE
print("📊 Distribusi Label Setelah SMOTE:")
print(y_train_bal.value_counts(), '\n')

# Inisialisasi dan latih model SVM
svm_model = SVC(kernel='linear', C=0.3, random_state=42)
svm_model.fit(X_train_bal, y_train_bal)

# Prediksi data uji
y_pred = svm_model.predict(X_test)

# Simpan classification report
print("\n📋 Classification Report (Testing):\n")
print(classification_report(y_test, y_pred, zero_division=0))

# Prediksi data latih
y_pred_train = svm_model.predict(X_train_bal)

# Hitung akurasi training
accuracy_train = accuracy_score(y_train_bal, y_pred_train)
print(f"📈 Akurasi Training: {accuracy_train:.4f}")

# 📊 Hitung Bias (Intercept) dan Margin
bias = svm_model.intercept_[0]
print(f"📉 Bias (Intercept): {bias:.4f}")

# Margin = 2 / ||w||, di mana w adalah bobot model SVM
weights = svm_model.coef_[0]  # Langsung ambil array pertama karena coef_ sudah berupa array NumPy
margin = 2 / np.linalg.norm(weights)
print(f"📏 Margin: {margin:.4f}")

# #  Simpan bobot fitur SVM ke file .xlsx
# feature_weights_df = pd.DataFrame({
#     'Feature': X.columns,
#     'Weight': weights
# })
# file_path = 'D:/SKRIPSI/svm_feature_weights.xlsx'
# feature_weights_df.to_excel(file_path, index=False)

# print(f"✅ Bobot fitur SVM disimpan di: {file_path}")

# Hitung confusion matrix
cm = confusion_matrix(y_test, y_pred)

# Tampilkan confusion matrix dalam bentuk grafik
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=svm_model.classes_)
disp.plot(cmap='Blues', values_format='d')
plt.title("Confusion Matrix - Data Testing")
plt.grid(False)
plt.show()

# # ✅ Simpan model ke file
# model_path = 'D:/SKRIPSI/model_svm_fix.pkl'
# joblib.dump(svm_model, model_path)
# print(f"✅ Model SVM berhasil disimpan di: {model_path}")

