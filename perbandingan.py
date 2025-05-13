import matplotlib.pyplot as plt
import numpy as np

# Nama model
models = ['SVM', 'Random Forest', 'KNN']

# Akurasi training dan testing (dalam persen)
training_accuracy = [84, 100, 82]
testing_accuracy = [75, 75, 62]

# Posisi bar di sumbu x
x = np.arange(len(models))
width = 0.35  # Lebar bar

# Membuat bar chart
fig, ax = plt.subplots(figsize=(8, 6))
bar1 = ax.bar(x - width/2, training_accuracy, width, label='Training Accuracy', color='skyblue')
bar2 = ax.bar(x + width/2, testing_accuracy, width, label='Testing Accuracy', color='salmon')

# Label dan judul
ax.set_ylabel('Accuracy (%)')
ax.set_title('Perbandingan Akurasi Training dan Testing')
ax.set_xticks(x)
ax.set_xticklabels(models)
ax.set_ylim(0, 110)
ax.legend()

# Menambahkan nilai akurasi di atas bar
for bars in [bar1, bar2]:
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height}%',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom')

plt.tight_layout()
plt.show()
