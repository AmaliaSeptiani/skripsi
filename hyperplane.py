import matplotlib.pyplot as plt
import numpy as np

# Buat grid untuk menggambar garis
x_vals = np.linspace(-5, 5, 100)

# Bobot dan bias
w = np.array([0.5, 0.5])
b = 0

# Persamaan: w.x + b = 0  =>  0.6*x1 + 0.4*x2 = 0
# x2 = -(w1/w2)*x1 untuk hyperplane dan +/-1 untuk margins

def line_eq(w, b, level):
    return -(w[0] * x_vals + b - level) / w[1]

# Hitung garis
hyperplane = line_eq(w, b, 0)
margin_pos = line_eq(w, b, 1)
margin_neg = line_eq(w, b, -1)

# Plot
plt.figure(figsize=(8, 6))
plt.plot(x_vals, hyperplane, 'k--', label='Hyperplane (w.x + b = 0)')
plt.plot(x_vals, margin_pos, 'r-', label='Margin +1 (w.x + b = 1)')
plt.plot(x_vals, margin_neg, 'b-', label='Margin -1 (w.x + b = -1)')

# Tambahkan support vectors
plt.scatter([1, -1], [1, -1], color='green', s=100, edgecolors='black', label='Support Vectors')
# Tambahkan data kelas +1
x_pos = [1, 2, 2, 3]
y_pos = [2, 3, 1, 2]
plt.scatter(x_pos, y_pos, color='red', s=80, marker='o', label='Data +1')

# Tambahkan data kelas -1
x_neg = [-1, -2, -2, -3]
y_neg = [-2, -3, -1, -2]
plt.scatter(x_neg, y_neg, color='blue', s=80, marker='o', label='Data -1')


plt.title("Visualisasi Hyperplane dan Margin SVM\n(dengan w = [0.5, 0.5], b = 0)")
plt.xlabel("x1")
plt.ylabel("x2")
plt.axhline(0, color='gray', lw=0.5)
plt.axvline(0, color='gray', lw=0.5)
plt.grid(True, linestyle='--', alpha=0.5)
plt.legend()
plt.axis('equal')
plt.xlim(-3, 3)
plt.ylim(-3, 3)
plt.show()
