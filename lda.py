import numpy as np
import matplotlib.pyplot as plt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
def lda_numpy(X, y, n_components=1):
    classes = np.unique(y)
    n_features = X.shape[1]
    overall_mean = np.mean(X, axis=0)
    class_means = {}
    class_counts = {}
    for cls in classes:
        class_data = X[y == cls]
        class_means[cls] = np.mean(class_data, axis=0)
        class_counts[cls] = len(class_data)
    Sw = np.zeros((n_features, n_features))
    for cls in classes:
        class_data = X[y == cls]
        centered = class_data - class_means[cls]
        Sw += centered.T @ centered
    Sb = np.zeros((n_features, n_features))
    for cls in classes:
        n = class_counts[cls]
        mean_diff = (class_means[cls] - overall_mean).reshape(-1, 1)
        Sb += n * (mean_diff @ mean_diff.T)
    Sw_inv = np.linalg.pinv(Sw)
    eigenvalues, eigenvectors = np.linalg.eig(Sw_inv @ Sb)
    idx = np.argsort(eigenvalues.real)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]
    eigenvectors = eigenvectors[:, :n_components]
    eigenvalues = eigenvalues[:n_components]
    X_lda = X @ eigenvectors
    return X_lda, eigenvalues.real, eigenvectors.real
X = np.array([[1, 2], [2, 1], [1.5, 1.5], [2.5, 2],[6, 7], [7, 6], [6.5, 6.5], [7.5, 7],[3, 6], [4, 5], [3.5, 5.5], [4.5, 6]])
y = np.array([0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2])
X_lda_numpy, eigenvals, eigenvecs = lda_numpy(X, y, n_components=1)
lda = LinearDiscriminantAnalysis(n_components=1)
X_lda_sklearn = lda.fit_transform(X, y)
print("NumPy LDA:")
print("Eigenvalues:", eigenvals)
print("Transformed data:")
for i, row in enumerate(X_lda_numpy):
    print(f"  Class {y[i]}: {row}")
print("\nScikit-learn LDA:")
print("Explained variance ratio:", lda.explained_variance_ratio_)
print("Transformed data:")
for i, row in enumerate(X_lda_sklearn):
    print(f"  Class {y[i]}: {row}")
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
colors = ['red', 'blue', 'green']
for i in range(3):
    mask = y == i
    ax1.scatter(X[mask, 0], X[mask, 1], c=colors[i], label=f'Class {i}', s=80)
ax1.set_title('Original Data')
ax1.set_xlabel('Feature 1')
ax1.set_ylabel('Feature 2')
ax1.legend()
ax1.grid(True)
for i in range(3):
    mask = y == i
    ax2.scatter(X_lda_numpy[mask, 0], np.zeros_like(X_lda_numpy[mask, 0]), 
                c=colors[i], label=f'Class {i}', s=80)
ax2.set_title('LDA Projection')
ax2.set_xlabel('LD1')
ax2.set_yticks([])
ax2.legend()
ax2.grid(True)
plt.tight_layout()
plt.show()