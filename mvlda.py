import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

class MvDA:
    def __init__(self, n_components=1):
        self.n_components = n_components
        self.transforms = None
        self.eigenvalues = None
        
    def fit(self, X_views, y):
        self.n_views = len(X_views)
        self.classes = np.unique(y)
        self.n_classes = len(self.classes)
        dimensions = [X.shape[1] for X in X_views]
        S, D = self._build_scatter_matrices(X_views, y)
        try:
            S_inv = np.linalg.inv(S)
        except np.linalg.LinAlgError:
            S_inv = np.linalg.pinv(S)
            
        eigenvals, eigenvecs = np.linalg.eig(S_inv @ D)
        idx = np.argsort(eigenvals.real)[::-1]
        eigenvals = eigenvals[idx]
        eigenvecs = eigenvecs[:, idx]
        self.n_components = min(self.n_components, self.n_classes - 1)
        self.eigenvalues = eigenvals[:self.n_components].real
        self.transforms = []
        start_idx = 0
        for dim in dimensions:
            end_idx = start_idx + dim
            view_transform = eigenvecs[start_idx:end_idx, :self.n_components].real
            self.transforms.append(view_transform)
            start_idx = end_idx
            
        return self
    
    def _build_scatter_matrices(self, X_views, y):
        total_dim = sum(X.shape[1] for X in X_views)
        S = np.zeros((total_dim, total_dim))
        D = np.zeros((total_dim, total_dim))
        class_means = {}
        class_counts = {}
        overall_mean = np.zeros(total_dim)
        n_total = 0
        
        for cls in self.classes:
            class_means[cls] = []
            class_counts[cls] = []
            
            for view_idx, X in enumerate(X_views):
                mask = y == cls
                if np.any(mask):
                    class_data = X[mask]
                    class_mean = np.mean(class_data, axis=0)
                    class_means[cls].append(class_mean)
                    class_counts[cls].append(len(class_data))
                else:
                    class_means[cls].append(np.zeros(X.shape[1]))
                    class_counts[cls].append(0)
        start_idx = 0
        for view_idx, X in enumerate(X_views):
            end_idx = start_idx + X.shape[1]
            overall_mean[start_idx:end_idx] = np.mean(X, axis=0)
            n_total += len(X)
            start_idx = end_idx
        for i in range(self.n_views):
            for j in range(self.n_views):
                start_i = sum(X_views[k].shape[1] for k in range(i))
                end_i = start_i + X_views[i].shape[1]
                start_j = sum(X_views[k].shape[1] for k in range(j))
                end_j = start_j + X_views[j].shape[1]
                
                S_ij = np.zeros((X_views[i].shape[1], X_views[j].shape[1]))
                
                for cls in self.classes:
                    n_i = class_counts[cls][i]
                    n_j = class_counts[cls][j]
                    
                    if n_i > 0 and n_j > 0:
                        if i == j:
                            mask = y == cls
                            X_centered = X_views[i][mask] - class_means[cls][i]
                            S_ij += X_centered.T @ X_centered
                        else:
                            n_cls = sum(class_counts[cls])
                            if n_cls > 0:
                                S_ij -= (n_i * n_j / n_cls) * np.outer(class_means[cls][i], class_means[cls][j])
                
                S[start_i:end_i, start_j:end_j] = S_ij
        
        for i in range(self.n_views):
            for j in range(self.n_views):
                start_i = sum(X_views[k].shape[1] for k in range(i))
                end_i = start_i + X_views[i].shape[1]
                start_j = sum(X_views[k].shape[1] for k in range(j))
                end_j = start_j + X_views[j].shape[1]
                
                D_ij = np.zeros((X_views[i].shape[1], X_views[j].shape[1]))
                
                for cls in self.classes:
                    n_i = class_counts[cls][i]
                    n_j = class_counts[cls][j]
                    n_cls = sum(class_counts[cls])
                    
                    if n_cls > 0:
                        D_ij += (n_i * n_j / n_cls) * np.outer(class_means[cls][i], class_means[cls][j])
                
                overall_mean_i = overall_mean[start_i:end_i]
                overall_mean_j = overall_mean[start_j:end_j]
                
                total_i = sum(class_counts[cls][i] for cls in self.classes)
                total_j = sum(class_counts[cls][j] for cls in self.classes)
                
                if total_i > 0 and total_j > 0:
                    D_ij -= np.outer(overall_mean_i, overall_mean_j)
                
                D[start_i:end_i, start_j:end_j] = D_ij
        
        return S, D
    
    def transform(self, X_views):
        if self.transforms is None:
            raise ValueError("Model not fitted yet")
        
        transformed_data = []
        for view_idx, X in enumerate(X_views):
            X_transformed = X @ self.transforms[view_idx]
            transformed_data.append(X_transformed)
        
        return np.hstack(transformed_data)
    
    def fit_transform(self, X_views, y):
        self.fit(X_views, y)
        return self.transform(X_views)

def generate_multiview_data(n_samples=200, n_classes=3, noise=0.1):
    np.random.seed(42)
    
    X_base, y = make_classification(n_samples=n_samples, n_features=4, 
                                   n_informative=3, n_redundant=0, 
                                   n_classes=n_classes, n_clusters_per_class=1,
                                   random_state=42)
    
    views = []
    
    view1 = X_base + np.random.normal(0, noise, X_base.shape)
    views.append(view1)
    
    T2 = np.random.randn(4, 3)
    view2 = X_base @ T2 + np.random.normal(0, noise, (n_samples, 3))
    views.append(view2)
    
    T3 = np.random.randn(4, 5)
    view3 = X_base @ T3 + np.random.normal(0, noise, (n_samples, 5))
    views.append(view3)
    
    return views, y

if __name__ == "__main__":
    X_views, y = generate_multiview_data(n_samples=150, n_classes=3)
    
    print("Multi-View Data:")
    for i, X in enumerate(X_views):
        print(f"View {i+1} shape: {X.shape}")
    print(f"Labels shape: {y.shape}")
    print(f"Number of classes: {len(np.unique(y))}")
    
    train_views = []
    test_views = []
    for X in X_views:
        X_train, X_test, _, _ = train_test_split(X, y, test_size=0.3, random_state=42)
        train_views.append(X_train)
        test_views.append(X_test)
    
    _, _, y_train, y_test = train_test_split(X_views[0], y, test_size=0.3, random_state=42)
    mvda = MvDA(n_components=2)
    X_train_mvda = mvda.fit_transform(train_views, y_train)
    X_test_mvda = mvda.transform(test_views)
    
    print(f"\nMvDA Results:")
    print(f"Eigenvalues: {mvda.eigenvalues}")
    print(f"Transformed training data shape: {X_train_mvda.shape}")
    print(f"Transformed test data shape: {X_test_mvda.shape}")
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    colors = ['red', 'blue', 'green']
    
    for i, cls in enumerate(np.unique(y_train)):
        mask = y_train == cls
        axes[0, 0].scatter(train_views[0][mask, 0], train_views[0][mask, 1], 
                          c=colors[i], label=f'Class {cls}', alpha=0.7)
    axes[0, 0].set_title('Original View 1 (First 2D)')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    for i, cls in enumerate(np.unique(y_train)):
        mask = y_train == cls
        axes[0, 1].scatter(train_views[1][mask, 0], train_views[1][mask, 1], 
                          c=colors[i], label=f'Class {cls}', alpha=0.7)
    axes[0, 1].set_title('Original View 2 (First 2D)')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    for i, cls in enumerate(np.unique(y_train)):
        mask = y_train == cls
        axes[1, 0].scatter(X_train_mvda[mask, 0], X_train_mvda[mask, 1], 
                          c=colors[i], label=f'Class {cls}', alpha=0.7)
    axes[1, 0].set_title('MvDA Transformed (Training)')
    axes[1, 0].set_xlabel('LD1')
    axes[1, 0].set_ylabel('LD2')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    for i, cls in enumerate(np.unique(y_test)):
        mask = y_test == cls
        axes[1, 1].scatter(X_test_mvda[mask, 0], X_test_mvda[mask, 1], 
                          c=colors[i], label=f'Class {cls}', alpha=0.7)
    axes[1, 1].set_title('MvDA Transformed (Test)')
    axes[1, 1].set_xlabel('LD1')
    axes[1, 1].set_ylabel('LD2')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    def classify_nearest_centroid(X_train, y_train, X_test):
        centroids = {}
        for cls in np.unique(y_train):
            mask = y_train == cls
            centroids[cls] = np.mean(X_train[mask], axis=0)
        
        predictions = []
        for x in X_test:
            distances = {cls: np.linalg.norm(x - centroid) 
                        for cls, centroid in centroids.items()}
            predictions.append(min(distances.keys(), key=lambda k: distances[k]))
        
        return np.array(predictions)
    
    y_pred = classify_nearest_centroid(X_train_mvda, y_train, X_test_mvda)
    accuracy = np.mean(y_pred == y_test)
    
    print(f"\nClassification Results:")
    print(f"Test accuracy: {accuracy:.4f}")
    print(f"Predicted classes: {np.unique(y_pred)}")
    print(f"True classes: {np.unique(y_test)}")