from sklearn.decomposition import PCA
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def apply_pca(X, n_components=None, variance_threshold=None, plot_variance=True):
    # Convert to numeric-only data (exclude datetime columns)
    X_numeric = X.select_dtypes(include=['number'])
    
    if variance_threshold is not None:
        pca = PCA(n_components=variance_threshold, svd_solver='full')
    else:
        pca = PCA(n_components=n_components)

    X_pca_array = pca.fit_transform(X_numeric)
    X_pca = pd.DataFrame(X_pca_array, 
                       columns=[f"PC{i+1}" for i in range(X_pca_array.shape[1])])

    if plot_variance:
        plt.figure(figsize=(10, 5))
        plt.bar(range(1, len(pca.explained_variance_ratio_) + 1),
                pca.explained_variance_ratio_)
        plt.plot(np.cumsum(pca.explained_variance_ratio_), 'ro-')
        plt.axhline(y=variance_threshold if variance_threshold else 0.95, 
                   color='g', linestyle='--')
        plt.title('Explained Variance Ratio by Principal Components')
        plt.xlabel('Principal Component')
        plt.ylabel('Explained Variance Ratio')
        plt.grid(True)
        plt.show()

    return X_pca, pca