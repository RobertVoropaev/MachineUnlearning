from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split


def create_blobs_dataset(n_clusters, cluster_std, n_features, full_size, test_shape, seed=0):
    X_full, y_full = make_blobs(
        n_samples=full_size, 
        centers=n_clusters, 
        n_features=n_features,
        cluster_std=cluster_std,
        random_state=seed, 
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X_full, y_full, 
        test_size=test_shape, 
        random_state=seed, 
        stratify=y_full
    )
    return X_train, X_test, y_train, y_test