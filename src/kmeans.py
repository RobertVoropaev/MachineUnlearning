import numpy as np


class Simple_kMeans:
    def __init__(
        self, 
        n_clusters, 
        tolerance = 0.01, 
        max_iter = 100,
        seed = 0,
        init_method = "random_points",
        _max = None,
        _min = None,
        _centers = None
    ):
        self.n_clusters = n_clusters
        self.tolerance = tolerance
        self.max_iter = max_iter
        self.seed = seed
        self.init_method = init_method
        self._max = _max
        self._min = _min
        self._centers = _centers
        
        
    def fit(self, X: np.ndarray):
        labels = np.zeros(X.shape[0])
        self.cluster_centers =  self._init_cluster_centers(X, method=self.init_method)
        fit_history = []
        
        for i in range(self.max_iter):            
            prev_centers = np.copy(self.cluster_centers)
            
            distances = self._compute_point_distances(X, self.cluster_centers)
            labels = distances.argmin(axis=1)

            self.cluster_centers = self._compute_cluster_centers(X, labels)

            clusters_not_changed = (
                np.linalg.norm(prev_centers - self.cluster_centers, axis=1) < self.tolerance
            )
            
            fit_history.append({
                "iter": i,
                "prev_centers": prev_centers,
                "new_centers": self.cluster_centers,
                "distances": distances,
                "labels": labels,
                "clusters_not_changed_mask": clusters_not_changed
            })
            
            if np.all(clusters_not_changed):
                break
                
        if i + 1 == self.max_iter:
            raise Exception("Max iter reached, increase max_iter param")
                
        return self.cluster_centers, labels, fit_history
    
    def predict(self, X: np.ndarray):
        distances = self._compute_point_distances(X, self.cluster_centers)
        labels = distances.argmin(axis=1)
        return labels
      
    def _init_cluster_centers(self, X: np.ndarray, method: str):
        np.random.seed(self.seed)
        match method:
            case "random_points":
                return X[np.random.choice(X.shape[0], size=self.n_clusters, replace=False)]
            case 'random_min_max':
                _min = X.min(axis=0) if self._min is None else self._min
                _max = X.max(axis=0) if self._max is None else self._max

                cluster_centers = [
                    np.random.uniform(low=_min, high=_max)
                    for i in range(self.n_clusters)
                ]
                return np.array(cluster_centers)
            case "fixed_centers":
                if self._centers is None:
                    raise Exception(f"Param _centers not specified")
                return self._centers
            case _:
                raise Exception(f"Unknown method: {method}")
        
    def _compute_point_distances(self, X: np.ndarray, cluster_centers: np.ndarray):
        distances = np.zeros((X.shape[0], self.n_clusters))
        for cluster_center_index, cluster_center in enumerate(cluster_centers):
            distances[:, cluster_center_index] = np.linalg.norm(X - cluster_center, axis=1)
        return distances
    
    def _compute_cluster_centers(self, X: np.ndarray, labels: np.ndarray):
        cluster_centers = np.zeros((self.n_clusters, X.shape[1]))
        for cluster_center_index in range(self.n_clusters):
            cluster_elements = X[labels == cluster_center_index]
            if len(cluster_elements):
                cluster_centers[cluster_center_index, :] = cluster_elements.mean(axis=0)     
        return cluster_centers