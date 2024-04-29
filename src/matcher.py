import numpy as np

from collections import Counter


class LabelMatcher:
    def __init__(self, labels_true, labels_pred):
        self.rename_dict = self._get_rename_dict(labels_true, labels_pred)
        
    def _get_rename_dict(self, old_labels, new_labels):
        counter = Counter()
        for new, old in zip(new_labels, old_labels):
            counter[(new, old)] += 1

        rename_dict = {}
        for val in counter.most_common():
            new, old= val[0]
            if new not in rename_dict:
                rename_dict[new] = old
        return rename_dict
    
    def transform_labels(self, labels):
        return np.array([self.rename_dict[label] for label in labels])
        
    def transform_centers(self, centers):
        new_order = [self.rename_dict[i] for i in range(centers.shape[0])]
        return centers[new_order]