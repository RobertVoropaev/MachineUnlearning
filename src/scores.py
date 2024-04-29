import pandas as pd
import numpy as np

from matcher import LabelMatcher

from sklearn.metrics import (
    adjusted_rand_score as ari_score, 
    accuracy_score as acc_score,
    normalized_mutual_info_score as nmi_score,
    f1_score 
)

def score_calculator(labels_1, labels_2, match_labels = True):
    if match_labels: 
        matcher = LabelMatcher(labels_1, labels_2)
        labels_2 = matcher.transform_labels(labels_2)
        
    
    df_scores = pd.DataFrame(
        columns=['ACC', 'F1', 'ARI', 'NMI', 'classes']
    )
    
    acc_origin = acc_score(labels_1, labels_2)
    f1_origin = f1_score(labels_1, labels_2, average='weighted')
    ari_origin = ari_score(labels_1, labels_2)
    nmi_origin = nmi_score(labels_1, labels_2)
    
    classes_1 = len(np.unique(labels_1))
    classes_2 = len(np.unique(labels_2))
    assert classes_1 == classes_2, "Different number of classes in true and pred"
    
    df_scores.loc[0] = [acc_origin, f1_origin, ari_origin, nmi_origin, classes_1]
    return df_scores