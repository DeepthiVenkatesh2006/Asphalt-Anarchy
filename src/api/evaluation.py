"""Server-side point metrics on the same approximate aligned-label subset."""
import numpy as np
from sklearn.metrics import precision_recall_fscore_support


def point_metrics(predicted, labels):
    eligible = labels[(labels.alignment_status == 'aligned') & labels.reference_arc_id.notna()]
    if eligible.empty:
        return dict(accuracy=None, macro_precision=None, macro_recall=None, macro_f1=None, observations=0)
    truth = eligible.reference_arc_id.to_numpy(dtype=int)
    inferred = np.asarray(predicted, dtype=int)[eligible.obs_idx.to_numpy(dtype=int)]
    precision, recall, f1, _ = precision_recall_fscore_support(truth, inferred, average='macro', zero_division=0)
    return dict(accuracy=float((truth == inferred).mean()), macro_precision=float(precision),
                macro_recall=float(recall), macro_f1=float(f1), observations=len(eligible))
