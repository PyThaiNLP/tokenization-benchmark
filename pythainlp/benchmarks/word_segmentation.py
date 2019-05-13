import numpy as np

SEPARATOR = "|"

def benchmark(ref_samples, samples ):
    for r, s in zip(ref_samples, samples):
        # @todo #1 compulte stats for all samples
        pass

def _compute_metrics(tp, fp, tn, fn):
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    return {
        'precision': precision,
        'recall': recall,
        'f1': 2*(precision*recall) / (precision + recall)
    }

def _compute_stats(ref_sample, sample):
    ref_sample, _ = _binary_representation(ref_sample)
    sample, _ = _binary_representation(sample)

    # Charater Level
    c_pos_pred, c_neg_pred = np.argwhere(sample==1), np.argwhere(sample==0)

    c_tp = np.sum(ref_sample[c_pos_pred] == 1)
    c_fp = np.sum(ref_sample[c_pos_pred] == 0)

    c_tn = np.sum(ref_sample[c_neg_pred] == 0)
    c_fn = np.sum(ref_sample[c_neg_pred] == 1)

    # @todo #1 compute stats for word level

    return {
        'char_level': {
            'tp': c_tp,
            'fp': c_fp,
            'tn': c_tn,
            'fn': c_fn,
            **_compute_metrics(c_tp, c_fp, c_tn, c_fn)
        }
    }

"""
ผม|ไม่|ชอบ|กิน|ผัก -> 10100...
"""
def _binary_representation(sample, verbose=False):
    chars = np.array(list(sample))
    boundary = np.argwhere(chars == SEPARATOR).reshape(-1)
    boundary = boundary - np.array(range(boundary.shape[0]))
    bin_rept = np.zeros(len(sample) - boundary.shape[0])
    bin_rept[list(boundary) + [0]] = 1

    sample_wo_seps = list(sample.replace(SEPARATOR, ""))
    assert len(sample_wo_seps) == len(bin_rept)

    if verbose:
        for c, m in zip(sample_wo_seps, bin_rept):
            print('%s -- %d' % (c, m))

    return bin_rept, sample_wo_seps