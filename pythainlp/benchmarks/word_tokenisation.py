import numpy as np
import pandas as pd

SEPARATOR = "|"

def flatten_dict(my_dict, parent_key="", sep=":"):
    items = []
    for k, v in my_dict.items():
        new_key = "%s%s%s" % (parent_key, sep, k) if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, parent_key=new_key).items())
        else:
            items.append((new_key, v))

    return dict(items)

def benchmark(ref_samples, samples):
    results = []
    for r, s in zip(ref_samples, samples):
        stats = flatten_dict(_compute_stats(r, s))
        results.append(stats)

    return pd.DataFrame(results)

def _compute_stats(ref_sample, sample):
    ref_sample, _ = _binary_representation(ref_sample)
    sample, _ = _binary_representation(sample)

    # Charater Level
    c_pos_pred, c_neg_pred = np.argwhere(sample==1), np.argwhere(sample==0)

    c_tp = np.sum(ref_sample[c_pos_pred] == 1)
    c_fp = np.sum(ref_sample[c_pos_pred] == 0)

    c_tn = np.sum(ref_sample[c_neg_pred] == 0)
    c_fn = np.sum(ref_sample[c_neg_pred] == 1)

    precision = c_tp / (c_tp + c_fp)
    recall = c_tp / (c_tp + c_fn)
    f1 = 2*(precision*recall) / (precision + recall)

    # Word Level
    boundary = np.argwhere(ref_sample == 1).reshape(-1)
    start_idx = boundary
    stop_idx = boundary[1:].tolist() + [ref_sample.shape[0]]

    is_correctly_tokenised = []
    for st, end in zip(start_idx, stop_idx):
        if sample[st] == 1 and np.sum(sample[st+1:end]) == 0:
            is_correctly_tokenised.append(1)
        else:
            is_correctly_tokenised.append(0)

    return {
        'char_level': {
            'tp': c_tp,
            'fp': c_fp,
            'tn': c_tn,
            'fn': c_fn,
            'precision': precision,
            'recall': recall,
            'f1': f1
        },
        'word_level': {
            'accuracy':  np.sum(is_correctly_tokenised) / len(is_correctly_tokenised)
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