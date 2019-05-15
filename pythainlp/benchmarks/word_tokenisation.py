import re
import numpy as np
import pandas as pd

SEPARATOR = "|"


def _f1(precision, recall):
    return 2*precision*recall / (precision + recall)

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
    for i, (r, s) in enumerate(zip(ref_samples, samples)):
        stats = flatten_dict(_compute_stats(r, s))
        results.append(stats)

    return pd.DataFrame(results)

def preprocessing(sample):
    # prevent tailing separator and <NE></NE> tag
    sample = re.sub(
        re.compile("{sep}? ?{sep}$".format(sep=re.escape(SEPARATOR))),
        "",
        sample
    )

    sample = re.sub(
        re.compile("^{sep}? ?{sep}".format(sep=re.escape(SEPARATOR))),
        "",
        sample
    )

    sample = re.sub(
        "\s+",
        "",
        sample
    )

    sample = re.sub(
        re.compile("{sep}+".format(sep=re.escape(SEPARATOR))),
        SEPARATOR,
        sample
    )

    sample = re.sub(r"<\/?[A-Z]+>", "", sample)

    return sample

def _compute_stats(ref_sample, sample):
    ref_sample, _ = _binary_representation(ref_sample)
    sample, _ = _binary_representation(sample)

    # Charater Level
    c_pos_pred, c_neg_pred = np.argwhere(sample==1), np.argwhere(sample==0)

    c_pos_pred = c_pos_pred[c_pos_pred < ref_sample.shape[0]]
    c_neg_pred = c_neg_pred[c_neg_pred < ref_sample.shape[0]]

    c_tp = np.sum(ref_sample[c_pos_pred] == 1)
    c_fp = np.sum(ref_sample[c_pos_pred] == 0)

    c_tn = np.sum(ref_sample[c_neg_pred] == 0)
    c_fn = np.sum(ref_sample[c_neg_pred] == 1)

    c_precision = c_tp / (c_tp + c_fp)
    c_recall = c_tp / (c_tp + c_fn)
    c_f1 = _f1(c_precision, c_recall)

    # Word Level
    word_boundaries = _find_word_boudaries(ref_sample)
    correctly_tokenised_words = _count_correctly_tokenised_words(
        sample,
        word_boundaries
    )

    w_precision = correctly_tokenised_words / np.sum(sample)
    w_recall = correctly_tokenised_words / np.sum(ref_sample)
    w_f1 = _f1(w_precision, w_recall)

    return {
        'char_level': {
            'tp': c_tp,
            'fp': c_fp,
            'tn': c_tn,
            'fn': c_fn,
            'precision': c_precision,
            'recall': c_recall,
            'f1': c_f1
        },
        'word_level': {
            'precision':  w_precision,
            'recall':  w_recall,
            'f1': w_f1
        }
    }

"""
ผม|ไม่|ชอบ|กิน|ผัก -> 10100...
"""
def _binary_representation(sample, verbose=False):
    sample = preprocessing(sample)
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

"""
sample: a binary representation
return array of (start, stop) indicating starting and ending position of each word
"""
def _find_word_boudaries(sample):
    boundary = np.argwhere(sample == 1).reshape(-1)
    start_idx = boundary
    stop_idx = boundary[1:].tolist() + [sample.shape[0]]

    return zip(start_idx, stop_idx)

"""
sample: a binary representation
word_boundaries: [ (start, stop), ... ]
"""
def _count_correctly_tokenised_words(sample, word_boundaries):
    count = 0
    for st, end in word_boundaries:
        pend = min(end, sample.shape[0])
        if ( sample[st] == 1 and np.sum(sample[st+1:pend]) == 0 ) \
            and (
                ( pend == sample.shape[0] ) or
                ( pend != sample.shape[0] and sample[pend] == 1 )
            ):
            count = count + 1

    return count