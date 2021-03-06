# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

"""Functions for computing metrics."""
import numpy as np
import torch
import sklearn.metrics as sk

def topks_correct(preds, labels, ks):
    """
    Given the predictions, labels, and a list of top-k values, compute the
    number of correct predictions for each top-k value.
    Args:
        preds (array): array of predictions. Dimension is batchsize
            N x ClassNum.
        labels (array): array of labels. Dimension is batchsize N.
        ks (list): list of top-k values. For example, ks = [1, 5] correspods
            to top-1 and top-5.
    Returns:
        topks_correct (list): list of numbers, where the `i`-th entry
            corresponds to the number of top-`ks[i]` correct predictions.
    """
    assert preds.size(0) == labels.size(
        0
    ), "Batch dim of predictions and labels must match"
    # Find the top max_k predictions for each sample
    _top_max_k_vals, top_max_k_inds = torch.topk(
        preds, max(ks), dim=1, largest=True, sorted=True
    )
    # (batch_size, max_k) -> (max_k, batch_size).
    top_max_k_inds = top_max_k_inds.t()
    # (batch_size, ) -> (max_k, batch_size).
    rep_max_k_labels = labels.view(1, -1).expand_as(top_max_k_inds)
    # (i, j) = 1 if top i-th prediction for the j-th sample is correct.
    top_max_k_correct = top_max_k_inds.eq(rep_max_k_labels)
    # Compute the number of topk correct predictions for each k.
    topks_correct = [
        top_max_k_correct[:k, :].view(-1).float().sum() for k in ks
    ]
    return topks_correct


def stable_cumsum(arr, rtol=1e-05, atol=1e-08):
    """Use high precision for cumsum and check that final value matches sum
    Parameters
    ----------
    arr : array-like
        To be cumulatively summed as flat
    rtol : float
        Relative tolerance, see ``np.allclose``
    atol : float
        Absolute tolerance, see ``np.allclose``
    """
    out = np.cumsum(arr, dtype=np.float64)
    expected = np.sum(arr, dtype=np.float64)
    if not np.allclose(out[-1], expected, rtol=rtol, atol=atol):
        raise RuntimeError('cumsum was found to be unstable: '
                           'its last element does not correspond to sum')
    return out


def fpr_and_fdr_at_recall(y_true, y_score, recall_level, pos_label=None):
    classes = np.unique(y_true)
    if (pos_label is None and
            not (np.array_equal(classes, [0, 1]) or
                     np.array_equal(classes, [-1, 1]) or
                     np.array_equal(classes, [0]) or
                     np.array_equal(classes, [-1]) or
                     np.array_equal(classes, [1]))):
        raise ValueError("Data is not binary and pos_label is not specified")
    elif pos_label is None:
        pos_label = 1.

    # make y_true a boolean vector
    y_true = (y_true == pos_label)

    # sort scores and corresponding truth values
    desc_score_indices = np.argsort(y_score, kind="mergesort")[::-1]
    y_score = y_score[desc_score_indices]
    y_true = y_true[desc_score_indices]

    # y_score typically has many tied values. Here we extract
    # the indices associated with the distinct values. We also
    # concatenate a value for the end of the curve.
    distinct_value_indices = np.where(np.diff(y_score))[0]
    threshold_idxs = np.r_[distinct_value_indices, y_true.size - 1]

    # accumulate the true positives with decreasing threshold
    tps = stable_cumsum(y_true)[threshold_idxs]
    fps = 1 + threshold_idxs - tps      # add one because of zero-based indexing

    thresholds = y_score[threshold_idxs]

    recall = tps / tps[-1]

    last_ind = tps.searchsorted(tps[-1])
    sl = slice(last_ind, None, -1)      # [last_ind::-1]
    recall, fps, tps, thresholds = np.r_[recall[sl], 1], np.r_[fps[sl], 0], np.r_[tps[sl], 0], thresholds[sl]

    cutoff = np.argmin(np.abs(recall - recall_level))

    return fps[cutoff] / (np.sum(np.logical_not(y_true)))   # , fps[cutoff]/(fps[cutoff] + tps[cutoff])


def get_ood_measures(confidences, targets, recall_level=0.95):
    num_inlier = targets.size(0)
    confidences = confidences.data.cpu().numpy()
    pos = np.array(confidences[:num_inlier]).reshape((-1, 1))
    neg = np.array(confidences[num_inlier:]).reshape((-1, 1))
    examples = np.squeeze(np.vstack((pos, neg)))
    
    labels = np.ones(len(examples), dtype=np.int32)
    labels[len(pos):] -= 1
    #print(examples, labels)
    auroc = sk.roc_auc_score(labels, examples)
    aupr = sk.average_precision_score(labels, examples)
    fpr = fpr_and_fdr_at_recall(labels, examples, recall_level)

    return auroc, aupr, fpr

def get_auout(confidences, targets):
    num_inlier = targets.size(0)
    confidences = confidences.data.cpu().numpy()
    neg = np.array(confidences[:num_inlier]).reshape((-1, 1))
    pos = np.array(confidences[num_inlier:]).reshape((-1, 1))
    examples = np.squeeze(np.vstack((pos, neg)))
    
    labels = np.ones(len(examples), dtype=np.int32)
    labels[len(pos):] -= 1
    aupr = sk.average_precision_score(labels, examples)

    return aupr

def classify_acc_w_ood(logits, targets, confidences, step=1000):
    threshold_min = torch.min(confidences)
    threshold_max = torch.max(confidences)
    threshold_diff = threshold_max - threshold_min
    total = logits.size(0) 

    class_correct = (torch.argmax(logits[:len(targets)], dim=1) == targets).float()
    
    max_threshold = threshold_min
    max_acc = -1.
    for i in range(step + 1):
        threshold = threshold_min + threshold_diff * (i / step)
        inliers = (confidences >= threshold).float()
        outliers = (confidences < threshold).float()
        inlier_correct = (torch.squeeze(inliers[:len(targets)], dim=1) * class_correct[:]).sum()
        outlier_correct = outliers[len(targets):].sum()
        acc = (inlier_correct + outlier_correct) / total
        if max_acc < acc:
            max_acc = acc
            max_threshold = threshold
    
    return max_acc


# Add new metrics here!!!
def show_wrong_samples_targets(logits, targets, log):
    predicts = logits.max(dim=1).indices
    wrong_targets = ((logits.max(dim=1).indices) != targets)
    for idx, i in enumerate(wrong_targets):
        if i:
            log.write("classifier predict [{}] / Ground truth [{}]\n".format(predicts[idx], targets[idx]))
            
            
def kl_div(d1, d2):
    """
    Compute KL-Divergence between d1 and d2.
    """
    dirty_logs = d1 * torch.log2(d1 / d2)
    return torch.sum(torch.where(d1 != 0, dirty_logs, torch.zeros_like(d1)), axis=1)


def jsd(d1, d2):
    """
    Calculate Jensen-Shannon Divergence between d1 and d2
    Square-root this to get the Jensen-Shannon distance
    """
    M = 0.5 * (d1 + d2)
    return 0.5 * kl_div(d1, M) + 0.5 * kl_div(d2, M)