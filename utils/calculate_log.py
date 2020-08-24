from __future__ import print_function
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import numpy as np
import time
from scipy import misc
import sklearn.metrics as sk



import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from config import cfg
import os


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

def get_curve(dir_name,split, stypes, adv_noise, random_noise_size):
    tp, fp = dict(), dict()
    tnr_at_tpr95 = dict()
    for stype in stypes:
        known = np.loadtxt(os.path.join(dir_name,'confidence_TMP_In_{}.txt'.format(split)), delimiter='\n')
        novel = np.loadtxt(os.path.join(dir_name,'confidence_TMP_Out_{}.txt'.format(split)), delimiter='\n')
    known.sort()
    novel.sort()

    num_k = known.shape[0]
    num_n = novel.shape[0]
    tp[stype] = -np.ones([num_k+num_n+1], dtype=int)
    fp[stype] = -np.ones([num_k+num_n+1], dtype=int)
    tp[stype][0], fp[stype][0] = num_k, num_n
    k, n = 0, 0
    for l in range(num_k+num_n):
        if k == num_k:
            tp[stype][l+1:] = tp[stype][l]
            fp[stype][l+1:] = np.arange(fp[stype][l]-1, -1, -1)
            break
        elif n == num_n:
            tp[stype][l+1:] = np.arange(tp[stype][l]-1, -1, -1)
            fp[stype][l+1:] = fp[stype][l]
            break
        else:
            if novel[n] < known[k]:
                n += 1
                tp[stype][l+1] = tp[stype][l]
                fp[stype][l+1] = fp[stype][l] - 1
            else:
                k += 1
                tp[stype][l+1] = tp[stype][l] - 1
                fp[stype][l+1] = fp[stype][l]
    tpr95_pos = np.abs(tp[stype] / num_k - .95).argmin()
    tnr_at_tpr95[stype] = 1. - fp[stype][tpr95_pos] / num_n
    return tp, fp, tnr_at_tpr95

def metric(dir_name, split, stypes, verbose, adv_noise, random_noise_size):
    tp, fp, tnr_at_tpr95 = get_curve(dir_name, split, stypes, adv_noise, random_noise_size)
    results = dict()
    mtypes = ['TNR', 'AUROC', 'DTACC', 'AUIN', 'AUOUT']
    if verbose:
        print('      ', end='')
        for mtype in mtypes:
            print(' {mtype:6s}'.format(mtype=mtype), end='')
        print('')
        
    for stype in stypes:
        if verbose:
            print('{stype:5s} '.format(stype=stype), end='')
        results[stype] = dict()
        
        # TNR
        mtype = 'TNR'
        results[stype][mtype] = tnr_at_tpr95[stype]
        if verbose:
            print(' {val:6.3f}'.format(val=100.*results[stype][mtype]), end='')
        
        # AUROC
        mtype = 'AUROC'
        tpr = np.concatenate([[1.], tp[stype]/tp[stype][0], [0.]])
        fpr = np.concatenate([[1.], fp[stype]/fp[stype][0], [0.]])
        results[stype][mtype] = -np.trapz(1.-fpr, tpr)
        if verbose:
            print(' {val:6.3f}'.format(val=100.*results[stype][mtype]), end='')
        
        # DTACC
        mtype = 'DTACC'
        results[stype][mtype] = .5 * (tp[stype]/tp[stype][0] + 1.-fp[stype]/fp[stype][0]).max()
        if verbose:
            print(' {val:6.3f}'.format(val=100.*results[stype][mtype]), end='')
        
        # AUIN
        mtype = 'AUIN'
        denom = tp[stype]+fp[stype]
        denom[denom == 0.] = -1.
        pin_ind = np.concatenate([[True], denom > 0., [True]])
        pin = np.concatenate([[.5], tp[stype]/denom, [0.]])
        results[stype][mtype] = -np.trapz(pin[pin_ind], tpr[pin_ind])
        if verbose:
            print(' {val:6.3f}'.format(val=100.*results[stype][mtype]), end='')
        
        # AUOUT
        mtype = 'AUOUT'
        denom = tp[stype][0]-tp[stype]+fp[stype][0]-fp[stype]
        denom[denom == 0.] = -1.
        pout_ind = np.concatenate([[True], denom > 0., [True]])
        pout = np.concatenate([[0.], (fp[stype][0]-fp[stype])/denom, [.5]])
        results[stype][mtype] = np.trapz(pout[pout_ind], 1.-fpr[pout_ind])
        if verbose:
            print(' {val:6.3f}'.format(val=100.*results[stype][mtype]), end='')
            print('')
    
    return results

def get_histogram(dir_name,split, stypes, adv_noise, random_noise_size):
    for stype in stypes:
        known = np.loadtxt(os.path.join(dir_name,'confidence_TMP_In_{}.txt'.format(split)), delimiter='\n')
        novel = np.loadtxt(os.path.join(dir_name,'confidence_TMP_Out_{}.txt'.format(split)), delimiter='\n')
    known.sort()
    novel.sort()

    end = np.max([np.max(known), np.max(novel)])
    start = np.min([np.min(known),np.min(novel)])
    outf = dir_name
    fig=plt.figure()
    ranges = (start, end)#(start, end)
    sns.distplot(known.ravel(), hist_kws={'range': ranges}, kde=False, bins=50, norm_hist=True, label='In-distribution')
    sns.distplot(novel.ravel(), hist_kws={'range': ranges}, kde=False, bins=50, norm_hist=True, label='Out-of-distribution')
#     plt.xlim(-0.01, 0)
    plt.xlabel('Confidence')
    plt.ylabel('Density')
    plt.title('In/Out='+cfg['in_dataset']['dataset']+'/'+cfg['out_dataset']['dataset']+',    Mahalanobis')
    fig.legend()
    fig.savefig(os.path.join(outf,'Mahalanobis_trained_with_ADV_%s_%s_%s.png')%(split,str(adv_noise),str(random_noise_size)))
    plt.close(fig)


def new_metric(dir_name,split, stypes, adv_noise, random_noise_size):
    for stype in stypes:
        known = np.loadtxt(os.path.join(dir_name,'confidence_TMP_In_{}.txt'.format(split)), delimiter='\n')
        novel = np.loadtxt(os.path.join(dir_name,'confidence_TMP_Out_{}.txt'.format(split)), delimiter='\n')
    known.sort()
    novel.sort()
    pos = np.array(known).reshape((-1, 1))
    neg = np.array(novel).reshape((-1, 1))
    examples = np.squeeze(np.vstack((pos, neg)))
    
    labels = np.ones(len(examples), dtype=np.int32)
    labels[len(pos):] -= 1
    auroc = sk.roc_auc_score(labels, examples)
    aupr = sk.average_precision_score(labels, examples)
    fpr = fpr_and_fdr_at_recall(labels, examples, recall_level=0.95)
    
    tp, fp, tnr_at_tpr95 = get_curve(dir_name, split, stypes, adv_noise, random_noise_size)

    results = dict()
    for stype in stypes:
        results[stype]=dict()
        results[stype]['AUROC']=auroc
        results[stype]['AUPR']=aupr
        results[stype]['TNR']=tnr_at_tpr95[stype]
        results[stype]['DTACC']=.5 * (tp[stype]/tp[stype][0] + 1.-fp[stype]/fp[stype][0]).max()
    return results


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

    