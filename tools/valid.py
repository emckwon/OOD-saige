import os
import torch
import argparse
import shutil
import time
import torch.backends.cudnn as cudnn
from tqdm import tqdm
import sys
sys.path.append('./')
import utils.losses as losses
import utils.detectors as detectors
import utils.metrics as metrics
import utils.optimizer as optim
from models.model_builder import getModel
from datasets.data_loader import getDataLoader
from utils.pgd_attack import LinfPGDAttack
from config import cfg
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

global global_cfg
global_cfg = dict()


def summary_write(summary, writer):
    for key in summary.keys():
        if key in ['logits', 'targets']:
            continue
        writer.add_scalar(key, summary[key], summary['epoch'])
        
    
def valid_epoch_wo_outlier(model, in_loader, loss_func, cur_epoch, logfile2, attack_in=None):
    global global_cfg  
    model.eval()
    avg_loss = 0
    correct = 0
    total = 0
    features_list = []
    targets_list = []
    
    in_data_size = len(in_loader.dataset)
    for cur_iter, in_set in enumerate(in_loader):        
        # Data to GPU
        data = in_set[0]
        targets = in_set[1].cuda()
        
        if attack_in is not None:
            adv_inputs = attack_in.perturb(data.cuda(), targets)
            data = adv_inputs.cuda()
        else:
            data, targets = data.cuda(), targets.cuda()
            # Foward propagation and Calculate loss
        
        (logits, features) = model(data)
        features_list.append(features.data.cpu())
        targets_list.append(targets.data.cpu())
        global_cfg['loss']['model'] = model
        global_cfg['loss']['data'] = data
        loss_dict = loss_func(logits, targets, global_cfg['loss'])
        loss = loss_dict['loss']
        
        # Calculate classifier error about in-distribution sample
        num_topks_correct = metrics.topks_correct(logits[:len(targets)], targets, (1,))
        [top1_correct] = [x for x in num_topks_correct]
        
        # Add additional metrics!!!
        metrics.show_wrong_samples_targets(logits[:len(targets)], targets, logfile2)
        
        loss, top1_correct = loss.item(), top1_correct.item()
        avg_loss += loss
        correct += top1_correct
        total += targets.size(0)
    
    logfile2.write("Dataset size [{}] | Total samples [{}] | Correct samples [{}]\n".format(in_data_size, total, correct))
    summary = {
        'avg_loss': avg_loss / in_data_size,
        'classifier_acc': correct / in_data_size,
        'epoch': cur_epoch,
        'logits' : torch.cat(features_list, dim=0),
        'targets': torch.cat(targets_list, dim=0), # (Bs,)
    }
    
    return summary
    

def valid_epoch_w_outlier(model, in_loader, out_loader, loss_func, detector_func, cur_epoch, logfile2, attack_in=None, attack_out=None):
    model.eval()
    global global_cfg  
    avg_loss = 0
    correct = 0
    total = 0
    max_iter = 0
    avg_auroc = 0
    avg_aupr = 0
    avg_fpr = 0
    inlier_conf = 0
    outlier_conf = 0
    avg_acc = 0
    in_data_size = len(in_loader.dataset)
    inliers_conf = []
    outliers_conf = []
    features_list = []
    targets_list = []
    for cur_iter, (in_set, out_set) in enumerate(zip(in_loader, out_loader)):        
        in_data = in_set[0]
        out_data = out_set[0]
        targets = in_set[1].cuda()
        
        if attack_in is not None:
            adv_inputs = attack_in.perturb(in_data.cuda(), targets)
            in_data = adv_inputs.cuda()
        
        if attack_out is not None:
            adv_inputs = attack_out.perturb(out_data.cuda())
            out_data = adv_inputs.cuda()
        
        # Data to GPU
        data = torch.cat((in_data, out_data), 0)
        data = data.cuda()
        #print("in {} out {}".format(in_set[0].size(), out_set[0].size()))
        # Foward propagation and Calculate loss and confidence
        (logits, features) = model(data)
        features_list.append(features.data.cpu())
        ood_num = features.size(0) - len(targets)
        ood_targets = -torch.ones(ood_num)
        save_targets = torch.cat((targets.data.cpu(), ood_targets.type(torch.LongTensor)), 0)
        targets_list.append(save_targets)
        print(targets)
        global_cfg['loss']['model'] = model
        global_cfg['loss']['data'] = data
        global_cfg['detector']['model'] = model
        global_cfg['detector']['data'] = data
        loss_dict = loss_func(logits, targets, global_cfg['loss'])
        loss = loss_dict['loss']
        confidences_dict = detector_func(logits, targets, global_cfg['detector'])
        confidences = confidences_dict['confidences']
        
        
        ## METRICS ##
        # Calculate classifier error about in-distribution sample
        num_topks_correct = metrics.topks_correct(logits[:len(targets)], targets, (1,))
        [top1_correct] = [x for x in num_topks_correct]
        
        # Calculate OOD metrics (auroc, aupr, fpr)
        (auroc, aupr, fpr) = metrics.get_ood_measures(confidences, targets)
        
        # Add additional metrics!!!
        metrics.show_wrong_samples_targets(logits[:len(targets)], targets, logfile2)
        acc = metrics.classify_acc_w_ood(logits, targets, confidences)
        
        ## Update stats ##
        loss, top1_correct = loss.item(), top1_correct.item()
        avg_loss += loss
        correct += top1_correct
        total += targets.size(0)
        max_iter += 1
        avg_auroc += auroc
        avg_aupr += aupr
        avg_fpr += fpr
        inlier_conf += confidences_dict['inlier_mean']
        outlier_conf += confidences_dict['outlier_mean']
        inliers_conf.append(confidences[:len(targets)].squeeze(1).data.cpu())
        outliers_conf.append(confidences[len(targets):].squeeze(1).data.cpu())
        avg_acc += acc
        
    
    summary = {
        'avg_loss': avg_loss / total,
        'classifier_acc': correct / total,
        'AUROC': avg_auroc / max_iter,
        'AUPR' : avg_aupr / max_iter,
        'FPR95': avg_fpr / max_iter,
        'inlier_confidence': inlier_conf / max_iter,
        'outlier_confidence' : outlier_conf / max_iter,
        'inliers' : torch.cat(inliers_conf).numpy(),
        'outliers': torch.cat(outliers_conf).numpy(),
        'acc': avg_acc / max_iter,
        'epoch': cur_epoch,
        'logits' : torch.cat(features_list, dim=0),
        'targets': torch.cat(targets_list, dim=0), # (Bs,)
    }
    
    return summary
    
def main():
    # Reproducibility
    np.random.seed(cfg['seed'])
    torch.manual_seed(cfg['seed'])
    
    # Model & Optimizer
    model = getModel(cfg['model'])
    start_epoch = 1
    max_epoch = 1
    
    # Load model and optimizer
    if cfg['load_ckpt'] != '':
        checkpoint = torch.load(cfg['load_ckpt'], map_location="cpu")
        model.load_state_dict(checkpoint['model_state'])
        print("load model on '{}' is complete.".format(cfg['load_ckpt']))
    cudnn.benchmark = True
    
    # Data Loader
    in_valid_loader = getDataLoader(ds_cfg=cfg['in_dataset'],
                                    dl_cfg=cfg['dataloader'],
                                    split="valid")
    attack_in = None
    if cfg['PGD'] is not None:
        print("Setup inlier PGD attack module.")
        attack_in = LinfPGDAttack(model=model, eps=cfg['PGD']['epsilon'],
                              nb_iter=cfg['PGD']['iters'],
                              eps_iter=cfg['PGD']['iter_size'],
                              rand_init=True, loss_func='CE')
    
    if cfg['out_dataset'] is not None:
        out_valid_loader = getDataLoader(ds_cfg=cfg['out_dataset'],
                                         dl_cfg=cfg['dataloader'],
                                         split="valid")
        exp_dir = os.path.join(cfg['exp_root'], cfg['exp_dir'], "valid", cfg['out_dataset']['dataset'])
        attack_out = None
        if cfg['PGD'] is not None:
            print("Setup outlier PGD attack module.")
            attack_out = LinfPGDAttack(model=model, eps=cfg['PGD']['epsilon'],
                              nb_iter=cfg['PGD']['iters'],
                              eps_iter=cfg['PGD']['iter_size'],
                              rand_init=True, loss_func='OE')
    else:
        out_train_loader = None
        out_valid_loader = None
        exp_dir = os.path.join(cfg['exp_root'], cfg['exp_dir'], "valid", "classifier")
        
    
    # Result directory and make tensorboard event file
    if os.path.exists(exp_dir) is False:
        os.makedirs(exp_dir)    
    shutil.copy('./config.py', os.path.join(exp_dir, "val_config.py"))
    
    # Loss function
    loss_func = losses.getLoss(cfg['loss'])
    global_cfg['loss'] = cfg['loss']
    
    # Outlier detector
    detector_func = detectors.getDetector(cfg['detector'])  
    global_cfg['detector'] = cfg['detector']
    print("=======================IMPORTANT CONFIG=======================")
    print(" Model    : {}\n \
Loss     : {}\n \
Detector : {}\n".format(cfg['model']['network_kind'], cfg['loss']['loss'], cfg['detector']['detector']))
    print("========Start validation. Result will be saved in {}".format(exp_dir))
    
    logfile = open(os.path.join(exp_dir, "validation_log.txt"), "w")
    logfile2 = open(os.path.join(exp_dir, "wrong_predict_log.txt"), "w")
    for cur_epoch in range(start_epoch, max_epoch + 1):
        if out_valid_loader is not None:
            valid_summary = valid_epoch_w_outlier(model, in_valid_loader,
                                                  out_valid_loader, loss_func,
                                                  detector_func, cur_epoch, logfile2, attack_in, attack_out)
            summary_log = "=============Epoch [{}]/[{}]=============\nloss: {} | acc: {} | acc_w_ood: {}\nAUROC: {} | AUPR: {} | FPR95: {}\nInlier Conf. {} | Outlier Conf. {}\n".format(cur_epoch, max_epoch, valid_summary['avg_loss'], valid_summary['classifier_acc'], valid_summary['acc'], valid_summary['AUROC'], valid_summary['AUPR'], valid_summary['FPR95'], valid_summary['inlier_confidence'], valid_summary['outlier_confidence'])
            
            ind_max, ind_min = np.max(valid_summary['inliers']), np.min(valid_summary['inliers'])
            ood_max, ood_min = np.max(valid_summary['outliers']), np.min(valid_summary['outliers'])


            ranges = (ind_min if ind_min < ood_min else ood_min,
                      ind_max if ind_max > ood_max else ood_max)
            
            fig=plt.figure()
            sns.distplot(valid_summary['inliers'].ravel(), hist_kws={'range': ranges}, kde=False, bins=50, norm_hist=True, label='In-distribution')
            sns.distplot(valid_summary['outliers'], hist_kws={'range': ranges}, kde=False, bins=50, norm_hist=True, label='Out-of-distribution')
            plt.xlabel('Confidence')
            plt.ylabel('Density')
            fig.legend()
            fig.savefig(os.path.join(exp_dir, "confidences.png"))
            torch.save(valid_summary['logits'], os.path.join(exp_dir, 'logits.pt'))
            torch.save(valid_summary['targets'], os.path.join(exp_dir, 'targets.pt'))
            
            
        else:
            valid_summary = valid_epoch_wo_outlier(model, in_valid_loader,
                                                   loss_func, cur_epoch, logfile2, attack_in)
            summary_log = "=============Epoch [{}]/[{}]=============\nloss: {} | acc: {}\n".format(cur_epoch, max_epoch, valid_summary['avg_loss'], valid_summary['classifier_acc'])
            torch.save(valid_summary['logits'], os.path.join(exp_dir, 'logits.pt'))
            torch.save(valid_summary['targets'], os.path.join(exp_dir, 'targets.pt'))
        print(summary_log)
        logfile.write(summary_log)
            
    logfile.close()
    logfile2.close()
        
    

if __name__=="__main__":
    print("Setup validation...")
    main()