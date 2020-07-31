import os
import numpy as np
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
from config import cfg
from torch.autograd import Variable

global global_cfg
global_cfg = dict()


def summary_write(summary, writer):
    for key in summary.keys():
        writer.add_scalar(key, summary[key], summary['epoch'])
        
    
def valid_epoch_wo_outlier(model, in_loader, detector_func):
    global global_cfg
    model.eval()

    h_confidences_list = []
    f_confidences_list = []
    targets_list = []
    h_logits_list = []
    total = 0
    correct = 0
    
    in_data_size = len(in_loader.dataset)
    for cur_iter, in_set in enumerate(in_loader):        
        # Data to GPU
        data = in_set[0]
        targets = in_set[1]
        data, targets = data.cuda(), targets.cuda()
        
        ### Input preprocessing
        inputs = Variable(data, requires_grad=True)
        _, h_logits = model(inputs)
        loss = torch.max(h_logits, dim=1).values.mean()
        loss.backward()        

        gradient = torch.ge(inputs.grad.data, 0)
        gradient = (gradient.float() - 0.5) * 2
        gradient.index_copy_(1, torch.LongTensor([0]).cuda(),
                             gradient.index_select(1, torch.LongTensor([0]).cuda()) / (0.2023))
        gradient.index_copy_(1, torch.LongTensor([1]).cuda(),
                                 gradient.index_select(1, torch.LongTensor([1]).cuda()) / (0.1994))
        gradient.index_copy_(1, torch.LongTensor([2]).cuda(),
                                 gradient.index_select(1, torch.LongTensor([2]).cuda()) / (0.2010))
        
        tempInputs = torch.add(data.data, gradient, alpha=-global_cfg['detector']['magnitude'])
        ###
        
        # Foward propagation and Calculate loss
        with torch.no_grad():
            f_logits, h_logits = model(Variable(tempInputs))
        
        global_cfg['detector']['model'] = model
        global_cfg['detector']['data'] = data
        h_confidences_dict = detector_func(h_logits, targets, global_cfg['detector'])
        f_confidences_dict = detector_func(f_logits, targets, global_cfg['detector'])
        
        h_confidences_list.append(h_confidences_dict['confidences'].cpu())
        f_confidences_list.append(f_confidences_dict['confidences'].cpu())
        targets_list.append(targets.cpu())
        h_logits_list.append(h_logits.cpu())
        
        num_topks_correct = metrics.topks_correct(h_logits[:len(targets)], targets, (1,))
        [top1_correct] = [x for x in num_topks_correct]
        top1_correct = top1_correct.item()
        correct += top1_correct
        total += targets.size(0)
        
    
    summary = {
        'classifier_acc': correct / total,
        'h_confidences': torch.cat(h_confidences_list, dim=0), # (Bs, 1)
        'f_confidences': torch.cat(f_confidences_list, dim=0),
        'targets': torch.cat(targets_list, dim=0), # (Bs,)
        'logits' : torch.cat(h_logits_list, dim=0) # (Bs, num_classes)
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
    
    
    
    if 'targets' in cfg['in_dataset'].keys():
        exp_dir = os.path.join(cfg['exp_root'], cfg['exp_dir'], "logits", cfg['in_dataset']['dataset'], cfg['in_dataset']['targets'][0])
    else: 
        exp_dir = os.path.join(cfg['exp_root'], cfg['exp_dir'], "logits", cfg['in_dataset']['dataset'])
        
    
    # Result directory and make tensorboard event file
    if os.path.exists(exp_dir) is False:
        os.makedirs(exp_dir)    
    
    # Outlier detector
    detector_func = detectors.getDetector(cfg['detector'])
    global_cfg['detector'] = cfg['detector']
    
    
    # Outlier detector
    print("=======================IMPORTANT CONFIG=======================")
    print(" Model    : {}\n \
Detector     : {}\n".format(cfg['model']['network_kind'], cfg['detector']['detector']))
    print("========Start logits extraction for GODIN. Result will be saved in {}".format(exp_dir))
    
    
    valid_summary = valid_epoch_wo_outlier(model, in_valid_loader, detector_func)
    summary_log = "Acc [{}]\n".format(valid_summary['classifier_acc'])
    print(summary_log)
    
    torch.save(valid_summary['f_confidences'], os.path.join(exp_dir, 'f_confidences.pt'))
    torch.save(valid_summary['h_confidences'], os.path.join(exp_dir, 'h_confidences.pt'))
    torch.save(valid_summary['targets'], os.path.join(exp_dir, 'targets.pt'))
    torch.save(valid_summary['logits'], os.path.join(exp_dir, 'logits.pt'))
    

if __name__=="__main__":
    print("Setup logits extraction...")
    main()