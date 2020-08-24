import os
import numpy as np
import torch
import argparse
from tensorboardX import SummaryWriter
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
import torch.nn.functional as F
from config import cfg
from models.sagan import Generator, Discriminator, Generator32, Discriminator32
from torch.autograd import Variable
from torchvision.utils import save_image
import torch.autograd as autograd

global global_cfg
global_cfg = dict()
cuda = True if torch.cuda.is_available() else False
Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor


def compute_gradient_penalty(feature_extractor, D, real_samples, fake_samples):
    """Calculates the gradient penalty loss for WGAN GP"""
    # Random weight term for interpolation between real and fake samples
    alpha = Tensor(np.random.random((real_samples.size(0), 1, 1, 1)))
    # Get random interpolation between real and fake samples
    interpolates = Variable(alpha * real_samples + ((1 - alpha) * fake_samples), requires_grad=True)
    _, interpolates_features = feature_extractor(interpolates, -1)
    
    d_interpolates,_,_ = D(interpolates, interpolates_features)
    # Get gradient w.r.t. interpolates
    gradients = autograd.grad(
        outputs=d_interpolates.sum(),
        inputs=interpolates,
        grad_outputs=torch.ones(d_interpolates.size()).cuda(),
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
   
    return gradient_penalty

def tensor2var(x, grad=False):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x, requires_grad=grad)


def summary_write(summary, writer):
    for key in summary.keys():
        if key in ['real_target', 'real_feature']:
            continue
        writer.add_scalar(key, summary[key], summary['epoch'])

        
def train_epoch_wo_outlier(feature_extractor, G, D, G_optimizer, D_optimizer, in_loader, cur_epoch, op_cfg, writer):
    global global_cfg
    D.train()
    G.train()
    feature_extractor.eval()
    avg_loss = 0
    correct = 0
    in_data_size = len(in_loader.dataset)
    real_feature = None
    real_target = None
    for cur_iter, in_set in enumerate(in_loader):
        #TODO: Dimension of in_set and out_set should be checked!
        
        # Data to GPU
        real_data = in_set[0]
        targets = in_set[1]
        if cur_iter == 0:
            writer.add_image('in_dist target {}'.format(targets[0]), real_data[0], cur_epoch)
        real_data, targets = real_data.cuda(), targets.cuda()
        
        _, real_features = feature_extractor(real_data, -1)
        if cur_iter == 0:
            real_feature = real_features[0].unsqueeze(0)
            real_target = targets[0]
        # Adjust Learning rate
        lr = optim.get_lr_at_epoch(op_cfg, cur_epoch + float(cur_iter) / in_data_size)
        optim.set_lr(G_optimizer, lr)
        optim.set_lr(D_optimizer, lr)
        
        ###
        d_out_real, dr1, dr2 = D(real_data, real_features)
        if global_cfg['loss']['adv_loss'] == 'wgan-gp':
            d_loss_real = - torch.mean(d_out_real)
        elif global_cfg['loss']['adv_loss'] == 'hinge':
            d_loss_real = torch.nn.ReLU()(1.0 - d_out_real).mean()
            
        z = tensor2var(torch.randn(real_data.size(0), G.z_dim))
        fake_images,gf1,gf2 = G(z, real_features)
        d_out_fake,df1,df2 = D(fake_images, real_features)
        

        if global_cfg['loss']['adv_loss'] == 'wgan-gp':
            d_loss_fake = d_out_fake.mean()
        elif global_cfg['loss']['adv_loss'] == 'hinge':
            d_loss_fake = torch.nn.ReLU()(1.0 + d_out_fake).mean()

            
        # Backward + Optimize
        d_loss = d_loss_real + d_loss_fake
        
        if global_cfg['loss']['adv_loss'] == 'wgan-gp':
            d_loss += op_cfg['lambda_gp'] * compute_gradient_penalty(feature_extractor, D, real_data.data, fake_images.data)
            
        D_optimizer.zero_grad()
        G_optimizer.zero_grad()
        d_loss.backward()
        D_optimizer.step()
            
        # ================== Train G and gumbel ================== #
        # Create random noise
        z = tensor2var(torch.randn(real_data.size(0), G.z_dim))
        _, real_features = feature_extractor(real_data, -1)
        fake_images,_,_ = G(z, real_features)

        # Compute loss with fake images
        g_out_fake,_,_ = D(fake_images, real_features)  # batch x n
        _, fake_features = feature_extractor(fake_images, -1)
        
        if global_cfg['loss']['adv_loss'] == 'wgan-gp':
            g_loss_fake = - g_out_fake.mean()
        elif global_cfg['loss']['adv_loss'] == 'hinge':
            g_loss_fake = - g_out_fake.mean()
            
        g_loss_feature = F.mse_loss(fake_features, real_features)

        g_loss = g_loss_fake + g_loss_feature
        
        D_optimizer.zero_grad()
        G_optimizer.zero_grad()
        g_loss.backward()
        G_optimizer.step()
        ###    
         
        # Add additional metrics!!!
        avg_loss += (d_loss + g_loss)
        
    ## Epoch    
    # Print out log info
    
    
        
    summary = {
        'avg_loss': avg_loss / in_data_size,
        'lr': optim.get_lr_at_epoch(op_cfg, cur_epoch),
        'epoch': cur_epoch,
        'real_feature': real_feature,
        'real_target': real_target,
    }
    
    return summary
  
    
def valid_epoch_wo_outlier(model, in_loader, loss_func, cur_epoch):
    global global_cfg
    model.eval()
    avg_loss = 0
    correct = 0
    in_data_size = len(in_loader.dataset)
    for cur_iter, in_set in enumerate(in_loader):        
        # Data to GPU
        data = in_set[0]
        targets = in_set[1]
        data, targets = data.cuda(), targets.cuda()
        
        # Foward propagation and Calculate loss
        logits, _ = model(data, cur_epoch)

        global_cfg['loss']['model'] = model
        global_cfg['loss']['data'] = data
        loss_dict = loss_func(logits, targets, global_cfg['loss'])
        loss = loss_dict['loss']
        
        # Calculate classifier error about in-distribution sample
        num_topks_correct = metrics.topks_correct(logits[:len(targets)], targets, (1,))
        [top1_correct] = [x for x in num_topks_correct]
        
        # Add additional metrics!!
        
        
        loss, top1_correct = loss.item(), top1_correct.item()
        avg_loss += loss
        correct += top1_correct
    
    summary = {
        'avg_loss': avg_loss / in_data_size,
        'classifier_acc': correct / in_data_size,
        'epoch': cur_epoch,
    }
    
    return summary
    


def train_epoch_w_outlier(model, optimizer, in_loader, out_loader, loss_func, detector_func, cur_epoch, op_cfg, writer):
    global global_cfg
    model.train()
    avg_loss = 0
    correct = 0
    total = 0
    in_data_size = len(in_loader.dataset)
    out_loader.dataset.offset = np.random.randint(len(out_loader.dataset))
    for cur_iter, (in_set, out_set) in enumerate(zip(in_loader, out_loader)):
        #TODO: Dimension of in_set and out_set should be checked!
        
        # Data to GPU
        data = torch.cat((in_set[0], out_set[0]), 0)
        targets = in_set[1]
        if cur_iter == 0:
            writer.add_image('in_dist sample, target:[{}]'.format(targets[0]), in_set[0][0], cur_epoch)
            writer.add_image('out_dist sample', out_set[0][0], cur_epoch)
        data, targets = data.cuda(), targets.cuda()
        
        # Adjust Learning rate
        lr = optim.get_lr_at_epoch(op_cfg, cur_epoch + float(cur_iter) / in_data_size)
        optim.set_lr(optimizer, lr)
        
        # Foward propagation and Calculate loss and confidence
        logits = model(data, cur_epoch)
        global_cfg['loss']['model'] = model
        global_cfg['loss']['data'] = data
        global_cfg['detector']['model'] = model
        global_cfg['detector']['data'] = data
        loss_dict = loss_func(logits, targets, global_cfg['loss'])
        loss = loss_dict['loss']
        confidences_dict = detector_func(logits, targets, global_cfg['detector'])
        confidences = confidences_dict['confidences']

        
        # Back propagation 
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        ## METRICS ##
        # Calculate classifier error about in-distribution sample
        num_topks_correct = metrics.topks_correct(logits[:len(targets)], targets, (1,))
        [top1_correct] = [x for x in num_topks_correct]
        
        # Calculate OOD metrics (auroc, aupr, fpr)
        #(auroc, aupr, fpr) = metrics.get_ood_measures(confidences, targets)
        
        # Add additional metrics!!!
        
        
        ## UDATE STATS ##
        loss, top1_correct = loss.item(), top1_correct.item()
        avg_loss += loss
        correct += top1_correct
        total += targets.size(0)
    
    summary = {
        'avg_loss': avg_loss / total,
        'classifier_acc': correct / total,
        'lr': optim.get_lr_at_epoch(op_cfg, cur_epoch),
        'epoch': cur_epoch,
    }
    
    return summary
  
    
def valid_epoch_w_outlier(model, in_loader, out_loader, loss_func, detector_func, cur_epoch):
    global global_cfg  
    model.eval()
    avg_loss = 0
    correct = 0
    total = 0
    max_iter = 0
    avg_auroc = 0
    avg_aupr = 0
    avg_fpr = 0
    in_data_size = len(in_loader.dataset)
    for cur_iter, (in_set, out_set) in enumerate(zip(in_loader, out_loader)):        
        # Data to GPU
        data = torch.cat((in_set[0], out_set[0]), 0)
        targets = in_set[1]
        data, targets = data.cuda(), targets.cuda()
        
        # Foward propagation and Calculate loss and confidence
        logits = model(data, cur_epoch)
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
        
        ## Update stats ##
        loss, top1_correct = loss.item(), top1_correct.item()
        avg_loss += loss
        correct += top1_correct
        total += targets.size(0)
        max_iter += 1
        avg_auroc += auroc
        avg_aupr += aupr
        avg_fpr += fpr
        
    
    summary = {
        'avg_loss': avg_loss / total,
        'classifier_acc': correct / total,
        'AUROC': avg_auroc / max_iter,
        'AUPR' : avg_aupr / max_iter,
        'FPR95': avg_fpr / max_iter,
        'epoch': cur_epoch,
    }
    
    return summary
    
def main():
    global global_cfg
    # Reproducibility
    np.random.seed(cfg['seed'])
    torch.manual_seed(cfg['seed'])
    start_epoch = 1
    
    # feature_extractor
    feature_extractor = getModel(cfg['model']['feature_extractor'])
    f_checkpoint = torch.load(cfg['f_load_ckpt'], map_location="cpu")
    feature_extractor.load_state_dict(f_checkpoint['model_state'])
    print("load feature_extractor on '{}' is complete.".format(cfg['f_load_ckpt']))

    if cfg['in_dataset']['img_size'] == 32:
        G = Generator32(cfg['in_dataset']['batch_size'], cfg['in_dataset']['img_size'],
                      cfg['model']['z_dim'], cfg['model']['g_conv_dim']).cuda()
        D = Discriminator32(cfg['in_dataset']['batch_size'], cfg['in_dataset']['img_size'],
                          cfg['model']['d_conv_dim']).cuda()
    else:
        G = Generator(cfg['in_dataset']['batch_size'], cfg['in_dataset']['img_size'],
                      cfg['model']['z_dim'], cfg['model']['g_conv_dim']).cuda()
        D = Discriminator(cfg['in_dataset']['batch_size'], cfg['in_dataset']['img_size'],
                          cfg['model']['d_conv_dim']).cuda()
    
    G_optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, G.parameters()), cfg['optim']['g_lr'], [cfg['optim']['beta1'], cfg['optim']['beta2']])
    D_optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, D.parameters()), cfg['optim']['d_lr'], [cfg['optim']['beta1'], cfg['optim']['beta2']])
            
    if cfg['g_load_ckpt'] != '':
        g_checkpoint = torch.load(cfg['g_load_ckpt'], map_location="cpu")
        G.load_state_dict(g_checkpoint['model_state'])
        print("load model on '{}' is complete.".format(cfg['g_load_ckpt']))
    if cfg['d_load_ckpt'] != '':
        d_checkpoint = torch.load(cfg['d_load_ckpt'], map_location="cpu")
        D.load_state_dict(d_checkpoint['model_state'])
        print("load model on '{}' is complete.".format(cfg['d_load_ckpt']))

    cudnn.benchmark = True
    
    # Data Loader
    in_train_loader = getDataLoader(ds_cfg=cfg['in_dataset'],
                                    dl_cfg=cfg['dataloader'],
                                    split="train")
    in_valid_loader = getDataLoader(ds_cfg=cfg['in_dataset'],
                                    dl_cfg=cfg['dataloader'],
                                    split="valid")
    
    if cfg['out_dataset'] is not None:
        out_train_loader = getDataLoader(ds_cfg=cfg['out_dataset'],
                                         dl_cfg=cfg['dataloader'],
                                         split="train")
        out_valid_loader = getDataLoader(ds_cfg=cfg['out_dataset'],
                                         dl_cfg=cfg['dataloader'],
                                         split="valid")
    else:
        out_train_loader = None
        out_valid_loader = None
    
    # Result directory and make tensorboard event file
    exp_dir = os.path.join(cfg['exp_root'], cfg['exp_dir'])
    if os.path.exists(exp_dir) is False:
        os.makedirs(exp_dir)
    shutil.copy('./config.py', os.path.join(exp_dir, "config.py"))
    writer_train = SummaryWriter(logdir=os.path.join(exp_dir, 'log', 'train'))
    writer_valid = SummaryWriter(logdir=os.path.join(exp_dir, 'log', 'valid'))
    
    global_cfg['loss'] = cfg['loss']
    
#     print("=======================IMPORTANT CONFIG=======================")
#     print(" Model    : {}\n \
# Loss     : {}\n \
# Detector : {}\n \
# Optimizer: {}\n".format(cfg['model']['network_kind'], cfg['loss']['loss'], cfg['detector']['detector'], cfg['optim']['optimizer']))
    print("============Start training. Result will be saved in {}".format(exp_dir))
    fixed_z = tensor2var(torch.randn(1, G.z_dim))
    for cur_epoch in range(start_epoch, cfg['max_epoch'] + 1):
        if out_train_loader is not None:
            train_summary = train_epoch_w_outlier(model, optimizer, in_train_loader, out_train_loader, loss_func, detector_func, cur_epoch, cfg['optim'], writer_train)
        else:
            train_summary = train_epoch_wo_outlier(feature_extractor, G, D, G_optimizer, D_optimizer, in_train_loader, cur_epoch, cfg['optim'], writer_train)
        summary_write(summary=train_summary, writer=writer_train)
        print("Training result=========Epoch [{}]/[{}]=========\nlr: {} | loss: {}".format(cur_epoch, cfg['max_epoch'], train_summary['lr'], train_summary['avg_loss']))
                       
        
        # Sample image
        G.eval()
        fake_images,_,_ = G(fixed_z, train_summary['real_feature'])
        save_image(((fake_images.data + 1)/2).clamp_(0, 1),
                   os.path.join(cfg['exp_root'], cfg['exp_dir'], '{}_fake_{}.png'.format(cur_epoch, train_summary['real_target'])))
        
#         if cur_epoch % cfg['valid_epoch'] == 0:
#             if out_valid_loader is not None:
#                 valid_summary = valid_epoch_w_outlier(model, in_valid_loader, out_valid_loader, loss_func, detector_func, cur_epoch)
#             else:
#                 valid_summary = valid_epoch_wo_outlier(model, in_valid_loader, loss_func, cur_epoch)
#             summary_write(summary=valid_summary, writer=writer_valid)
#             print("Validate result=========Epoch [{}]/[{}]=========\nloss: {} | acc: {}".format(cur_epoch, cfg['max_epoch'], valid_summary['avg_loss'], valid_summary['classifier_acc']))
        
        if cur_epoch % cfg['ckpt_epoch'] == 0:
            ckpt_dir = os.path.join(cfg['exp_root'], cfg['exp_dir'], "ckpt")
            if os.path.exists(ckpt_dir) is False:
                os.makedirs(ckpt_dir)
                
            G_state = G.module.state_dict() if cfg['ngpu'] > 1 else G.state_dict()
            G_checkpoint = {
                "epoch": cur_epoch,
                "model_state": G_state,
                "optimizer_state": G_optimizer.state_dict(),
            }
            ckpt_name = "G_checkpoint_epoch_{}".format(cur_epoch)
            ckpt_path = os.path.join(ckpt_dir, ckpt_name + ".pyth")
            torch.save(G_checkpoint, ckpt_path)
            
            D_state = D.module.state_dict() if cfg['ngpu'] > 1 else D.state_dict()
            D_checkpoint = {
                "epoch": cur_epoch,
                "model_state": D_state,
                "optimizer_state": D_optimizer.state_dict(),
            }
            ckpt_name = "D_checkpoint_epoch_{}".format(cur_epoch)
            ckpt_path = os.path.join(ckpt_dir, ckpt_name + ".pyth")
            torch.save(D_checkpoint, ckpt_path)
    
        
if __name__=="__main__":
    print("Setup Training...")
    main()