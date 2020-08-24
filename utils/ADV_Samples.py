"""
Created on Sun Oct 25 2018
@author: Kimin Lee
"""
from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import numpy as np
import os
import utils.adversary as adversary
from config import cfg
import sys
sys.path.append('./')

from torchvision import transforms
from torch.autograd import Variable

from models.model_builder import getModel
from datasets.data_loader import getDataLoader


# parser = argparse.ArgumentParser(description='PyTorch code: Mahalanobis detector')
# parser.add_argument('--batch_size', type=int, default=200, metavar='N', help='batch size for data loader')
# parser.add_argument('--dataset', required=True, help='cifar10 | cifar100 | svhn')
# parser.add_argument('--dataroot', default='./data', help='path to dataset')
# parser.add_argument('--outf', default='./adv_output/', help='folder to output results')
# parser.add_argument('--num_classes', type=int, default=10, help='the # of classes')
# parser.add_argument('--net_type', required=True, help='resnet | densenet')
# parser.add_argument('--gpu', type=int, default=0, help='gpu index')
# parser.add_argument('--adv_type', required=True, help='FGSM | BIM | DeepFool | CWL2')
# args = parser.parse_args()
# print(args)

def main(cfg):
    # set the path to pre-trained model and output
    pre_trained_net = cfg['load_ckpt']
    outf=os.path.join(cfg['exp_root'],cfg['exp_dir'],'valid','MD_ADV')
    if os.path.isdir(outf) == False:
        os.makedirs(outf)
        
    torch.cuda.manual_seed(cfg['seed'])
    device = torch.device('cuda:0')    
    
    # check the in-distribution dataset
    num_classes=cfg['model']['num_classes']
    if cfg['detector']['adv_method'] == 'FGSM':
        adv_noise = 0.05
    elif cfg['detector']['adv_method'] == 'BIM':
        adv_noise = 0.01
    elif cfg['detector']['adv_method'] == 'DeepFool':
        if cfg['model']['net_type'] == 'resnet':
            if cfg['in_dataset']['dataset'] == 'cifar10':
                adv_noise = 0.18
            elif cfg['in_dataset']['dataset'] == 'cifar100':
                adv_noise = 0.03
            else:
                adv_noise = 0.1
        else:
            if cfg['in_dataset']['dataset'] == 'cifar10':
                adv_noise = 0.6
            elif cfg['in_dataset']['dataset'] == 'cifar100':
                adv_noise = 0.1
            else:
                adv_noise = 0.5

    # load networks
#     if args.net_type == 'densenet':
#         if args.dataset == 'svhn':
    model = getModel(cfg['model'])
    net_type='resnet'    ## Should change in the future
    checkpoint = torch.load(cfg['load_ckpt'], map_location="cpu")
    model.load_state_dict(checkpoint['model_state'])

    in_transform = cfg['in_dataset']['valid_transform']
        
    min_pixel = -2.42906570435
    max_pixel = 2.75373125076
    if cfg['in_dataset']['dataset'] == 'cifar10':
        if cfg['detector']['adv_method'] == 'FGSM':
            random_noise_size = 0.25 / 4
        elif cfg['detector']['adv_method'] == 'BIM':
            random_noise_size = 0.13 / 2
        elif cfg['detector']['adv_method'] == 'DeepFool':
            random_noise_size = 0.25 / 4
        elif cfg['detector']['adv_method'] == 'CWL2':
                random_noise_size = 0.05 / 2
    elif cfg['in_dataset']['dataset'] == 'cifar100':
        if cfg['detector']['adv_method'] == 'FGSM':
            random_noise_size = 0.25 / 8
        elif cfg['detector']['adv_method'] == 'BIM':
                random_noise_size = 0.13 / 4
        elif cfg['detector']['adv_method'] == 'DeepFool':
                random_noise_size = 0.13 / 4
        elif cfg['detector']['adv_method'] == 'CWL2':
                random_noise_size = 0.05 / 2
    else:
        if cfg['detector']['adv_method'] == 'FGSM':
            random_noise_size = 1
        elif cfg['detector']['adv_method'] == 'BIM':
            random_noise_size = 1
        elif cfg['detector']['adv_method'] == 'DeepFool':
            random_noise_size = 1
        elif cfg['detector']['adv_method'] == 'CWL2':
            random_noise_size = 1
            
    model.cuda()
    print("load model on '{}' is completed.".format(cfg['load_ckpt']))
    
    # load dataset
    print('load target data: ', cfg['in_dataset']['dataset'])
    test_loader =  getDataLoader(ds_cfg=cfg['in_dataset'],
                                    dl_cfg=cfg['dataloader'],
                                    split='valid')

    
    print('Attack: ' + cfg['detector']['adv_method']  +  ', Dist: ' + cfg['detector']['adv_method'] + '\n')
    model.eval()
    adv_data_tot, clean_data_tot, noisy_data_tot = 0, 0, 0
    label_tot = 0
    
    correct, adv_correct, noise_correct = 0, 0, 0
    total, generated_noise = 0, 0

    criterion = nn.CrossEntropyLoss().cuda()

    selected_list = []
    selected_index = 0
    
    for data, target in test_loader:
        data, target = data.cuda(), target.cuda()
        with torch.no_grad(): data, target = Variable(data), Variable(target)
        output = model(data)

        # compute the accuracy
        pred = output.data.max(1)[1]
        equal_flag = pred.eq(target.data).cpu()
        correct += equal_flag.sum()

        noisy_data = torch.add(data.data, torch.randn(data.size()).cuda(), alpha = random_noise_size) 
        noisy_data = torch.clamp(noisy_data, min_pixel, max_pixel)

        if total == 0:
            clean_data_tot = data.clone().data.cpu()
            label_tot = target.clone().data.cpu()
            noisy_data_tot = noisy_data.clone().cpu()
        else:
            clean_data_tot = torch.cat((clean_data_tot, data.clone().data.cpu()),0)
            label_tot = torch.cat((label_tot, target.clone().data.cpu()), 0)
            noisy_data_tot = torch.cat((noisy_data_tot, noisy_data.clone().cpu()),0)
            
        # generate adversarial
        model.zero_grad()
        inputs = Variable(data.data, requires_grad=True)
        output = model(inputs)
        loss = criterion(output, target)
        loss.backward()

        if cfg['detector']['adv_method'] == 'FGSM': 
            gradient = torch.ge(inputs.grad.data, 0)
            gradient = (gradient.float()-0.5)*2
            if cfg['model']['net_type'] == 'densenet':
                gradient.index_copy_(1, torch.LongTensor([0]).cuda(), \
                                     gradient.index_select(1, torch.LongTensor([0]).cuda()) / (63.0/255.0))
                gradient.index_copy_(1, torch.LongTensor([1]).cuda(), \
                                     gradient.index_select(1, torch.LongTensor([1]).cuda()) / (62.1/255.0))
                gradient.index_copy_(1, torch.LongTensor([2]).cuda(), \
                                     gradient.index_select(1, torch.LongTensor([2]).cuda()) / (66.7/255.0))
            else:
                gradient.index_copy_(1, torch.LongTensor([0]).cuda(), \
                                     gradient.index_select(1, torch.LongTensor([0]).cuda()) / (0.2023))
                gradient.index_copy_(1, torch.LongTensor([1]).cuda(), \
                                     gradient.index_select(1, torch.LongTensor([1]).cuda()) / (0.1994))
                gradient.index_copy_(1, torch.LongTensor([2]).cuda(), \
                                     gradient.index_select(1, torch.LongTensor([2]).cuda()) / (0.2010))

        elif cfg['detector']['adv_method'] == 'BIM': 
            gradient = torch.sign(inputs.grad.data)
            for k in range(5):
                inputs = torch.add(inputs.data, gradient, alpha = adv_noise)
                inputs = torch.clamp(inputs, min_pixel, max_pixel)
                inputs = Variable(inputs, requires_grad=True)
                output = model(inputs)
                loss = criterion(output, target)
                loss.backward()
                gradient = torch.sign(inputs.grad.data)
                if cfg['model']['net_type'] == 'densenet':
                    gradient.index_copy_(1, torch.LongTensor([0]).cuda(), \
                                         gradient.index_select(1, torch.LongTensor([0]).cuda()) / (63.0/255.0))
                    gradient.index_copy_(1, torch.LongTensor([1]).cuda(), \
                                         gradient.index_select(1, torch.LongTensor([1]).cuda()) / (62.1/255.0))
                    gradient.index_copy_(1, torch.LongTensor([2]).cuda(), \
                                         gradient.index_select(1, torch.LongTensor([2]).cuda()) / (66.7/255.0))
                else:
                    gradient.index_copy_(1, torch.LongTensor([0]).cuda(), \
                                         gradient.index_select(1, torch.LongTensor([0]).cuda()) / (0.2023))
                    gradient.index_copy_(1, torch.LongTensor([1]).cuda(), \
                                         gradient.index_select(1, torch.LongTensor([1]).cuda()) / (0.1994))
                    gradient.index_copy_(1, torch.LongTensor([2]).cuda(), \
                                         gradient.index_select(1, torch.LongTensor([2]).cuda()) / (0.2010))

        if cfg['detector']['adv_method'] == 'DeepFool':
            _, adv_data = adversary.deepfool(model, data.data.clone(), target.data.cpu(), \
                                             args.num_classes, step_size=adv_noise, train_mode=False)
            adv_data = adv_data.cuda()
        elif cfg['detector']['adv_method'] == 'CWL2':
            _, adv_data = adversary.cw(model, data.data.clone(), target.data.cpu(), 1.0, 'l2', crop_frac=1.0)
        else:
            adv_data = torch.add(inputs.data, gradient, alpha = adv_noise)
            
        adv_data = torch.clamp(adv_data, min_pixel, max_pixel)
        
        # measure the noise 
        temp_noise_max = torch.abs((data.data - adv_data).view(adv_data.size(0), -1))
        temp_noise_max, _ = torch.max(temp_noise_max, dim=1)
        generated_noise += torch.sum(temp_noise_max)


        if total == 0:
            flag = 1
            adv_data_tot = adv_data.clone().cpu()
        else:
            adv_data_tot = torch.cat((adv_data_tot, adv_data.clone().cpu()),0)

        with torch.no_grad(): output = model(Variable(adv_data))
        # compute the accuracy
        pred = output.data.max(1)[1]
        equal_flag_adv = pred.eq(target.data).cpu()
        adv_correct += equal_flag_adv.sum()
        
        with torch.no_grad(): output = model(Variable(noisy_data))
        # compute the accuracy
        pred = output.data.max(1)[1]
        equal_flag_noise = pred.eq(target.data).cpu()
        noise_correct += equal_flag_noise.sum()
        
        for i in range(data.size(0)):
            if equal_flag[i] == 1 and equal_flag_noise[i] == 1 and equal_flag_adv[i] == 0:
                selected_list.append(selected_index)
            selected_index += 1
            
        total += data.size(0)

    selected_list = torch.LongTensor(selected_list)
    clean_data_tot = torch.index_select(clean_data_tot, 0, selected_list)
    adv_data_tot = torch.index_select(adv_data_tot, 0, selected_list)
    noisy_data_tot = torch.index_select(noisy_data_tot, 0, selected_list)
    label_tot = torch.index_select(label_tot, 0, selected_list)

    torch.save(clean_data_tot, '%s/clean_data_%s_%s_%s.pth' % (args.outf, args.net_type, args.dataset, args.adv_type))
    torch.save(adv_data_tot, '%s/adv_data_%s_%s_%s.pth' % (args.outf, args.net_type, args.dataset, args.adv_type))
    torch.save(noisy_data_tot, '%s/noisy_data_%s_%s_%s.pth' % (args.outf, args.net_type, args.dataset, args.adv_type))
    torch.save(label_tot, '%s/label_%s_%s_%s.pth' % (args.outf, args.net_type, args.dataset, args.adv_type))

    print('Adversarial Noise:({:.2f})\n'.format(generated_noise / total))
    print('Final Accuracy: {}/{} ({:.2f}%)\n'.format(correct, total, 100. * correct / total))
    print('Adversarial Accuracy: {}/{} ({:.2f}%)\n'.format(adv_correct, total, 100. * adv_correct / total))
    print('Noisy Accuracy: {}/{} ({:.2f}%)\n'.format(noise_correct, total, 100. * noise_correct / total))
    
if __name__ == '__main__':
    main()
