from __future__ import print_function

import os
import numpy as np
import torch
import argparse
import shutil
import time
import torch.backends.cudnn as cudnn
from tensorboardX import SummaryWriter
from tqdm import tqdm
import sys
sys.path.append('./')
import utils.metrics as metrics
import utils.optimizer as optim
from models.model_builder import getModel
from datasets.data_loader import getDataLoader
import matplotlib.pyplot as plt
import seaborn as sns
import torch.nn as nn

import utils.lib_generation as lib_generation
import utils.lib_regression as lib_regression
import utils.adversary as adversary
import utils.calculate_log as callog

from torchvision import transforms
from torch.autograd import Variable

from sklearn.linear_model import LogisticRegressionCV

from config import cfg

def summary_write(summary, writer):
    for key in summary.keys():
        writer.add_scalar(key, summary[key])

        
# set the path to pre-trained model and output
pre_trained_net = cfg['load_ckpt']
outf=os.path.join(cfg['exp_root'],cfg['exp_dir'],'valid','MD_ADV')
if os.path.isdir(outf) == False:
    os.makedirs(outf)

torch.cuda.manual_seed(cfg['seed'])

shutil.copy('./config.py', os.path.join(outf, "config_"+cfg['exp_dir']+".py"))

writer_valid = SummaryWriter(logdir=os.path.join(outf, 'log', 'valid'))
writer_test = SummaryWriter(logdir=os.path.join(outf, 'log', 'test'))
writer_train = SummaryWriter(logdir=os.path.join(outf, 'log', 'train'))


# check the in-distribution dataset
num_classes=cfg['model']['num_classes']

best_auroc, best_result, best_index = 0, 0, 0
temp_file_name_1='%s/ADV_and_RANDOM_NOISE_SIZE_TRAIN.txt'%(outf)
f = open(temp_file_name_1, 'w')
temp_file_name_2='%s/ADV_and_RANDOM_NOISE_SIZE_VALID.txt'%(outf)
g = open(temp_file_name_2, 'w')
temp_file_name_3='%s/ADV_and_RANDOM_NOISE_SIZE_TEST.txt'%(outf)
h = open(temp_file_name_3, 'w')

model = getModel(cfg['model'])
checkpoint = torch.load(cfg['load_ckpt'], map_location="cpu")
model.load_state_dict(checkpoint['model_state'])

min_pixel = cfg['min']
max_pixel = cfg['max']

model.cuda()
print("load model on '{}' is completed.".format(cfg['load_ckpt']))

# load dataset
test_loader =  getDataLoader(ds_cfg=cfg['in_dataset'],
                                dl_cfg=cfg['dataloader'],
                                split='valid')

fooled_log = open(os.path.join(outf, 'fooled.txt'), 'w')

# Tuning on FGSM examples
# Repeat for Various Hyperparameters :
for adv_noise in cfg['detector']['adv_noise']:
    for random_noise_size in cfg['detector']['epsilon']:
        
        print("=================CURRENT STATE = ADV / UNI = {} / {}".format(adv_noise, random_noise_size))
        print('Attack: ' + cfg['detector']['adv_method']  +  ', Dist: ' + cfg['in_dataset']['dataset'] + '\n')
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
#                 print(clean_data_tot)
                label_tot = target.clone().data.cpu()
                noisy_data_tot = noisy_data.clone().cpu()
            else:
                clean_data_tot = torch.cat((clean_data_tot, data.clone().data.cpu()),0)
                label_tot = torch.cat((label_tot, target.clone().data.cpu()), 0)
                noisy_data_tot = torch.cat((noisy_data_tot, noisy_data.clone().cpu()),0)
            model.zero_grad()
            inputs = Variable(data.data, requires_grad=True)
            output = model(inputs)
            loss = criterion(output, target)
            loss.backward()

            if cfg['detector']['adv_method'] == 'FGSM': 
                gradient = torch.ge(inputs.grad.data, 0)
                gradient = (gradient.float()-0.5)*2
                gradient.index_copy_(1, torch.LongTensor([0]).cuda(), \
                                     gradient.index_select(1, torch.LongTensor([0]).cuda())/cfg['std'][0])
                gradient.index_copy_(1, torch.LongTensor([1]).cuda(), \
                                     gradient.index_select(1, torch.LongTensor([1]).cuda())/cfg['std'][1])
                gradient.index_copy_(1, torch.LongTensor([2]).cuda(), \
                                     gradient.index_select(1, torch.LongTensor([2]).cuda())/cfg['std'][2])

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

        torch.save(clean_data_tot, '%s/clean_data_%s_%s_%s.pth' % (outf, cfg['model']['net_type'], cfg['in_dataset']['dataset'].replace('/','_'), cfg['detector']['adv_method']))
        torch.save(adv_data_tot, '%s/adv_data_%s_%s_%s.pth' % (outf, cfg['model']['net_type'], cfg['in_dataset']['dataset'].replace('/','_'), cfg['detector']['adv_method']))
        torch.save(noisy_data_tot, '%s/noisy_data_%s_%s_%s.pth' % (outf, cfg['model']['net_type'], cfg['in_dataset']['dataset'].replace('/','_'), cfg['detector']['adv_method']))
        torch.save(label_tot, '%s/label_%s_%s_%s.pth' % (outf, cfg['model']['net_type'], cfg['in_dataset']['dataset'].replace('/','_'), cfg['detector']['adv_method']))

        print('Adversarial Noise:({:.2f})\n'.format(generated_noise / total))
        print('Final Accuracy: {}/{} ({:.2f}%)\n'.format(correct, total, 100. * correct / total))
        print('Adversarial Accuracy: {}/{} ({:.2f}%)\n'.format(adv_correct, total, 100. * adv_correct / total))
        print('Noisy Accuracy: {}/{} ({:.2f}%)\n'.format(noise_correct, total, 100. * noise_correct / total))


        # set the path to the Mahal oe_tune model and output
        pre_trained_net = cfg['load_ckpt']
        outf=os.path.join(cfg['exp_root'],cfg['exp_dir'],'valid','MD_ADV')

        if os.path.isdir(outf) == False:
            os.makedirs(outf)

        ##### load model

        model = getModel(cfg['model'])
        checkpoint = torch.load(cfg['load_ckpt'], map_location="cpu")
        model.load_state_dict(checkpoint['model_state'])

        model = model.cuda()
        print("\nload model on '{}' is completed.".format(cfg['load_ckpt']))

        ##### load dataset
        print('\nloaing in-distribution target data: ', cfg['in_dataset']['dataset'])

        print('loaing adversarial noise data : ')
        valid_clean_data = torch.load(outf + '/clean_data_%s_%s_%s.pth' % (cfg['model']['net_type'], cfg['in_dataset']['dataset'].replace('/','_'), cfg['detector']['adv_method']))
        valid_adv_data = torch.load(outf + '/adv_data_%s_%s_%s.pth' % (cfg['model']['net_type'], cfg['in_dataset']['dataset'].replace('/','_'), cfg['detector']['adv_method']))
        valid_noisy_data = torch.load(outf + '/noisy_data_%s_%s_%s.pth' % (cfg['model']['net_type'], cfg['in_dataset']['dataset'].replace('/','_'), cfg['detector']['adv_method']))
        valid_label = torch.load(outf + '/label_%s_%s_%s.pth' % (cfg['model']['net_type'], cfg['in_dataset']['dataset'].replace('/','_'), cfg['detector']['adv_method']))

        print('\nload out-of-distribution data: ', cfg['out_dataset']['dataset'])

        out_test_loader = getDataLoader(ds_cfg=cfg['out_dataset'],dl_cfg=cfg['dataloader'],split='train')
        public_out_loader = getDataLoader(ds_cfg=cfg['public_out_dataset'],dl_cfg=cfg['dataloader'],split='train')

        ###### setting information about feature extaction
        model.eval()
        temp_x = torch.rand(2,3,cfg['model']['image_size'],cfg['model']['image_size']).cuda()
        temp_x = Variable(temp_x)
        temp_list = model.feature_list(temp_x)[1]
        num_output = len(temp_list)
        feature_list = np.empty(num_output)
#        count = 0
#         for out in temp_list:
#             feature_list[count] = out.size(1)
#             count += 1
        for list_idx, out in enumerate(temp_list):
            feature_list[list_idx] = out.size(1)

        print('get sample mean and covariance')
        sample_mean, precision, covariance = lib_generation.sample_estimator(model, cfg['model']['num_classes'], feature_list, test_loader)

        print('get Mahalanobis scores for ADV Validation Set')
        m_list=[0.0]       ### <- Input Pre-Processing Magnitude
#         m_list = [0.0, 0.01, 0.005, 0.002, 0.0014, 0.001, 0.0005]
        for magnitude in m_list:
            print('\n =========== Noise: ' + str(magnitude))
            progress_bar=tqdm(range(num_output))
            for i in progress_bar:
                progress_bar.set_description('VALID-CLEAN')
                M_in, _\
                = lib_generation.get_Mahalanobis_score_adv(model, valid_clean_data, valid_label, \
                                                           cfg['model']['num_classes'], outf, cfg['model']['net_type'], \
                                                           sample_mean, precision, covariance, i, magnitude)
                M_in = np.asarray(M_in, dtype=np.float32)
                if i == 0:
                    Mahalanobis_in = M_in.reshape((M_in.shape[0], -1))
                else:
                    Mahalanobis_in = np.concatenate((Mahalanobis_in, M_in.reshape((M_in.shape[0], -1))), axis=1)

            progress_bar=tqdm(range(num_output))
            for i in progress_bar:
                progress_bar.set_description('VALID-ADV')
                M_out, _ \
                = lib_generation.get_Mahalanobis_score_adv(model, valid_adv_data, valid_label, \
                                                           cfg['model']['num_classes'], outf, cfg['model']['net_type'], \
                                                           sample_mean, precision, covariance, i, magnitude)
                M_out = np.asarray(M_out, dtype=np.float32)
                if i == 0:
                    Mahalanobis_out = M_out.reshape((M_out.shape[0], -1))
                else:
                    Mahalanobis_out = np.concatenate((Mahalanobis_out, M_out.reshape((M_out.shape[0], -1))), axis=1)

            progress_bar=tqdm(range(num_output))
            for i in progress_bar:
                progress_bar.set_description('VALID-NOISY')
                M_noisy, _ \
                = lib_generation.get_Mahalanobis_score_adv(model, valid_noisy_data, valid_label, \
                                                           cfg['model']['num_classes'], outf, cfg['model']['net_type'], \
                                                           sample_mean, precision, covariance, i, magnitude)
                M_noisy = np.asarray(M_noisy, dtype=np.float32)
                if i == 0:
                    Mahalanobis_noisy = M_noisy.reshape((M_noisy.shape[0], -1))
                else:
                    Mahalanobis_noisy = np.concatenate((Mahalanobis_noisy, M_noisy.reshape((M_noisy.shape[0], -1))), axis=1)
            Mahalanobis_in = np.asarray(Mahalanobis_in, dtype=np.float32)
            Mahalanobis_out = np.asarray(Mahalanobis_out, dtype=np.float32)
            Mahalanobis_noisy = np.asarray(Mahalanobis_noisy, dtype=np.float32)
            Mahalanobis_pos = np.concatenate((Mahalanobis_in, Mahalanobis_noisy))

            Mahalanobis_data, Mahalanobis_labels = lib_generation.merge_and_generate_labels(Mahalanobis_out, Mahalanobis_pos)
            file_name = os.path.join(outf, 'Mahalanobis_%s_%s_%s_%s.npy' % (str(magnitude), cfg['in_dataset']['dataset'].replace('/','_'), cfg['detector']['adv_method'],'adv_valid'))

            Mahalanobis_data = np.concatenate((Mahalanobis_data, Mahalanobis_labels), axis=1)
            np.save(file_name, Mahalanobis_data)
            
            
            
            
        print('\n\nget Mahalanobis scores for Real OOD Test Set')
        for magnitude in m_list:
            print('\n===========Noise: ' + str(magnitude))
            progress_bar=tqdm(range(num_output))
            for i in progress_bar:
                progress_bar.set_description('REAL-IN-DATA ')
                M_in, fooled, loglikelihood = lib_generation.get_Mahalanobis_score(model, test_loader, cfg['model']['num_classes'], outf, \
                                                            True, cfg['model']['net_type'], sample_mean, precision, covariance, i, magnitude,3,True)
                M_in = np.asarray(M_in, dtype=np.float32)
                if i == 0:
                    Mahalanobis_in = M_in.reshape((M_in.shape[0], -1))
                else:
                    Mahalanobis_in = np.concatenate((Mahalanobis_in, M_in.reshape((M_in.shape[0], -1))), axis=1)
                    
                    
                dist_check = "\n=====================Distribution Check=====================\n"
                dist_check += "Layer {} | Magnitude {} | noise {} | Adv_attack {} => fooled {}\n".format(i+1, magnitude, random_noise_size, adv_noise, fooled)
                dist_check += "=====================Class-wise log-likelihood=====================\n"
                for idx, llh in enumerate(loglikelihood):
                    dist_check += "class [{}]: {}\n".format(idx, llh)
                print(dist_check)
                fooled_log.write(dist_check)        
                                

            progress_bar=tqdm(range(num_output))
            for i in progress_bar:
                progress_bar.set_description('REAL-OUT-DATA')
                M_out, _, _ = lib_generation.get_Mahalanobis_score(model, out_test_loader, cfg['model']['num_classes'], outf, \
                                                                 False, cfg['model']['net_type'], sample_mean, precision, covariance, i, magnitude,3)
                M_out = np.asarray(M_out, dtype=np.float32)
                if i == 0:
                    Mahalanobis_out = M_out.reshape((M_out.shape[0], -1))
                else:
                    Mahalanobis_out = np.concatenate((Mahalanobis_out, M_out.reshape((M_out.shape[0], -1))), axis=1)

            Mahalanobis_in = np.asarray(Mahalanobis_in, dtype=np.float32)
            Mahalanobis_out = np.asarray(Mahalanobis_out, dtype=np.float32)

            Mahalanobis_data, Mahalanobis_labels = lib_generation.merge_and_generate_labels(Mahalanobis_out, Mahalanobis_in)
            file_name = os.path.join(outf, 'Mahalanobis_%s_%s_%s_%s.npy' % (str(magnitude), cfg['in_dataset']['dataset'].replace("/","_") , cfg['out_dataset']['dataset'].replace("/","_"),'Real_OOD'))

            Mahalanobis_data = np.concatenate((Mahalanobis_data, Mahalanobis_labels), axis=1)
            np.save(file_name, Mahalanobis_data)
            
            
        print('\n\nget Mahalanobis scores for Fake OOD Set')
        for magnitude in m_list:
            print('\n===========Noise: ' + str(magnitude))
            progress_bar=tqdm(range(num_output))
            for i in progress_bar:
                progress_bar.set_description('REAL-IN-DATA ')
                M_in, fooled, loglikelihood = lib_generation.get_Mahalanobis_score(model, test_loader, cfg['model']['num_classes'], outf, \
                                                            True, cfg['model']['net_type'], sample_mean, precision, covariance, i, magnitude,3)
                M_in = np.asarray(M_in, dtype=np.float32)
                if i == 0:
                    Mahalanobis_in = M_in.reshape((M_in.shape[0], -1))
                else:
                    Mahalanobis_in = np.concatenate((Mahalanobis_in, M_in.reshape((M_in.shape[0], -1))), axis=1)

            progress_bar=tqdm(range(num_output))
            for i in progress_bar:
                progress_bar.set_description('FAKE-OUT-DATA')
                M_out, _, _ = lib_generation.get_Mahalanobis_score(model, public_out_loader, cfg['model']['num_classes'], outf, \
                                                                 False, cfg['model']['net_type'], sample_mean, precision, covariance, i, magnitude,3)
                M_out = np.asarray(M_out, dtype=np.float32)
                if i == 0:
                    Mahalanobis_out = M_out.reshape((M_out.shape[0], -1))
                else:
                    Mahalanobis_out = np.concatenate((Mahalanobis_out, M_out.reshape((M_out.shape[0], -1))), axis=1)

            Mahalanobis_in = np.asarray(Mahalanobis_in, dtype=np.float32)
            Mahalanobis_out = np.asarray(Mahalanobis_out, dtype=np.float32)

            Mahalanobis_data, Mahalanobis_labels = lib_generation.merge_and_generate_labels(Mahalanobis_out, Mahalanobis_in)
            file_name = os.path.join(outf, 'Mahalanobis_%s_%s_%s_%s.npy' % (str(magnitude), cfg['in_dataset']['dataset'].replace("/","_") , cfg['public_out_dataset']['dataset'].replace("/","_"),'Fake_OOD'))

            Mahalanobis_data = np.concatenate((Mahalanobis_data, Mahalanobis_labels), axis=1)
            np.save(file_name, Mahalanobis_data)


        ##### Regression 
        
        print('\n\n evaluate the Mahalanobis estimator')
        score_list=['Mahalanobis_0.0']   
#         score_list = ['Mahalanobis_0.0', 'Mahalanobis_0.01', 'Mahalanobis_0.005', \
#                       'Mahalanobis_0.002', 'Mahalanobis_0.0014', 'Mahalanobis_0.001', 'Mahalanobis_0.0005']
        list_best_results_ours, list_best_results_index_ours = [], []
        list_best_results_out, list_best_results_index_out = [], []
        out = cfg['out_dataset']['dataset']
        for score in score_list:
            print('load train data: ', out, ' of ', score)

            train_X, train_Y = lib_regression.load_characteristics(score, cfg['in_dataset']['dataset'], cfg['detector']['adv_method'], outf,'adv_valid')
            valid_X, valid_Y =  lib_regression.load_characteristics(score, cfg['in_dataset']['dataset'], cfg['public_out_dataset']['dataset'], outf,'Fake_OOD')
            test_X, test_Y =  lib_regression.load_characteristics(score, cfg['in_dataset']['dataset'], cfg['out_dataset']['dataset'], outf,'Real_OOD')

            
            lr = LogisticRegressionCV(n_jobs=-1,max_iter=10000).fit(train_X, train_Y)

            train_results = lib_regression.detection_performance(lr, train_X, train_Y, outf, 'train', adv_noise, random_noise_size)
            if cfg['detector']['graph']==True:
                lib_regression.get_histogram_front(lr, train_X, train_Y, outf, 'train',adv_noise, random_noise_size)
                
            test_results=lib_regression.detection_performance(lr, test_X, test_Y, outf, 'test',adv_noise, random_noise_size)
            if cfg['detector']['graph']==True:
                lib_regression.get_histogram_front(lr, test_X, test_Y, outf, 'test', adv_noise, random_noise_size)
            
            valid_results=lib_regression.detection_performance(lr, valid_X, valid_Y, outf, 'valid',adv_noise, random_noise_size)
            if cfg['detector']['graph']==True:
                lib_regression.get_histogram_front(lr, valid_X, valid_Y, outf, 'valid', adv_noise, random_noise_size)


            test_summary={
                'TNR': test_results['TMP']['TNR'],
                'AUROC': test_results['TMP']['AUROC'],
                'DTACC': test_results['TMP']['DTACC'],
                'AUPR': test_results['TMP']['AUPR'],
                'adv_noise':adv_noise,
                'random_noise_size':random_noise_size
            }
            train_summary={
                'TNR': train_results['TMP']['TNR'],
                'AUROC': train_results['TMP']['AUROC'],
                'DTACC': train_results['TMP']['DTACC'],
                'AUPR': train_results['TMP']['AUPR'],
                'adv_noise':adv_noise,
                'random_noise_size':random_noise_size
            }
            valid_summary={
                'TNR': valid_results['TMP']['TNR'],
                'AUROC': valid_results['TMP']['AUROC'],
                'DTACC': valid_results['TMP']['DTACC'],
                'AUPR': valid_results['TMP']['AUPR'],
                'adv_noise':adv_noise,
                'random_noise_size':random_noise_size
            }            
            summary_write(summary=valid_summary, writer=writer_valid)
            summary_write(summary=test_summary, writer=writer_test)
            summary_write(summary=train_summary, writer=writer_train)
            
            print('\nTrain Results on Adversarial Attack Examples\n')
            mtypes = ['TNR', 'AUROC', 'DTACC', 'AUPR']
            for mtype in mtypes:
                print(' {mtype:6s}'.format(mtype=mtype), end='')
            print('\n{val:6.2f}'.format(val=100.*train_results['TMP']['TNR']), end='')
            print(' {val:6.2f}'.format(val=100.*train_results['TMP']['AUROC']), end='')
            print(' {val:6.2f}'.format(val=100.*train_results['TMP']['DTACC']), end='')
            print(' {val:6.2f}\n'.format(val=100.*train_results['TMP']['AUPR']), end='')
            
            print('\nValidation Results on Public (Fake) OOD Image Sets\n')
            for mtype in mtypes:
                print(' {mtype:6s}'.format(mtype=mtype), end='')
            print('\n{val:6.2f}'.format(val=100.*valid_results['TMP']['TNR']), end='')
            print(' {val:6.2f}'.format(val=100.*valid_results['TMP']['AUROC']), end='')
            print(' {val:6.2f}'.format(val=100.*valid_results['TMP']['DTACC']), end='')
            print(' {val:6.2f}\n'.format(val=100.*valid_results['TMP']['AUPR']), end='')    
            
            print('\nTest Results on Real OOD Image Sets\n')
            mtypes = ['TNR', 'AUROC', 'DTACC', 'AUPR']
            for mtype in mtypes:
                print(' {mtype:6s}'.format(mtype=mtype), end='')
            print('\n{val:6.2f}'.format(val=100.*test_results['TMP']['TNR']), end='')
            print(' {val:6.2f}'.format(val=100.*test_results['TMP']['AUROC']), end='')
            print(' {val:6.2f}'.format(val=100.*test_results['TMP']['DTACC']), end='')
            print(' {val:6.2f}\n\n'.format(val=100.*test_results['TMP']['AUPR']), end='')


            f.write('{},{},{},{},{},{}\n'.format(adv_noise, random_noise_size, train_results['TMP']['TNR'],train_results['TMP']['AUROC'],train_results['TMP']['DTACC'],train_results['TMP']['AUPR']))
            g.write('{},{},{},{},{},{}\n'.format(adv_noise, random_noise_size, valid_results['TMP']['TNR'],valid_results['TMP']['AUROC'],valid_results['TMP']['DTACC'],valid_results['TMP']['AUPR']))
            h.write('{},{},{},{},{},{}\n'.format(adv_noise, random_noise_size, test_results['TMP']['TNR'],test_results['TMP']['AUROC'],test_results['TMP']['DTACC'],test_results['TMP']['AUPR']))
            
fooled_log.close()

