from __future__ import print_function
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import os
import sys
from config import cfg
import sklearn
from scipy.stats import multivariate_normal

from torch.autograd import Variable
from scipy.spatial.distance import pdist, cdist, squareform

def blockPrint():
    sys.stdout = open(os.devnull, 'w')
def enablePrint():
    sys.stdout = sys.__stdout__
    

# lid of a batch of query points X
def mle_batch(data, batch, k):
    '''
    commpute lid score using data & batch with k-neighbors
    return: a: computed LID score
    '''
    data = np.asarray(data, dtype=np.float32)
    batch = np.asarray(batch, dtype=np.float32)

    k = min(k, len(data)-1)
    f = lambda v: - k / np.sum(np.log(v/v[-1]))
    a = cdist(batch, data)
    a = np.apply_along_axis(np.sort, axis=1, arr=a)[:,1:k+1]
    a = np.apply_along_axis(f, axis=1, arr=a)
    return a

# this function is from https://github.com/xingjunm/lid_adversarial_subspace_detection
def merge_and_generate_labels(X_pos, X_neg):
    """
    merge positve and nagative artifact and generate labels
    return: X: merged samples, 2D ndarray
             y: generated labels (0/1): 2D ndarray same size as X
    """
    X_pos = np.asarray(X_pos, dtype=np.float32)
    X_pos = X_pos.reshape((X_pos.shape[0], -1))

    X_neg = np.asarray(X_neg, dtype=np.float32)
    X_neg = X_neg.reshape((X_neg.shape[0], -1))

    X = np.concatenate((X_pos, X_neg))
    y = np.concatenate((np.ones(X_pos.shape[0]), np.zeros(X_neg.shape[0])))
    y = y.reshape((X.shape[0], 1))

    return X, y

def sample_estimator(model, num_classes, feature_list, train_loader, maximum=100):
    """
    compute sample mean and precision (inverse of covariance)
    return: sample_class_mean: list of class mean
             precision: list of precisions
    """
    import sklearn.covariance
    
    model.eval()
    group_lasso = sklearn.covariance.EmpiricalCovariance(assume_centered=False)
    correct, total = 0, 0
    num_output = len(feature_list)
    num_sample_per_class = np.empty(num_classes)
    num_sample_per_class.fill(0)
    list_features = []
    for i in range(num_output):
        temp_list = []
        for j in range(num_classes):
            temp_list.append(0)
        list_features.append(temp_list)
    count=0
    for data, target in tqdm(train_loader):
        count+=1
        total += data.size(0)
        data = data.cuda()
        with torch.no_grad(): data = Variable(data)
        output, out_features = model.feature_list(data)
        
        # get hidden features
        for i in range(num_output):
            out_features[i] = out_features[i].view(out_features[i].size(0), out_features[i].size(1), -1)
            out_features[i] = torch.mean(out_features[i].data, 2)
            
        # compute the accuracy
        pred = output.data.max(1)[1]
        equal_flag = pred.eq(target.cuda()).cpu()
        correct += equal_flag.sum()
        
        # construct the sample matrix
        for i in range(data.size(0)):
            label = target[i]
            if num_sample_per_class[label] == 0:
                out_count = 0
                for out in out_features:
                    list_features[out_count][label] = out[i].view(1, -1)
                    out_count += 1
            else:
                out_count = 0
                for out in out_features:
                    list_features[out_count][label] \
                    = torch.cat((list_features[out_count][label], out[i].view(1, -1)), 0)
                    out_count += 1                
            num_sample_per_class[label] += 1
        if count-1==maximum: break
            
    sample_class_mean = []
    out_count = 0
    for num_feature in feature_list:
        temp_list = torch.Tensor(num_classes, int(num_feature)).cuda()
        for j in range(num_classes):
            temp_list[j] = torch.mean(list_features[out_count][j], 0)
        sample_class_mean.append(temp_list)
        out_count += 1
        
    precision = []
    covariance = []
    for k in range(num_output):
        X = 0
        for i in range(num_classes):
            if i == 0:
                X = list_features[k][i] - sample_class_mean[k][i]
            else:
                X = torch.cat((X, list_features[k][i] - sample_class_mean[k][i]), 0)
                
        # find inverse            
        group_lasso.fit(X.cpu().numpy())
        temp_precision = group_lasso.precision_
        temp_precision = torch.from_numpy(temp_precision).float().cuda()
        temp_covariance = group_lasso.covariance_
        precision.append(temp_precision)
        covariance.append(temp_covariance)
        
    print('\n Training Accuracy:({:.2f}%)\n'.format(100. * correct / total))
    #print('num_sample per class ', num_sample_per_class)
    return sample_class_mean, precision, covariance


def semi_tied_sample_estimator(model, num_classes, feature_list, train_loader, maximum=100):
    """
    compute sample mean and precision (inverse of covariance)
    return: sample_class_mean: list of class mean
             precision: list of precisions
    """
    import sklearn.covariance
    
    model.eval()
    group_lasso = sklearn.covariance.EmpiricalCovariance(assume_centered=False)
    correct, total = 0, 0
    num_output = len(feature_list)
    num_sample_per_class = np.empty(num_classes)
    num_sample_per_class.fill(0)
    list_features = []
    for i in range(num_output):
        temp_list = []
        for j in range(num_classes):
            temp_list.append(0)
        list_features.append(temp_list)
    count=0
    for data, target in tqdm(train_loader):
        count+=1
        total += data.size(0)
        data = data.cuda()
        with torch.no_grad(): data = Variable(data)
        output, out_features = model.feature_list(data)
        
        # get hidden features
        for i in range(num_output):
            out_features[i] = out_features[i].view(out_features[i].size(0), out_features[i].size(1), -1)
            out_features[i] = torch.mean(out_features[i].data, 2)
            
        # compute the accuracy
        pred = output.data.max(1)[1]
        equal_flag = pred.eq(target.cuda()).cpu()
        correct += equal_flag.sum()
        
        # construct the sample matrix
        for i in range(data.size(0)):
            label = target[i]
            if num_sample_per_class[label] == 0:
                out_count = 0
                for out in out_features:
                    list_features[out_count][label] = out[i].view(1, -1)
                    out_count += 1
            else:
                out_count = 0
                for out in out_features:
                    list_features[out_count][label] \
                    = torch.cat((list_features[out_count][label], out[i].view(1, -1)), 0)
                    out_count += 1                
            num_sample_per_class[label] += 1
        if count-1==maximum: break
            
    sample_class_mean = []
    out_count = 0
    for num_feature in feature_list:
        temp_list = torch.Tensor(num_classes, int(num_feature)).cuda()
        for j in range(num_classes):
            temp_list[j] = torch.mean(list_features[out_count][j], 0)
        sample_class_mean.append(temp_list)
        out_count += 1
        
    precision = []
    for k in range(num_output):
        X = 0
        class_precision = []
        for i in range(num_classes):
            X = list_features[k][i] - sample_class_mean[k][i]       
            # find classwise inverse            
            group_lasso.fit(X.cpu().numpy())
            temp_precision = group_lasso.precision_
            temp_precision = torch.from_numpy(temp_precision).float().cuda()
            class_precision.append(temp_precision)
        class_precision = torch.stack(class_precision, 0)
        precision.append(class_precision)
        
    print('\n Training Accuracy:({:.2f}%)\n'.format(100. * correct / total))

    
    # sample_class_mean: [num_output(layer num), tensor(num_classes, num_features)]
    # precision: [num_output(layer num), tensor(num_classes, preision_mat)]
    return sample_class_mean, precision


# def GaussianMixture_estimator(model, num_classes, feature_list, train_loader, n_components=1, covtype='full', max_iter=20):
    
#     model.eval()
#     correct, total = 0, 0
#     num_output = len(feature_list)
#     num_sample_per_class = np.empty(num_classes)
#     num_sample_per_class.fill(0)
    
    
#     list_features = []
#     for i in range(num_output):
#         temp_list = []
#         for j in range(num_classes):
#             temp_list.append(0)
#         list_features.append(temp_list)
#     count=0
    
#     for data, target in tqdm(train_loader):
#         count+=1
#         total += data.size(0)
#         data = data.cuda()
#         with torch.no_grad(): data = Variable(data)
#         output, out_features = model.feature_list(data)
        
#         # get hidden features
#         for i in range(num_output):
#             out_features[i] = out_features[i].view(out_features[i].size(0), out_features[i].size(1), -1)
#             out_features[i] = torch.mean(out_features[i].data, 2)
            
#         # compute the accuracy
#         pred = output.data.max(1)[1]
#         equal_flag = pred.eq(target.cuda()).cpu()
#         correct += equal_flag.sum()
        
#         # construct the sample matrix
#         for i in range(data.size(0)):
#             label = target[i]
#             if num_sample_per_class[label] == 0:
#                 out_count = 0
#                 for out in out_features:
#                     list_features[out_count][label] = out[i].view(1, -1)
#                     out_count += 1
#             else:
#                 out_count = 0
#                 for out in out_features:
#                     list_features[out_count][label] \
#                     = torch.cat((list_features[out_count][label], out[i].view(1, -1)), 0)
#                     out_count += 1                
#             num_sample_per_class[label] += 1
#         if count-1==maximum: break
            
#     sample_class_mean = []
#     out_count = 0
#     for num_feature in feature_list:
#         #temp_list = torch.Tensor(num_classes, int(num_feature)).cuda()
#         for j in range(num_classes):
#             #temp_list[j] = torch.mean(list_features[out_count][j], 0)
#             estimator = GaussianMixture(n_components=components, covariance_type=cov_type, max_iter=max_iter, random_state=0)
#             estimator.means_init
#         sample_class_mean.append(temp_list)
#         out_count += 1       
    

#     return sample_class_mean, precision




def get_Mahalanobis_score(model, loader, num_classes, outf, out_flag, net_type, sample_mean, precision, covariance, layer_index, magnitude, maximum=100, llh=False):
    '''
    Compute the proposed Mahalanobis confidence score on input dataset
    return: Mahalanobis score from layer_index
    '''
    model.eval()
    Mahalanobis = []
    
    if out_flag == True:
        temp_file_name = '%s/confidence_Ga%s_In.txt'%(outf, str(layer_index))
    else:
        temp_file_name = '%s/confidence_Ga%s_Out.txt'%(outf, str(layer_index))

    
############ Test
    count=0
    g1 = open(temp_file_name, 'w')
    ###
    fooled = 0
    loglikelihood = [0] * num_classes
    num_samples = [0] * num_classes
    ###
    
    for data, target in loader:
        count+=1
#         progress_bar.set_description('Epoch ' + str(epoch))
        data, target = data.cuda(), target.cuda()
        data, target = Variable(data, requires_grad = True), Variable(target)
        
        out_features = model.intermediate_forward(data, layer_index)
        out_features = out_features.view(out_features.size(0), out_features.size(1), -1)
        out_features = torch.mean(out_features, 2)
        
        # compute Mahalanobis score
        gaussian_score = 0
        
        for i in range(num_classes):
            num_samples[i] += (target == i).sum()
            batch_sample_mean = sample_mean[layer_index][i]
            
            ### Compute LLH
            if llh:
                mg = multivariate_normal(mean=batch_sample_mean.detach().cpu(), cov=covariance[layer_index], allow_singular=True)
                #mg = multivariate_normal(mean=batch_sample_mean.detach().cpu(), allow_singular=True)
                #print("======PDF=======",mg.pdf(batch_sample_mean.detach().cpu()))
                for idx, t in enumerate(target):
                    if i == t:
                        #print("======PDF=======",mg.pdf(out_features[idx].detach().cpu()))
                        #print("======LLH=======",mg.logpdf(out_features[idx].detach().cpu()))
                        loglikelihood[i] += mg.logpdf(out_features[idx].detach().cpu())
            ###
            zero_f = out_features.data - batch_sample_mean
            term_gau = -0.5*torch.mm(torch.mm(zero_f, precision[layer_index]), zero_f.t()).diag()
            
            if i == 0:
                gaussian_score = term_gau.view(-1,1)
            else:
                gaussian_score = torch.cat((gaussian_score, term_gau.view(-1,1)), 1)
        
        
        
        # Input_processing
        sample_pred = gaussian_score.max(1)[1]
        batch_sample_mean = sample_mean[layer_index].index_select(0, sample_pred)
        zero_f = out_features - Variable(batch_sample_mean)
        pure_gau = -0.5*torch.mm(torch.mm(zero_f, Variable(precision[layer_index])), zero_f.t()).diag()
        loss = torch.mean(-pure_gau)
        loss.backward()
         
        gradient =  torch.ge(data.grad.data, 0)
        gradient = (gradient.float() - 0.5) * 2
        gradient.index_copy_(1, torch.LongTensor([0]).cuda(), gradient.index_select(1, torch.LongTensor([0]).cuda())/0.5)
        gradient.index_copy_(1, torch.LongTensor([1]).cuda(), gradient.index_select(1, torch.LongTensor([1]).cuda())/0.5)
        gradient.index_copy_(1, torch.LongTensor([2]).cuda(), gradient.index_select(1, torch.LongTensor([2]).cuda())/0.5)
        blockPrint()
        tempInputs = torch.add(data.data, gradient,alpha=-magnitude)
        enablePrint()
 
        with torch.no_grad(): noise_out_features = model. intermediate_forward(Variable(tempInputs), layer_index)
        noise_out_features = noise_out_features.view(noise_out_features.size(0), noise_out_features.size(1), -1)
        noise_out_features = torch.mean(noise_out_features, 2)
        noise_gaussian_score = 0
        for i in range(num_classes):
            batch_sample_mean = sample_mean[layer_index][i]
            zero_f = noise_out_features.data - batch_sample_mean
            term_gau = -0.5*torch.mm(torch.mm(zero_f, precision[layer_index]), zero_f.t()).diag()
            if i == 0:
                noise_gaussian_score = term_gau.view(-1,1)
            else:
                noise_gaussian_score = torch.cat((noise_gaussian_score, term_gau.view(-1,1)), 1)      

        max_noise_gaussian_score, argmax_noise_gaussian_score = torch.max(noise_gaussian_score, dim=1)
        fooled += ((argmax_noise_gaussian_score != target).int()).sum()
        Mahalanobis.extend(max_noise_gaussian_score.cpu().numpy())
        for i in range(data.size(0)):
            g1.write("{}\n".format(max_noise_gaussian_score[i]))
            
        if count==maximum : break

    g1.close()
    #print(torch.Tensor(num_samples))
    return Mahalanobis, fooled, torch.Tensor(loglikelihood) / torch.Tensor(num_samples)


def get_Mahalanobis_score_semi_tied(model, loader, num_classes, outf, out_flag, net_type, sample_mean, precision, layer_index, magnitude, maximum=100):
    '''
    Compute the proposed Mahalanobis confidence score on input dataset
    return: Mahalanobis score from layer_index
    '''
    model.eval()
    Mahalanobis = []
    
    if out_flag == True:
        temp_file_name = '%s/confidence_Ga%s_In.txt'%(outf, str(layer_index))
    else:
        temp_file_name = '%s/confidence_Ga%s_Out.txt'%(outf, str(layer_index))

    
############ Test
    count=0
    g1 = open(temp_file_name, 'w')
    ###
#     unmatch_samples_dist = []
#     unmatch_samples_real_target = []
#     unmatch_samples_fool_target = []
    fooled = 0
    ###
    
    for data, target in loader:
        count+=1
#         progress_bar.set_description('Epoch ' + str(epoch))
        data, target = data.cuda(), target.cuda()
        data, target = Variable(data, requires_grad = True), Variable(target)
        
        out_features = model.intermediate_forward(data, layer_index)
        out_features = out_features.view(out_features.size(0), out_features.size(1), -1)
        out_features = torch.mean(out_features, 2)
        
        # compute Mahalanobis score
        gaussian_score = 0
        for i in range(num_classes):
            batch_sample_mean = sample_mean[layer_index][i]
            zero_f = out_features.data - batch_sample_mean
            term_gau = -0.5*torch.mm(torch.mm(zero_f, precision[layer_index][i]), zero_f.t()).diag()
            if i == 0:
                gaussian_score = term_gau.view(-1,1)
            else:
                gaussian_score = torch.cat((gaussian_score, term_gau.view(-1,1)), 1)
        
        # Input_processing
        sample_pred = gaussian_score.max(1)[1]
        batch_sample_mean = sample_mean[layer_index].index_select(0, sample_pred)
        zero_f = out_features - Variable(batch_sample_mean)
        pure_gau = -0.5*torch.mm(torch.mm(zero_f, Variable(precision[layer_index].mean(0))), zero_f.t()).diag()
        loss = torch.mean(-pure_gau)
        loss.backward()
         
        gradient =  torch.ge(data.grad.data, 0)
        gradient = (gradient.float() - 0.5) * 2
        gradient.index_copy_(1, torch.LongTensor([0]).cuda(), gradient.index_select(1, torch.LongTensor([0]).cuda())/0.5)
        gradient.index_copy_(1, torch.LongTensor([1]).cuda(), gradient.index_select(1, torch.LongTensor([1]).cuda())/0.5)
        gradient.index_copy_(1, torch.LongTensor([2]).cuda(), gradient.index_select(1, torch.LongTensor([2]).cuda())/0.5)
        blockPrint()
        tempInputs = torch.add(data.data, gradient,alpha=-magnitude)
        enablePrint()
 
        with torch.no_grad(): noise_out_features = model. intermediate_forward(Variable(tempInputs), layer_index)
        noise_out_features = noise_out_features.view(noise_out_features.size(0), noise_out_features.size(1), -1)
        noise_out_features = torch.mean(noise_out_features, 2)
        noise_gaussian_score = 0
        for i in range(num_classes):
            batch_sample_mean = sample_mean[layer_index][i]
            zero_f = noise_out_features.data - batch_sample_mean
            term_gau = -0.5*torch.mm(torch.mm(zero_f, precision[layer_index][i]), zero_f.t()).diag()
            if i == 0:
                noise_gaussian_score = term_gau.view(-1,1)
            else:
                noise_gaussian_score = torch.cat((noise_gaussian_score, term_gau.view(-1,1)), 1)      

        max_noise_gaussian_score, argmax_noise_gaussian_score = torch.max(noise_gaussian_score, dim=1)
        fooled += ((argmax_noise_gaussian_score != target).int()).sum()
        Mahalanobis.extend(max_noise_gaussian_score.cpu().numpy())
        for i in range(data.size(0)):
            g1.write("{}\n".format(max_noise_gaussian_score[i]))
            
        if count==maximum : break

    g1.close()
    
    return Mahalanobis, fooled



def get_posterior(model, net_type, test_loader, valid_loader, magnitude, temperature, outf, out_flag, test_count=10000):
    '''
    Compute the maximum value of (processed) posterior distribution - ODIN
    return: null
    '''
    criterion = nn.CrossEntropyLoss()
    model.eval()
    total = 0
    if out_flag == True:
        temp_file_name_val = '%s/confidence_PoV_In.txt'%(outf)
        temp_file_name_test = '%s/confidence_PoT_In.txt'%(outf)
    else:
        temp_file_name_val = '%s/confidence_PoV_Out.txt'%(outf)
        temp_file_name_test = '%s/confidence_PoT_Out.txt'%(outf)
        
    g = open(temp_file_name_val, 'w')
    f = open(temp_file_name_test, 'w')
    
    count=0
#     print(" ** TEST SET CALCULATING ** ")
    for data, _ in test_loader:
        count+=1
        if(count>test_count): break
        total += data.size(0)
        data = data.cuda()
        data = Variable(data, requires_grad = True)
        batch_output = model(data)
            
        # temperature scaling
        outputs = batch_output / temperature
        labels = outputs.data.max(1)[1]
        labels = Variable(labels)
        loss = criterion(outputs, labels)
        loss.backward()
         
        # Normalizing the gradient to binary in {0, 1}
        gradient =  torch.ge(data.grad.data, 0)
        gradient = (gradient.float() - 0.5) * 2
        gradient.index_copy_(1, torch.LongTensor([0]).cuda(), gradient.index_select(1, torch.LongTensor([0]).cuda())/cfg['std'][0])
        gradient.index_copy_(1, torch.LongTensor([1]).cuda(), gradient.index_select(1, torch.LongTensor([1]).cuda())/cfg['std'][1])
        gradient.index_copy_(1, torch.LongTensor([2]).cuda(), gradient.index_select(1, torch.LongTensor([2]).cuda())/cfg['std'][2])
            
        blockPrint()
        tempInputs = torch.add(data.data, gradient,alpha=-magnitude)
        enablePrint()
        with torch.no_grad(): outputs = model(Variable(tempInputs))
        outputs = outputs / temperature
        soft_out = F.softmax(outputs, dim=1)
        soft_out, _ = torch.max(soft_out.data, dim=1)
        
        for i in range(data.size(0)):
            f.write("{}\n".format(soft_out[i]))  # Write to Test
            
#     print(" ** VALIDATION SET CALCULATING ** ")

    count=0
    for data, _ in valid_loader:
        count+=1
        if(count>test_count): break

        total += data.size(0)
        data = data.cuda()
        data = Variable(data, requires_grad = True)
        batch_output = model(data)
            
        # temperature scaling
        outputs = batch_output / temperature
        labels = outputs.data.max(1)[1]
        labels = Variable(labels)
        loss = criterion(outputs, labels)
        loss.backward()
         
        # Normalizing the gradient to binary in {0, 1}
        gradient =  torch.ge(data.grad.data, 0)
        gradient = (gradient.float() - 0.5) * 2
        gradient.index_copy_(1, torch.LongTensor([0]).cuda(), gradient.index_select(1, torch.LongTensor([0]).cuda())/cfg['std'][0])
        gradient.index_copy_(1, torch.LongTensor([1]).cuda(), gradient.index_select(1, torch.LongTensor([1]).cuda())/cfg['std'][1])
        gradient.index_copy_(1, torch.LongTensor([2]).cuda(), gradient.index_select(1, torch.LongTensor([2]).cuda())/cfg['std'][2])
        blockPrint()
        tempInputs = torch.add(data.data, gradient,alpha=-magnitude)
        enablePrint()
        with torch.no_grad(): outputs = model(Variable(tempInputs))
        outputs = outputs / temperature
        soft_out = F.softmax(outputs, dim=1)
        soft_out, _ = torch.max(soft_out.data, dim=1)
        
        for i in range(data.size(0)):
            g.write("{}\n".format(soft_out[i],'.4f'))  # Write to Test

    f.close()
    g.close()
    
    return count
    
def get_Mahalanobis_score_adv(model, test_data, test_label, num_classes, outf, net_type, sample_mean, precision, covariance, layer_index, magnitude):
    '''
    Compute the proposed Mahalanobis confidence score on adversarial samples
    return: Mahalanobis score from layer_index
    '''
    model.eval()
    Mahalanobis = []
    batch_size = 1
    total = 0
    fooled = 0
#     print(test_data.size(0))
    for data_index in range(test_data.size(0)):
        target = test_label[total : total + batch_size].cuda()
        data = test_data[total : total + batch_size].cuda()
        total += batch_size
        data, target = Variable(data, requires_grad = True), Variable(target)
        
        out_features = model.intermediate_forward(data, layer_index)
        out_features = out_features.view(out_features.size(0), out_features.size(1), -1)
        out_features = torch.mean(out_features, 2)
        
        # compute Mahalanobis score
        gaussian_score = 0
        for i in range(num_classes):
            batch_sample_mean = sample_mean[layer_index][i]
            zero_f = out_features.data - batch_sample_mean
            term_gau = -0.5*torch.mm(torch.mm(zero_f, precision[layer_index]), zero_f.t()).diag()
            if i == 0:
                gaussian_score = term_gau.view(-1,1)
            else:
                gaussian_score = torch.cat((gaussian_score, term_gau.view(-1,1)), 1)
        
        # Input_processing
        sample_pred = gaussian_score.max(1)[1]
        batch_sample_mean = sample_mean[layer_index].index_select(0, sample_pred)
        zero_f = out_features - Variable(batch_sample_mean)
        pure_gau = -0.5*torch.mm(torch.mm(zero_f, Variable(precision[layer_index])), zero_f.t()).diag()
        loss = torch.mean(-pure_gau)
        loss.backward()
         
        gradient =  torch.ge(data.grad.data, 0)
        gradient = (gradient.float() - 0.5) * 2
        gradient.index_copy_(1, torch.LongTensor([0]).cuda(), gradient.index_select(1, torch.LongTensor([0]).cuda())/cfg['std'][0])
        gradient.index_copy_(1, torch.LongTensor([1]).cuda(), gradient.index_select(1, torch.LongTensor([1]).cuda())/cfg['std'][1])
        gradient.index_copy_(1, torch.LongTensor([2]).cuda(), gradient.index_select(1, torch.LongTensor([2]).cuda())/cfg['std'][2])
        
        tempInputs = torch.add(data.data, gradient,alpha=-magnitude)
 
        with torch.no_grad(): noise_out_features = model.intermediate_forward(Variable(tempInputs), layer_index)
        noise_out_features = noise_out_features.view(noise_out_features.size(0), noise_out_features.size(1), -1)
        noise_out_features = torch.mean(noise_out_features, 2)
        noise_gaussian_score = 0
        for i in range(num_classes):
            batch_sample_mean = sample_mean[layer_index][i]
            zero_f = noise_out_features.data - batch_sample_mean
            term_gau = -0.5*torch.mm(torch.mm(zero_f, precision[layer_index]), zero_f.t()).diag()
            if i == 0:
                noise_gaussian_score = term_gau.view(-1,1)
            else:
                noise_gaussian_score = torch.cat((noise_gaussian_score, term_gau.view(-1,1)), 1)      
        
        noise_gaussian_score, arg_noise_gaussian_score = torch.max(noise_gaussian_score, dim=1)
        fooled += ((arg_noise_gaussian_score != target).int()).sum()
        Mahalanobis.extend(noise_gaussian_score.cpu().numpy())
#         print(Mahalanobis)

    return Mahalanobis, fooled


def get_Mahalanobis_score_adv_semi_tied(model, test_data, test_label, num_classes, outf, net_type, sample_mean, precision, layer_index, magnitude):
    '''
    Compute the proposed Mahalanobis confidence score on adversarial samples
    return: Mahalanobis score from layer_index
    '''
    model.eval()
    Mahalanobis = []
    batch_size = 1
    total = 0
    fooled = 0
#     print(test_data.size(0))
    for data_index in range(test_data.size(0)):
        target = test_label[total : total + batch_size].cuda()
        data = test_data[total : total + batch_size].cuda()
        total += batch_size
        data, target = Variable(data, requires_grad = True), Variable(target)
        
        out_features = model.intermediate_forward(data, layer_index)
        out_features = out_features.view(out_features.size(0), out_features.size(1), -1)
        out_features = torch.mean(out_features, 2)
        
        gaussian_score = 0
        for i in range(num_classes):
            batch_sample_mean = sample_mean[layer_index][i]
            zero_f = out_features.data - batch_sample_mean
            term_gau = -0.5*torch.mm(torch.mm(zero_f, precision[layer_index][i]), zero_f.t()).diag()
            if i == 0:
                gaussian_score = term_gau.view(-1,1)
            else:
                gaussian_score = torch.cat((gaussian_score, term_gau.view(-1,1)), 1)
        
        # Input_processing
        sample_pred = gaussian_score.max(1)[1]
        batch_sample_mean = sample_mean[layer_index].index_select(0, sample_pred)
        zero_f = out_features - Variable(batch_sample_mean)
        pure_gau = -0.5*torch.mm(torch.mm(zero_f, Variable(precision[layer_index].mean(0))), zero_f.t()).diag()
        loss = torch.mean(-pure_gau)
        loss.backward()
         
        gradient =  torch.ge(data.grad.data, 0)
        gradient = (gradient.float() - 0.5) * 2
        gradient.index_copy_(1, torch.LongTensor([0]).cuda(), gradient.index_select(1, torch.LongTensor([0]).cuda())/cfg['std'][0])
        gradient.index_copy_(1, torch.LongTensor([1]).cuda(), gradient.index_select(1, torch.LongTensor([1]).cuda())/cfg['std'][1])
        gradient.index_copy_(1, torch.LongTensor([2]).cuda(), gradient.index_select(1, torch.LongTensor([2]).cuda())/cfg['std'][2])
        
        tempInputs = torch.add(data.data, gradient,alpha=-magnitude)
 
        with torch.no_grad(): noise_out_features = model.intermediate_forward(Variable(tempInputs), layer_index)
        noise_out_features = noise_out_features.view(noise_out_features.size(0), noise_out_features.size(1), -1)
        noise_out_features = torch.mean(noise_out_features, 2)
        noise_gaussian_score = 0
        for i in range(num_classes):
            batch_sample_mean = sample_mean[layer_index][i]
            zero_f = noise_out_features.data - batch_sample_mean
            term_gau = -0.5*torch.mm(torch.mm(zero_f, precision[layer_index][i]), zero_f.t()).diag()
            if i == 0:
                noise_gaussian_score = term_gau.view(-1,1)
            else:
                noise_gaussian_score = torch.cat((noise_gaussian_score, term_gau.view(-1,1)), 1)      
        
        noise_gaussian_score, arg_noise_gaussian_score = torch.max(noise_gaussian_score, dim=1)
        fooled += ((arg_noise_gaussian_score != target).int()).sum()
        Mahalanobis.extend(noise_gaussian_score.cpu().numpy())
#         print(Mahalanobis)
    return Mahalanobis, fooled


def get_LID(model, test_clean_data, test_adv_data, test_noisy_data, test_label, num_output):
    '''
    Compute LID score on adversarial samples
    return: LID score
    '''
    model.eval()  
    total = 0
    batch_size = 100
    
    LID, LID_adv, LID_noisy = [], [], []    
    overlap_list = [10, 20, 30, 40, 50, 60, 70, 80, 90]
    for i in overlap_list:
        LID.append([])
        LID_adv.append([])
        LID_noisy.append([])
        
    for data_index in range(int(np.floor(test_clean_data.size(0)/batch_size))):
        data = test_clean_data[total : total + batch_size].cuda()
        adv_data = test_adv_data[total : total + batch_size].cuda()
        noisy_data = test_noisy_data[total : total + batch_size].cuda()
        target = test_label[total : total + batch_size].cuda()

        total += batch_size
        with torch.no_grad(): data, target = Variable(data), Variable(target)
        
        output, out_features = model.module.feature_list(data)
        X_act = []
        for i in range(num_output):
            out_features[i] = out_features[i].view(out_features[i].size(0), out_features[i].size(1), -1)
            out_features[i] = torch.mean(out_features[i].data, 2)
            X_act.append(np.asarray(out_features[i], dtype=np.float32).reshape((out_features[i].size(0), -1)))
        
        with torch.no_grad(): output, out_features = model.module.feature_list(Variable(adv_data))
        X_act_adv = []
        for i in range(num_output):
            out_features[i] = out_features[i].view(out_features[i].size(0), out_features[i].size(1), -1)
            out_features[i] = torch.mean(out_features[i].data, 2)
            X_act_adv.append(np.asarray(out_features[i], dtype=np.float32).reshape((out_features[i].size(0), -1)))

        with torch.no_grad(): output, out_features = model.module.feature_list(Variable(noisy_data))
        X_act_noisy = []
        for i in range(num_output):
            out_features[i] = out_features[i].view(out_features[i].size(0), out_features[i].size(1), -1)
            out_features[i] = torch.mean(out_features[i].data, 2)
            X_act_noisy.append(np.asarray(out_features[i], dtype=np.float32).reshape((out_features[i].size(0), -1)))
        
        # LID
        list_counter = 0 
        for overlap in overlap_list:
            LID_list = []
            LID_adv_list = []
            LID_noisy_list = []

            for j in range(num_output):
                lid_score = mle_batch(X_act[j], X_act[j], k = overlap)
                lid_score = lid_score.reshape((lid_score.shape[0], -1))
                lid_adv_score = mle_batch(X_act[j], X_act_adv[j], k = overlap)
                lid_adv_score = lid_adv_score.reshape((lid_adv_score.shape[0], -1))
                lid_noisy_score = mle_batch(X_act[j], X_act_noisy[j], k = overlap)
                lid_noisy_score = lid_noisy_score.reshape((lid_noisy_score.shape[0], -1))
                
                LID_list.append(lid_score)
                LID_adv_list.append(lid_adv_score)
                LID_noisy_list.append(lid_noisy_score)

            LID_concat = LID_list[0]
            LID_adv_concat = LID_adv_list[0]
            LID_noisy_concat = LID_noisy_list[0]

            for i in range(1, num_output):
                LID_concat = np.concatenate((LID_concat, LID_list[i]), axis=1)
                LID_adv_concat = np.concatenate((LID_adv_concat, LID_adv_list[i]), axis=1)
                LID_noisy_concat = np.concatenate((LID_noisy_concat, LID_noisy_list[i]), axis=1)
                
            LID[list_counter].extend(LID_concat)
            LID_adv[list_counter].extend(LID_adv_concat)
            LID_noisy[list_counter].extend(LID_noisy_concat)
            list_counter += 1
            
    return LID, LID_adv, LID_noisy

