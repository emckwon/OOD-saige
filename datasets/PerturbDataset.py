# -*- coding: utf-8 -*-
import numpy as np
import os
import random
import torch
import torchvision.transforms as trn
import torchvision.transforms.functional as trnF
import torch.nn.functional as F
from PIL import Image
import utils.opencv_functional as cv2f
import cv2
import itertools


class PerturbDataset(torch.utils.data.Dataset):

    def __init__(self, dataset, train_mode=True):
        self.dataset = dataset
        self.num_points = len(self.dataset.data)
        self.train_mode = train_mode

    def __getitem__(self, index):
        x_orig, classifier_target = self.dataset[index]
        
        x_orig = x_orig.numpy()
        x_tf_0 = np.copy(x_orig)
        
        x_tf_90 = np.rot90(x_orig.copy(), k=1, axes=(1,2)).copy()
        x_tf_180 = np.rot90(x_orig.copy(), k=2, axes=(1,2)).copy()
        x_tf_270 = np.rot90(x_orig.copy(), k=3, axes=(1,2)).copy()
        
        #possible_translations = list(itertools.product([0, 8, -8], [0, 8, -8]))
        #num_possible_translations = len(possible_translations)
        #tx, ty = possible_translations[random.randint(0, num_possible_translations - 1)]
        #tx_target = {0: 0, 8: 1, -8: 2}[tx]
        #ty_target = {0: 0, 8: 1, -8: 2}[ty]
        #x_tf_trans = cv2f.affine(np.asarray(x_orig).copy(), 0, (tx, ty), 1, 0, interpolation=cv2.INTER_CUBIC, mode=cv2.BORDER_REFLECT_101)

        return \
            torch.from_numpy(x_tf_0), \
            torch.from_numpy(x_tf_90), \
            torch.from_numpy(x_tf_180), \
            torch.from_numpy(x_tf_270), \
            torch.tensor(classifier_target)
            #trnF.to_tensor(x_tf_trans), \
            #torch.tensor(tx_target), \
            #torch.tensor(ty_target), \
            

    def __len__(self):
        return self.num_points
