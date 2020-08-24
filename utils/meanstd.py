import torch
from torchvision import datasets, transforms as T
import sys
sys.path.append('./')
from dataset_config import cfg
from tqdm import tqdm
import numpy as numpy
from datasets.data_loader import getDataLoader

loader = getDataLoader(ds_cfg=cfg['estimate_dataset'],
                                    dl_cfg=cfg['dataloader'],
                                    split="train")
means = []
stds = []
mins=[]
maxs=[]
count=0
def meanstd(loader):
    for img in tqdm(loader):
#         try:
        #     print(img)
        rdata=img[0].data.numpy()[0][0]
        gdata=img[0].data.numpy()[0][1]
        bdata=img[0].data.numpy()[0][2]

        rmean=numpy.mean(rdata)
        gmean=numpy.mean(gdata)
        bmean=numpy.mean(bdata)
        rstd=numpy.std(rdata)
        gstd=numpy.std(gdata)
        bstd=numpy.std(bdata)

        mean=[rmean, gmean, bmean]
        std=[rstd,gstd,bstd]
        means.append(mean)
        stds.append(std)
#         except Exception:
#             pass

    mean = numpy.mean(means,axis=0)
    std = numpy.mean(stds,axis=0)
    print(mean, std)
    return mean, std

def minmax(loader):
    for img in tqdm(loader):
#         try:
        rmin=numpy.amin(img[0].data.numpy()[0][0])
        gmin=numpy.amin(img[0].data.numpy()[0][1])
        bmin=numpy.amin(img[0].data.numpy()[0][2])
        rmax=numpy.amax(img[0].data.numpy()[0][0])
        gmax=numpy.amax(img[0].data.numpy()[0][1])
        bmax=numpy.amax(img[0].data.numpy()[0][2])

        minpixels=min([rmin,gmin,bmin])
        maxpixels=max([rmax,gmax,bmax])

        mins.append(minpixels)
        maxs.append(maxpixels)
#         except Excpetion:
#             pass
        

    minpixel = min(mins)
    maxpixel = max(maxs)
    print(minpixel,maxpixel)
    return minpixel, maxpixel

# meanstd(loader)
meanstd(loader)
minmax(loader)




