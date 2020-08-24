from torchvision import transforms as trn

cfg = dict()


# DataLoader config
cfg['dataloader'] = dict()
cfg['dataloader']['num_workers'] = 4
cfg['dataloader']['pin_memory'] = True

# In-Dataset config
cfg['estimate_dataset'] = dict()
cfg['estimate_dataset']['batch_size'] = 1
cfg['estimate_dataset']['split'] = 'train'
cfg['estimate_dataset']['dataset'] = 'Daeduck'
cfg['estimate_dataset']['targets'] = ['OK', 'bonding', 'bump', 'dent', 'hole']
#cfg['in_dataset']['targets'] = ['OK', 'bonding', 'bump', 'dent', 'hole', 'open', 'short', 'slit']
cfg['estimate_dataset']['train_transform'] = trn.Compose([
                                                    trn.ToTensor(),
                                                    ])
cfg['estimate_dataset']['valid_transform'] = trn.Compose([
                                                    trn.ToTensor(),
                                                   ])
cfg['estimate_dataset']['data_root'] = '/home/sr2/HDD2/Openset/'
cfg['estimate_dataset']['split_root'] = '/home/sr2/Hyeokjun/OOD-saige/datasets/data_split/'
