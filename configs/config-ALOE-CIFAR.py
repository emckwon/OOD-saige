from torchvision import transforms as trn

cfg = dict()

# Training config
cfg['finetuning'] = False
cfg['max_epoch'] = 100
cfg['seed'] = 1
cfg['valid_epoch'] = 1  # Epoch term of validation
cfg['ngpu'] = 1

# Log config
cfg['log_step'] = 100  # Step term of log
cfg['exp_root'] = '/home/sr2/Hyeokjun/OOD-saige/results/'
cfg['exp_dir'] = 'wrn_cifar10_aloe'
cfg['ckpt_epoch'] = 20  # Epoch term of saving checkpoint file.

# Load Checkpoint
cfg['load_ckpt'] = '' # ckpt file(.pyth) "absolute path"

# DataLoader config
cfg['dataloader'] = dict()
cfg['dataloader']['batch_size'] = 128
cfg['dataloader']['num_workers'] = 4
cfg['dataloader']['pin_memory'] = True

# In-Dataset config
cfg['in_dataset'] = dict()
cfg['in_dataset']['split'] = 'train'
cfg['in_dataset']['dataset'] = 'cifar10'
#cfg['in_dataset']['targets'] = ['ok', '1', '2', '3', '4']
cfg['in_dataset']['train_transform'] = trn.Compose([trn.RandomCrop(32),
                                                    trn.RandomHorizontalFlip(), 
                                                    trn.ToTensor()])
cfg['in_dataset']['valid_transform'] = trn.Compose([
                                                    trn.ToTensor()])
cfg['in_dataset']['data_root'] = '/home/sr2/HDD2/Openset/'
cfg['in_dataset']['split_root'] = '/home/sr2/Hyeokjun/OOD-saige/datasets/data_split/'

# Out-Dataset config
cfg['out_dataset'] = dict()
# cfg['out_dataset']['split'] = 'train'
# cfg['out_dataset']['dataset'] = 'SDI/34Ah'
# cfg['out_dataset']['targets'] = ['ok', '1', '2', '3', '4', '5', '6', '7', '8']
# cfg['out_dataset']['train_transform'] = trn.Compose([trn.RandomHorizontalFlip(),
#                                                      trn.Resize(256),
#                                                      trn.RandomCrop(224),
#                                                      trn.ToTensor(),
#                                                      trn.Lambda(lambda x: x.repeat(3, 1, 1))])
# cfg['out_dataset']['valid_transform'] = trn.Compose([trn.Resize(224),
#                                                      trn.ToTensor(),
#                                                      ])
# cfg['out_dataset']['data_root'] = '/home/sr2/HDD2/Openset/'
# cfg['out_dataset']['split_root'] = '/home/sr2/Hyeokjun/OOD-saige/datasets/data_split/'
#cfg['out_dataset'] = dict()
# cfg['out_dataset']['split'] = 'valid'
# cfg['out_dataset']['dataset'] = 'Severstal'
# cfg['out_dataset']['targets'] = ['3', '4']
# cfg['out_dataset']['train_transform'] = trn.Compose([trn.RandomHorizontalFlip(), 
#                                                     trn.RandomCrop(224),
#                                                     trn.ToTensor()])
# cfg['out_dataset']['valid_transform'] = trn.Compose([trn.Resize(224),
#                                                     trn.ToTensor()])
# cfg['out_dataset']['data_root'] = '/home/sr2/HDD2/Openset/'
# cfg['out_dataset']['split_root'] = '/home/sr2/Hyeokjun/OOD-saige/datasets/data_split/'
# cfg['out_dataset'] = dict()
cfg['out_dataset']['dataset'] = 'tinyimagenet'
cfg['out_dataset']['split'] = 'train'
cfg['out_dataset']['train_transform'] = trn.Compose([trn.ToTensor(),
                                                     trn.ToPILImage(),
                                                     trn.RandomCrop(32),
                                                    trn.RandomHorizontalFlip(),
                                                     trn.ToTensor()])
cfg['out_dataset']['valid_transform'] = trn.Compose([trn.ToTensor(),
                                                     trn.ToPILImage(),
                                                     trn.CenterCrop(32),
                                                     trn.ToTensor()])
#cfg['out_dataset']['data_root'] = '/home/sr2/HDD2/Openset/'
cfg['out_dataset']['data_root'] = '/home/sr2/HDD/Hyeokjun/'
cfg['out_dataset']['split_root'] = '/home/sr2/Hyeokjun/OOD-saige/datasets/data_split/'



# Model config
cfg['model'] = dict()
cfg['model']['network_kind'] = 'wrn'
cfg['model']['depth'] = 40
cfg['model']['widen_factor'] = 2
cfg['model']['num_classes'] = 10
cfg['model']['drop_rate'] = 0.0


# Loss config
cfg['loss'] = dict()

#cfg['loss']['ova_target'] = 4
#cfg['loss']['oe_weight'] = 0.1

# Detector config
cfg['detector'] = dict()
cfg['detector']['detector'] = 'msp'


# Optimizer & scheduler config
cfg['optim'] = dict()
cfg['optim']['max_epoch'] = cfg['max_epoch']
cfg['optim']['optimizer'] = 'sgd'
cfg['optim']['momentum'] = 0.9
cfg['optim']['nesterov'] = True
cfg['optim']['weight_decay'] = 0.0001
cfg['optim']['lr'] = 0.1
cfg['optim']['policy'] = 'cosine'
cfg['optim']['warm_epoch'] = 0 # Warm starting epoch if smaller than zero, no warm starting.
cfg['optim']['warm_lr'] = 0.0 # Warm starting learning rate.


# Additional configuration add here!!
# PGD option
cfg['PGD'] = dict()
cfg['PGD']['epsilon'] = 1.0
cfg['PGD']['iters'] = 2
cfg['PGD']['iter_size'] = 1.0

cfg['loss']['loss'] = 'aloe'
cfg['loss']['beta1'] = 1.0
cfg['loss']['beta3'] = 1.0

