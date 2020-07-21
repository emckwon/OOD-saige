from torchvision import transfroms as trn

cfg = dict()

# Training config
cfg['finetuning'] = False
cfg['max_epoch'] = 100
cfg['seed'] = 1
cfg['valid_epoch'] = 1  # Epoch term of validation
cfg['ngpu'] = 1

# Log config
cfg['log_step'] = 100  # Step term of log
cfg['exp_root'] = '/home/sr2/Hyeokjun/results/'
cfg['exp_dir'] = 'wrn_severstal_classifier'
cfg['load_ckpt'] = '' # ckpt file(.pyth) name
cfg['ckpt_epoch'] = 10  # Epoch term of saving checkpoint file.

# DataLoader config
cfg['dataloader'] = dict()
cfg['dataloader']['batch_size'] = 128
cfg['dataloader']['num_workers'] = 4
cfg['dataloader']['pin_memory'] = True

# In-Dataset config
cfg['in_dataset'] = dict()
cfg['in_dataset']['dataset'] = 'Severstal'
cfg['in_dataset']['targets'] = ['ok', '1', '2', '3', '4']
cfg['in_dataset']['train_transform'] = trn.Compose([trn.RandomHorizontalFilp(),
                                         trn.RandomCrop(224),
                                         trn.ToTensor()])
cfg['in_dataset']['valid_transform'] = trn.Compose([trn.CenterCrop(256),
                                         trn.ToTensor()])
cfg['in_dataset']['data_root'] = '/home/sr2/HDD2/Openset/'
cfg['in_dataset']['split_root'] = '/home/sr2/Hyeokjun/datasets/data_split/'

# Out-Dataset config
cfg['out_dataset'] = dict()
cfg['out_dataset']['dataset'] = 'Severstal'
cfg['out_dataset']['targets'] = []
cfg['out_dataset']['train_transform'] = trn.Compose([trn.RandomHorizontalFilp(),
                                          trn.RandomCrop(224),
                                          trn.ToTensor()])
cfg['out_dataset']['valid_transfrom'] = trn.Compose([trn.CenterCrop(256),
                                          trn.ToTensor()])
cfg['out_dataset']['data_root'] = '/home/sr2/HDD2/Openset/'
cfg['out_dataset']['split_root'] = '/home/sr2/Hyeokjun/datasets/data_split/'

# Model config
cfg['model'] = dict()
cfg['model']['network_kind'] = 'wrn'
cfg['model']['network_depth'] = 40
cfg['model']['network_wide'] = 2
cfg['model']['num_classes'] = 5 # might be len(cfg['in_target'])
cfg['model']['drop_rate'] = 0.0

# Loss config
cfg['loss'] = dict()
cfg['loss']['loss_kind'] = 'cross_entropy'

# Optimizer & scheduler config
cfg['optim'] = dict()
cfg['optim']['optimizer'] = 'sgd'
cfg['optim']['nesterov'] = True
cfg['optim']['weight_decay'] = 0.0
cfg['optim']['lr'] = 0.01
cfg['optim']['policy'] = 'cosine_annealing'
cfg['optim']['warm_epoch'] = -1 # Warm starting epoch if negative, no warm starting.
cfg['optim']['warm_lr'] = 0.0 # Warm starting learning rate.


# Additional configuration add here!!