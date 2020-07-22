from torchvision import transforms as trn

cfg = dict()

# Training config
cfg['finetuning'] = True
cfg['max_epoch'] = 15
cfg['seed'] = 1
cfg['valid_epoch'] = 1  # Epoch term of validation
cfg['ngpu'] = 1

# Log config
cfg['log_step'] = 100  # Step term of log
cfg['exp_root'] = '/home/sr2/Hyeokjun/OOD-saige/results/'
cfg['exp_dir'] = 'wrn_severstal_oe_cifar100'
cfg['ckpt_epoch'] = 5  # Epoch term of saving checkpoint file.

# Load Checkpoint
cfg['load_ckpt'] = '/home/sr2/Hyeokjun/OOD-saige/results/wrn_severstal012_baseline/ckpt/checkpoint_epoch_100.pyth' # ckpt file(.pyth) "absolute path"

# DataLoader config
cfg['dataloader'] = dict()
cfg['dataloader']['batch_size'] = 128
cfg['dataloader']['num_workers'] = 8
cfg['dataloader']['pin_memory'] = True

# In-Dataset config
cfg['in_dataset'] = dict()
cfg['in_dataset']['dataset'] = 'Severstal'
cfg['in_dataset']['targets'] = ['ok','1', '2']
cfg['in_dataset']['train_transform'] = trn.Compose([trn.RandomHorizontalFlip(),
                                         trn.RandomCrop(224),
                                         trn.ToTensor()])
cfg['in_dataset']['valid_transform'] = trn.Compose([trn.CenterCrop(224),
                                         trn.ToTensor()])
cfg['in_dataset']['data_root'] = '/home/sr2/HDD2/Openset/'
cfg['in_dataset']['split_root'] = '/home/sr2/Hyeokjun/OOD-saige/datasets/data_split/'

# Out-Dataset config
cfg['out_dataset'] = dict()
cfg['out_dataset']['dataset'] = 'cifar100'
cfg['out_dataset']['train_transform'] = trn.Compose([trn.RandomHorizontalFlip(),
                                         trn.Resize(224),
                                         trn.ToTensor()])
cfg['out_dataset']['valid_transform'] = trn.Compose([trn.Resize(224),
                                         trn.ToTensor()])
cfg['out_dataset']['data_root'] = '/home/sr2/HDD2/Openset/'
cfg['out_dataset']['split_root'] = '/home/sr2/Hyeokjun/OOD-saige/datasets/data_split/'


# Model config
cfg['model'] = dict()
cfg['model']['network_kind'] = 'wrn224'
cfg['model']['depth'] = 40
cfg['model']['widen_factor'] = 2
cfg['model']['num_classes'] = len(cfg['in_dataset']['targets'])
cfg['model']['drop_rate'] = 0.3

# Loss config
cfg['loss'] = dict()
cfg['loss']['loss'] = 'oe'
cfg['loss']['oe_weight'] = 0.1

# Detector config
cfg['detector'] = dict()
cfg['detector']['detector'] = 'msp'

# Optimizer & scheduler config
cfg['optim'] = dict()
cfg['optim']['max_epoch'] = cfg['max_epoch']
cfg['optim']['optimizer'] = 'sgd'
cfg['optim']['momentum'] = 0.9
cfg['optim']['nesterov'] = True
cfg['optim']['weight_decay'] = 0.0005
cfg['optim']['lr'] = 0.001
cfg['optim']['policy'] = 'cosine'
cfg['optim']['warm_epoch'] = 0 # Warm starting epoch if smaller than zero, no warm starting.
cfg['optim']['warm_lr'] = 0.0 # Warm starting learning rate.


# Additional configuration add here!!