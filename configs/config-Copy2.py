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
cfg['exp_dir'] = 'wrn_cifar10_godin_hc_gi'
cfg['ckpt_epoch'] = 50  # Epoch term of saving checkpoint file.

# Load Checkpoint
cfg['load_ckpt'] = '/home/sr2/Hyeokjun/OOD-saige/results/wrn_cifar10_godin_hc_gi/ckpt/checkpoint_epoch_100.pyth' # ckpt file(.pyth) "absolute path"

# DataLoader config
cfg['dataloader'] = dict()
cfg['dataloader']['batch_size'] = 1
cfg['dataloader']['num_workers'] = 8
cfg['dataloader']['pin_memory'] = True

# In-Dataset config
cfg['in_dataset'] = dict()
cfg['in_dataset']['split'] = 'valid'
cfg['in_dataset']['dataset'] = 'tinyimagenet'
#cfg['in_dataset']['targets'] = ['4']
cfg['in_dataset']['train_transform'] = trn.Compose([trn.ToTensor(),
                                                    trn.ToPILImage(),
                                                    trn.RandomHorizontalFlip(), 
                                                    trn.Resize(32),
                                                    trn.ToTensor()])
cfg['in_dataset']['valid_transform'] = trn.Compose([trn.ToTensor(),
                                                    trn.ToPILImage(),
                                                    trn.Resize(32),
                                                    trn.ToTensor()])
cfg['in_dataset']['data_root'] = '/home/sr2/HDD/Hyeokjun/'
cfg['in_dataset']['split_root'] = '/home/sr2/Hyeokjun/OOD-saige/datasets/data_split/'
# mean = [x / 255 for x in [125.3, 123.0, 113.9]]
# std = [x / 255 for x in [63.0, 62.1, 66.7]]
# cfg['in_dataset']['dataset'] = 'cifar100'
# cfg['in_dataset']['split'] = 'valid'
# cfg['in_dataset']['train_transform'] = trn.Compose([
#                                                      trn.RandomHorizontalFlip(), 
#                                                      trn.Resize(224),
#                                                      trn.ToTensor()])
#                                                     #trn.Normalize(mean, std)])
# cfg['in_dataset']['valid_transform'] = trn.Compose([
#                                                      trn.Resize(224),
#                                                      trn.ToTensor(),
#                                                      trn.Normalize(mean, std)])
# cfg['in_dataset']['data_root'] = '/home/sr2/HDD/Hyeokjun/'


# Out-Dataset config
cfg['out_dataset'] = None
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
# mean = [x / 255 for x in [125.3, 123.0, 113.9]]
# std = [x / 255 for x in [63.0, 62.1, 66.7]]
# cfg['out_dataset'] = dict()
# cfg['out_dataset']['dataset'] = 'tinyimagenet'
# cfg['out_dataset']['split'] = 'valid'
# cfg['out_dataset']['train_transform'] = trn.Compose([trn.ToTensor(),
#                                                      trn.ToPILImage(),
#                                                      trn.RandomHorizontalFlip(), 
#                                                      trn.Resize(224),
#                                                      trn.ToTensor()])
#                                                     #trn.Normalize(mean, std)])
# cfg['out_dataset']['valid_transform'] = trn.Compose([trn.ToTensor(),
#                                                      trn.ToPILImage(),
#                                                      trn.Resize(224),
#                                                      trn.ToTensor(),
#                                                      trn.Normalize(mean, std)])
# #cfg['out_dataset']['data_root'] = '/home/sr2/HDD2/Openset/'
# cfg['out_dataset']['data_root'] = '/home/sr2/HDD/Hyeokjun/'
# cfg['out_dataset']['split_root'] = '/home/sr2/Hyeokjun/OOD-saige/datasets/data_split/'



# Model config
cfg['model'] = dict()
cfg['model']['network_kind'] = 'wrngodin32'
cfg['model']['depth'] = 40
cfg['model']['widen_factor'] = 2
#cfg['model']['num_classes'] = len(cfg['in_dataset']['targets'])
cfg['model']['num_classes'] = 10
cfg['model']['drop_rate'] = 0.3

# model config for GODIN
cfg['model']['H'] = 'hc'
cfg['model']['G'] = 'gi'


# Loss config
cfg['loss'] = dict()
cfg['loss']['loss'] = 'cross_entropy_in_distribution'
#cfg['loss']['ova_target'] = 4
#cfg['loss']['oe_weight'] = 0.1

# Detector config
cfg['detector'] = dict()
cfg['detector']['detector'] = 'msp'

# config for GODIN
cfg['detector']['magnitude'] = 0.05


# Optimizer & scheduler config
cfg['optim'] = dict()
cfg['optim']['max_epoch'] = cfg['max_epoch']
cfg['optim']['optimizer'] = 'sgd'
cfg['optim']['momentum'] = 0.9
cfg['optim']['nesterov'] = True
cfg['optim']['weight_decay'] = 0.0001
cfg['optim']['lr'] = 0.01
cfg['optim']['policy'] = 'cosine'
cfg['optim']['warm_epoch'] = 0 # Warm starting epoch if smaller than zero, no warm starting.
cfg['optim']['warm_lr'] = 0.0 # Warm starting learning rate.


# Additional configuration add here!!
