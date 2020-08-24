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
cfg['exp_dir'] = 'res18_opengan_HBTNUDE_isomax_2-2_3'
cfg['ckpt_epoch'] = 10  # Epoch term of saving checkpoint file.

# Load Checkpoint
cfg['f_load_ckpt'] = '/home/sr2/Hyeokjun/OOD-saige/results/res18_pretrained_HBTNUDE_isomax_2-2/ckpt/checkpoint_epoch_500.pyth'
cfg['g_load_ckpt'] = ''
cfg['d_load_ckpt'] = ''
 # ckpt file(.pyth) "absolute path"

# DataLoader config
cfg['dataloader'] = dict()

cfg['dataloader']['num_workers'] = 4
cfg['dataloader']['pin_memory'] = True

# In-Dataset config
cfg['in_dataset'] = dict()
cfg['in_dataset']['batch_size'] = 8
#cfg['in_dataset']['split'] = 'valid'
cfg['in_dataset']['dataset'] = 'HBT/NUDE'
cfg['in_dataset']['targets'] = ['Scratch', 'ND', 'Dent']
cfg['in_dataset']['train_transform'] = trn.Compose([trn.RandomHorizontalFlip(),
                                                    trn.Resize(256),
                                                    trn.ToTensor()])
cfg['in_dataset']['valid_transform'] = trn.Compose([trn.Resize(256),
                                                    trn.ToTensor()])
cfg['in_dataset']['data_root'] = '/home/sr2/HDD2/Openset/'
cfg['in_dataset']['split_root'] = '/home/sr2/Hyeokjun/OOD-saige/datasets/data_split/'

cfg['in_dataset']['img_size'] = 256

# Out-Dataset config
cfg['out_dataset'] = None
# cfg['out_dataset'] = dict()
# cfg['out_dataset']['batch_size'] = 16
# cfg['out_dataset']['split'] = 'train'
# cfg['out_dataset']['dataset'] = 'HBT/NUDE'
# cfg['out_dataset']['targets'] = ['Electrolyte']
# cfg['out_dataset']['train_transform'] = trn.Compose([trn.Resize(224),
#                                                     trn.ToTensor()])
# cfg['out_dataset']['valid_transform'] = trn.Compose([trn.Resize(224),
#                                                     trn.ToTensor()])
# cfg['out_dataset']['data_root'] = '/home/sr2/HDD2/Openset/'
# cfg['out_dataset']['split_root'] = '/home/sr2/Hyeokjun/OOD-saige/datasets/data_split/'
# mean = [x / 255 for x in [125.3, 123.0, 113.9]]
# std = [x / 255 for x in [63.0, 62.1, 66.7]]
# # cfg['out_dataset'] = dict()
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
cfg['model']['feature_extractor'] = dict()
cfg['model']['feature_extractor']['network_kind'] = 'pisomax'
cfg['model']['feature_extractor']['num_classes'] = 3
cfg['model']['feature_extractor']['pretrained'] = 'resnet18'
cfg['model']['feature_extractor']['freeze'] = -1
cfg['model']['feature_extractor']['alpha'] = 1

cfg['model']['z_dim'] = 128
cfg['model']['g_conv_dim'] = 64
cfg['model']['d_conv_dim'] = 64



# model config for GODIN
# cfg['model']['H'] = 'hc'
# cfg['model']['G'] = 'gi'


# Loss config
cfg['loss'] = dict()
cfg['loss']['adv_loss'] = 'wgan-gp'
#cfg['loss']['ova_target'] = 4
#cfg['loss']['oe_weight'] = 0.1

# Detector config
cfg['detector'] = dict()
cfg['detector']['detector'] = 'msp'
cfg['detector']['magnitude'] = 0.1


# Optimizer & scheduler config
cfg['optim'] = dict()
cfg['optim']['max_epoch'] = cfg['max_epoch']
cfg['optim']['optimizer'] = 'sgd'
cfg['optim']['momentum'] = 0.9
cfg['optim']['nesterov'] = True
cfg['optim']['weight_decay'] = 0.0001
cfg['optim']['lr'] = 0.001
cfg['optim']['policy'] = 'cosine'
cfg['optim']['warm_epoch'] = 0 # Warm starting epoch if smaller than zero, no warm starting.
cfg['optim']['warm_lr'] = 0.0 # Warm starting learning rate.

cfg['optim']['g_lr'] = 0.0001
cfg['optim']['d_lr'] = 0.0001
cfg['optim']['beta1'] = 0
cfg['optim']['beta2'] = 0.999
cfg['optim']['lambda_gp'] = 0.01

cfg['PGD'] = None
# Additional configuration add here!!
