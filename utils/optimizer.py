import math
import torch

_LR_POLICY = {
    'cosine' : lr_func_cosine,
}


# Implement new optimizer here
def getOptimizer(model, cfg):
    if cfg['optimizer'] == 'sgd':
        return torch.optim.SGD(model.parameters(),
                               lr=cfg['lr'],
                               momentum=cfg['momentum'],
                               weight_decay=cfg['weight_decay'],
                               nesterov=cfg['nesterov'])
    elif cfg['optimizer'] == 'adam':
        return torch.optim.Adam(
            model.parameters(),
            lr=cfg['lr'],
            betas=(0.9, 0.999),
            weight_decay=cfg['weight_decay']
        )
    else:
        raise NotImplementedError(
            "Does not support '{}' optimizer".format(cfg['optimizer'])
        )
        

def set_lr(optimizer, new_lr):
    for param_group in optimizer.param_groups:
        param_group["lr"] = new_lr
        
        
def get_lr_at_epoch(cfg, cur_epoch):
    lr = get_lr_func(cfg['policy'])(cfg, cur_epoch)
    # Perform warm up.
    if cur_epoch < cfg['warm_epoch']:
        lr_start = cfg['warm_lr']
        lr_end = get_lr_func(cfg['policy'])(
            cfg, cfg['warm_epoch']
        )
        alpha = (lr_end - lr_start) / cfg['warm_epoch']
        lr = cur_epoch * alpha + lr_start
    return lr
        
    
def get_lr_func(policy):
    if policy in _LR_POLICY.keys():
        return _LR_POLICY[policy]
    else:
        raise NotImplementedError(
            "Does not support '{}' lr policy".format(policy)
        )
        
# Implement new policy here!
def lr_func_cosine(cfg, cur_epoch)
    return (
            cfg['lr']
            * (math.cos(math.pi * cur_epoch / cfg['max_epoch']) + 1.0)
            * 0.5
        )
