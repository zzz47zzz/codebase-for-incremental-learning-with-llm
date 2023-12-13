import torch
import torch.nn as nn

def get_optimizer(params, model, classifier_list):

    # build optimizer
    tg_params = []
    if isinstance(model,nn.Module) and params.method == 'SEQ':
        # If params.SEQ_fix_encoder and params.SEQ_warmup_epoch_before_fix_encoder>0, the encoder will be fixed by setting its learning rate as zero
        if not (params.SEQ_fix_encoder and params.SEQ_warmup_epoch_before_fix_encoder==0):
            tg_params.append({'params': model.parameters(), 'lr': float(params.lr), 'weight_decay': float(params.weight_decay)}) 

    elif isinstance(model,nn.Module):
        tg_params.append({'params': model.parameters(), 'lr': float(params.lr), 'weight_decay': float(params.weight_decay)}) 
    else:
        raise NotImplementedError()

    if isinstance(classifier_list,nn.ModuleList):
        for i in range(len(classifier_list)):
            tg_params.append({'params': classifier_list[i].parameters(), 'lr': float(params.classifier_lr)})

    optimizer = torch.optim.AdamW(tg_params)

    return optimizer