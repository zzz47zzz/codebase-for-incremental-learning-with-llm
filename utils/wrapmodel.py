import torch.nn as nn

class WrapModel(nn.Module):
    '''
        We use a wrapper for a backbone and a classifier 
        because Deepspeed only support one nn.Mudule() when calling accelerate.prepare().
    '''
    def __init__(self, model, classifier_list) -> None:
        super().__init__()
        self.model = model
        self.classifier_list = classifier_list