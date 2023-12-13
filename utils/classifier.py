import math
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.nn.parameter import Parameter

def get_classifier(params, feature_dim, out_dim_list):
    if params.classifier == 'None':
        return None
    elif params.classifier == 'CosineLinear':
        return nn.ModuleList([CosineLinear(in_features=feature_dim,out_features=out_dim_list[t_id]) for t_id in range(len(out_dim_list))])
    elif params.classifier == 'Linear':
        return nn.ModuleList([nn.Linear(in_features=feature_dim,out_features=out_dim_list[t_id]) for t_id in range(len(out_dim_list))])
    else:
        raise NotImplementedError()

class MultiProtoCosineLinear(nn.Module):
    def __init__(self, in_features, out_features, num_proto=5):
        super(MultiProtoCosineLinear, self).__init__()
        self.num_proto = num_proto
        self.out_features = out_features
        self.weight = Parameter(torch.Tensor(out_features*num_proto, in_features))
        
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)

    def forward(self, input):
        out = F.linear(F.normalize(input, p=2,dim=1), F.normalize(self.weight, p=2, dim=1))
        # out = out.view(-1, self.out_features, self.num_proto).max(dim=-1).values
        return out
        

class CosineLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super(CosineLinear, self).__init__()
        self.weight = Parameter(torch.Tensor(out_features, in_features))
        
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)

    def forward(self, input):
        out = F.linear(F.normalize(input, p=2,dim=1), F.normalize(self.weight, p=2, dim=1))

        return out
