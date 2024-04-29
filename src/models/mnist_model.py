import torch.nn as nn


class MnistModel(nn.Module):
    def __init__(self, base_model):
        super(MnistModel, self).__init__()
        self.base_model = base_model
        self.output_fc = nn.Linear(1000, 10)
        
    def forward(self, x):
        x = self.base_model(x)
        x = self.output_fc(x)
        return x
