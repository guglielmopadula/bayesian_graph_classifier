from torch import nn


class LBR(nn.Module):
    def __init__(self,in_features,out_features,drop_prob):
        super().__init__()
        self.lin=nn.Linear(in_features, out_features)
        self.batch=nn.BatchNorm1d(out_features)
        self.relu=nn.ReLU()
        self.dropout=nn.Dropout(drop_prob)
    
    def forward(self,x):
        return self.dropout(self.relu(self.batch(self.lin(x))))

