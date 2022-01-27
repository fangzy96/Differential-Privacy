import torch.nn as nn


class FNN(nn.Module):
    def __init__(self, num_feature):
        super(FNN, self).__init__()

        self.layer1 = nn.Sequential(
            # After experiments, Batch Norm has better performance than Sigmoid() and ReLU()nn.Dropout(1000),
            nn.BatchNorm1d(num_feature),
            nn.Linear(num_feature,1000),
            # After experiments, Tanh() has better performance than Sigmoid() and ReLU()
            nn.Tanh()
        )
        self.layer2 = nn.Sequential(
            nn.Linear(1000,1000),
            nn.Tanh()
        )
        self.layer3 = nn.Sequential(
            nn.Linear(1000,128),
            nn.Tanh()
        )
        self.layer4 = nn.Sequential(
            nn.Linear(128,2),
            nn.Tanh()
        )

        self.softmax = nn.Softmax(dim=1)

    def forward(self,x):

        # print(len(x))
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.softmax(out)

        return out
