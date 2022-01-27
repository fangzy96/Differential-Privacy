from torch.utils import data
import torch

# this is the dataset for torch
class dataset(data.Dataset):
    def __init__(self,data):
        self.data = data
        self.feature = self.data.iloc[:,0:-1].values
        self.label = self.data.iloc[:, -1].values # label is the last column
        self.len = len(data)

    def __getitem__(self, item):
        feature = torch.tensor(self.feature[item],dtype=torch.float)
        label = torch.tensor(self.label[item],dtype=torch.float)
        return feature, label

    def __len__(self):
        return self.len
