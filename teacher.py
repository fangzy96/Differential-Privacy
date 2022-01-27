import pandas as pd
import torch
import torch.nn as nn
from torch.distributions.laplace import Laplace

# use GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# define Teacher class
class Teacher:

    def __init__(self, model, num_teacher, epsilon):
        # define some parameters
        self.num_teacher = num_teacher
        self.model = model
        self.teachers = {}
        self.load_model()
        self.epsilon = epsilon
        self.epoch = 10
        self.batch_size = 256
        self.learning_rate = 0.01


    def split_data(self, data):

        """
        For non Tensor data
        new_data = []
        data_split_len = int(len(data)/self.num_teacher)
        for num in range(self.num_teacher):
            subset = data.iloc[num*data_split_len:data_split_len*(num+1), :]
            new_data.append(subset)
        new_data = torch.tensor(new_data, dtype=torch.float)
        return new_data
        """

        # split tensor data
        data_split_len = int(len(data) / self.num_teacher)
        i = 0
        index = 0
        res = []
        end = data_split_len * self.num_teacher
        for teacher in range(0, self.num_teacher):
            res.append([])
        for (x, y) in data:
            if (i) % data_split_len == 0 and i != 0:
                index += 1
            res[index].append([x, y])
            i += 1
            if i == end:
                return res
        return res

    # this function is for loading teacher models and name them
    def load_model(self):
        for num in range(self.num_teacher):
            model = self.model
            self.teachers['teacher' + str(num)] = model

    # add noise
    def add_noise_data(self, data):
        if self.epsilon == 0:
            return data
        else:
            noise_tensor = torch.ones(data.size())
            noise = Laplace(noise_tensor, torch.tensor([1/self.epsilon])).sample()
            new_data = data + noise
            return new_data

    # single model training
    def single_model_train(self, data, teacher):
        model = self.teachers[teacher]
        # loss and optimizer
        data = torch.utils.data.DataLoader(dataset=data,batch_size=self.batch_size,shuffle=True)
        criter = nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(model.parameters(),lr=self.learning_rate)
        # train
        i = 0
        for epoch in range(self.epoch):
            for num, (x, y) in enumerate(data):
                x = x.to(device)
                x = x.to(torch.float)
                y = y.to(device)
                y = y.to(torch.long)
                # forward pass
                outputs = model(x)
                loss = criter(outputs, y)
                # backward and optimize
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                if (i+1) % 100 == 0:
                    print('loss: {:.4f}'.format(loss.item()))
                i += 1

    # train data
    def train(self, data):
        subset = self.split_data(data)
        for epoch in range(1, self.epoch + 1):
            print('-----------------------------------------')
            print('Epoch: '+str(epoch))
            current = 0
            for teacher in self.teachers:
                print("This is No. " + str(current+1)+'teacher.')
                self.single_model_train(subset[current], teacher)
                current += 1

    # merge all teachers' results
    def merge(self, voting, batch_size):
        counts = torch.zeros([batch_size, 2])
        voting_res = torch.zeros([self.num_teacher, batch_size])
        model_id = 0
        for teacher in voting:
            i = 0
            for res in voting[teacher]:
                for vote in res:
                    counts[i][vote] += 1
                    voting_res[model_id][i] = vote
                    i += 1
            model_id += 1
        return counts, voting_res

    # predict function
    def predict(self, data):
        pred_res = {}
        for teacher in self.teachers:
            out = []
            output = self.teachers[teacher](data)
            output = output.max(dim=1)[1]
            out.append(output)
            pred_res[teacher] = out

        counts, voting_res = self.merge(pred_res, len(data))
        # add laplace noise to counts
        counts = self.add_noise_data(counts)
        pred = []
        for i in counts:
            pred.append(torch.tensor(i.max(dim=0)[1].long()).clone().detach())
        output = pred
        return output
