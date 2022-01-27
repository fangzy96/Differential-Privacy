import torch
import torch.nn as nn
# use GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# this is the student class
class Student:

    def __init__(self, model):
        # define parameters
        self.model = model
        self.epoch = 500 # student's training epoch
        self.learning_rate = 0.01
        self.batch_size = 256

    # predict function
    def predict(self, train_loader, test_loader,counts):

        # loss and optimizer
        criter = nn.CrossEntropyLoss() # I use cross entropy loss, also can use NLL loss
        optimizer = torch.optim.SGD(self.model.parameters(),lr=self.learning_rate)

        # train
        i = 0
        for epoch in range(self.epoch):
            for num, (x, y) in enumerate(train_loader):
                i += 1
                x = x.to(device)
                x = x.to(torch.float) # x should be float
                y = y.to(device)
                y = y.to(torch.long) # y should be long

                # forward pass
                outputs = self.model(x)
                # print(len(outputs))
                loss = criter(outputs, y)

                # backward and optimize
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # print loss during training
                if (i+1) % 100 == 0:
                    print('Epoch [{}/{}], loss: {:.4f}'.format(epoch+1,self.epoch,loss.item()))
        # test
        self.model.eval()
        with torch.no_grad():
            correct = 0
            total = 0
            for x, y in test_loader:
                x = x.to(device)
                y = y.to(device)
                outputs = self.model(x)
                _, predict = torch.max(outputs.data,1)
                total += y.size(0)
                correct += (predict == y).sum().item()
            print('Test Accuracy of the model is: {} %'.format(100 * correct / total))
