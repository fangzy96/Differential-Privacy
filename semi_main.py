import torch
import pandas as pd
from teacher import Teacher
from student import Student
import numpy as np
import fnn_model
import argparse
import os
import util

# use GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

# setting some parameters
batch_size = 256
epochs = 10
classes = 2 # data's classes: volcanoes has have classes
learning_rate = 0.01


def fnn(datapath, num_teacher, epsilon,noise_type):

    # get data
    labeled_dataset, unlabeled_dataset, original_labeled, original_unlabeled, ori_y,num_feature = util.split_unlabeled_data(datapath,epsilon,noise_type)
    unlabeled_loader = torch.utils.data.DataLoader(dataset=unlabeled_dataset,
                                              batch_size=batch_size,
                                              shuffle=False)
    # initialize the model
    model = fnn_model.FNN(num_feature).to(device)
    # initialize the Teacher
    teacher = Teacher(model, num_teacher, epsilon)
    # Teacher training
    teacher.train(labeled_dataset)

    predict = []
    counts = []

    for x, y in unlabeled_loader:

        x = x.to(device)
        y = y.to(device)
        # get teacher results
        output = teacher.predict(x)
        predict.append(output)

    label_res = []
    for i in range(len(predict)):
        for val in predict[i]:
            label_res.append(val.item())
    if noise_type == 1:
        original_unlabeled.iloc[:,-1] = label_res # unlabeled data with pseudolabel
    labeled_and_unlabeled_data = np.vstack((original_labeled, original_unlabeled))
    # get the whole date contains labeled data and pseudo labeled data
    labeled_and_unlabeled_data = pd.DataFrame(labeled_and_unlabeled_data)

    # count = 0
    # for i in range(len(ori_y)):
    #     if ori_y[i] == label_res[i]:
    #         count += 1

    # get new data loader from new data
    train_loader, test_loader = util.get_student_data(labeled_and_unlabeled_data, batch_size)
    ori_y = list(ori_y)
    count = 0
    for i in range(len(ori_y)):
        if ori_y[i] == label_res[i]:
            count += 1
    # print('Certainty of pseudolabel: ',count/len(ori_y))
    # initialize Student model
    stu_model = fnn_model.FNN(num_feature).to(device)
    student = Student(stu_model)
    # test via Student
    student.predict(train_loader, test_loader,counts)


if __name__ == '__main__':
    # set the argparse arguments
    parser = argparse.ArgumentParser(description='Run a decision tree algorithm.')
    parser.add_argument('path', metavar='PATH', type=str, help='The path to the data.')
    parser.add_argument('num_teacher', metavar='num_teacher', type=int, help='num_teacher')
    parser.add_argument('epsilon', metavar='epsilon', type=float, help='epsilon')
    parser.add_argument('noise_type', metavar='noise_type', type=int, help='noise_type')
    args = parser.parse_args()

    # if epsilon is negative
    if args.epsilon < 0:
        raise argparse.ArgumentTypeError('epsilon must be an positive number')
    # if num_teacher is negative
    if args.num_teacher <= 0:
        raise argparse.ArgumentTypeError('epsilon must be an num_teacher number')
    # # if noise type is not in [1,2]
    if args.noise_type not in [1, 2]:
        raise argparse.ArgumentTypeError('noise_type must in [1,2]: 1 is counts perturbing, 2 is input perturbing')

    data_path = os.path.expanduser(args.path)
    num_teacher = args.num_teacher
    epsilon = args.epsilon
    noise_type = args.noise_type
    fnn(data_path, num_teacher, epsilon, noise_type)
