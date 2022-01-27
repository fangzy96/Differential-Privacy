import argparse
import copy
import os.path
import numpy as np
from sting.data import Feature, parse_c45
import util
import logreg_model

def logreg(data_path, lamda, noise_type, epsilon, learning_rate,use_cross_validation):
    # get data
    path = os.path.expanduser(data_path).split(os.sep)
    file_base = path[-1]
    root_dir = os.sep.join(path[:-1])
    schema, x, y = parse_c45(file_base, root_dir)
    x = x[:,1:]
    num_features = len(x[0])
    # shuffle data
    x = util.normalization4(x)
    c = np.column_stack((x,y))
    np.random.shuffle(c)
    x = c[:,:-1]
    y = c[:,-1]
    # calculate sensitivity
    sensitivity = util.cal_sensitivity(x)
    # judge the noise type
    if noise_type == 1:
        x = logreg_model.laplace_input(x, sensitivity, epsilon) # laplace noise: x, sensitivity, epsilon
    acc_result_list = []
    # cross validation
    if use_cross_validation:
        data_sample_list, label_sample_list = util.cv_split(x,y,5)
        for h in range(5):
            tmp_data_sample_list = copy.deepcopy(data_sample_list)
            tmp_label_sample_list = copy.deepcopy(label_sample_list)
            print("Cross validation time:" + str(h+1))
            validation_training = tmp_data_sample_list.pop(h)
            validation_label = tmp_label_sample_list.pop(h)
            training_data = np.vstack((tmp_data_sample_list[0],tmp_data_sample_list[1]))
            training_data = np.vstack((training_data,tmp_data_sample_list[2]))
            training_data = np.vstack((training_data,tmp_data_sample_list[3]))
            training_label = np.hstack((tmp_label_sample_list[0],tmp_label_sample_list[1]))
            training_label = np.hstack((training_label,tmp_label_sample_list[2]))
            training_label = np.hstack((training_label,tmp_label_sample_list[3]))
            lr = logreg_model.Logreg(num_features, lamda)
            # train
            lr.fit_input_output_noise(training_data, training_label,learning_rate)

            acc = lr.evaluation_IO(validation_training, validation_label, noise_type, sensitivity, epsilon)
            acc_result_list.append(acc)
    else:
        datasets = ((x, y, x, y),)
        for X_train, y_train, X_test, y_test in datasets:
            lr = logreg_model.Logreg(num_features, lamda)
            # train
            lr.fit_input_output_noise(X_train, y_train,learning_rate)

            acc = lr.evaluation_IO(X_test, y_test, noise_type, sensitivity, epsilon)
            acc_result_list.append(acc)

    acc_result_list: np.ndarray = np.array(acc_result_list)

    acc_average: np.ndarray = np.mean(acc_result_list)
    acc_std: np.ndarray = np.std(acc_result_list)

    print(f'Accuracy: {acc_average:.3f}, {acc_std:.3f}')



if __name__ == '__main__':
    # set the argparse arguments
    parser = argparse.ArgumentParser(description='Run a decision tree algorithm.')
    parser.add_argument('path', metavar='PATH', type=str, help='The path to the data.')
    parser.add_argument('lamda', metavar='lamda', type=float, help='lamda')
    parser.add_argument('noise_type', metavar='noise_type', type=int, help='1:input,2:output')
    parser.add_argument('epsilon', metavar='EPSILON', type=float, help='epsilon')
    parser.add_argument('learning_rate', metavar='learning_rate', type=float, help='learning_rate')
    parser.add_argument('--no-cv', dest='cv', action='store_false',
                        help='Disables cross validation and trains on the full dataset.')
    args = parser.parse_args()

    # if lamda is negative
    if args.noise_type not in [0, 1, 2]:
        raise argparse.ArgumentTypeError('lamda must be an positive number between {1,2}')
    # if epsilon is negative
    if args.epsilon < 0:
        raise argparse.ArgumentTypeError('epsilon must be an positive number')
    # if learning rate is negative
    if args.learning_rate < 0:
        raise argparse.ArgumentTypeError('learning_rate must be an positive number')

    data_path = os.path.expanduser(args.path)
    use_cross_validation = args.cv
    lamda = args.lamda
    noise_type = args.noise_type
    epsilon = args.epsilon
    learning_rate = args.learning_rate
    logreg(data_path, lamda, noise_type, epsilon, learning_rate, use_cross_validation)
