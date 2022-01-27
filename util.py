from numba import jit
from torch.utils import data
import pandas as pd
import torch
import numpy as np
import torch_dataset
from sklearn.preprocessing import Normalizer, MinMaxScaler


# sigmoid function
def sigmoid(num):
    sig_res = 1 / (1 + np.exp(-num))
    return sig_res

@jit(nopython=True, fastmath = True)
def confusion_matrix(y: np.ndarray, y_hat: np.ndarray):
    # to calculate TP, FN, FP, TN
    TP = 0
    FN = 0
    FP = 0
    TN = 0

    for i in range(0, len(y)-1):
        if y_hat[i] == 1:
            if y[i] == 1:
                TP += 1
            else:
                FP += 1
        else:
            if y[i] == 0:
                TN += 1
            else:
                FN += 1

    return TP, FP, FN, TN

@jit(nopython=True, fastmath = True)
def fp_rate(y: np.ndarray, y_hat: np.ndarray) -> float:
    # y: True labels.
    # y_hat: Predicted labels.
    fp_num = ((y == 0) & (y_hat == 1)).sum()
    tn_num = ((y == 0) & (y_hat == 0)).sum()
    return fp_num / (fp_num + tn_num)

def accuracy(y: np.ndarray, y_hat: np.ndarray) -> float:
    # y: True labels.
    # y_hat: Predicted labels.
    if y.size != y_hat.size:
        raise ValueError('y and y_hat must be the same shape/size!')
    n = y.size # (TP + FP + TN + FN)

    return (y == y_hat).sum() / n # (TP + TN)/(TP + FP + TN + FN)


def precision(y: np.ndarray, y_hat: np.ndarray) -> float:
    # y: True labels.
    # y_hat: Predicted labels.
    tp_num = ((y == 1) & (y_hat == 1)).sum()
    fp_num = ((y == 0) & (y_hat == 1)).sum()
    return tp_num / (tp_num + fp_num)


def recall(y: np.ndarray, y_hat: np.ndarray) -> float:
    # y: True labels.
    # y_hat: Predicted labels.
    tp_num = ((y == 1) & (y_hat == 1)).sum()
    fn_num = ((y == 1) & (y_hat == 0)).sum()
    return tp_num / (tp_num + fn_num)

@jit(nopython=True, fastmath = True)
def specificity(y: np.ndarray, y_hat: np.ndarray) -> float:
    # y: True labels.
    # y_hat: Predicted labels.
    if y.size != y_hat.size:
        raise ValueError('y and y_hat must be the same shape/size!')
    TN = 0
    FP = 0

    for i in range(0, len(y)):
        if y[i] == 0 and y_hat[i] == 0:
            TN += 1
        elif y[i] == 0 and y_hat[i] == 1:
            FP += 1

    return TN / (TN + FP)

# calculate roc pairs
def roc_curve_pairs(y: np.ndarray, confidence: np.ndarray):

    thresholds = np.arange(0.0, 1.001, 0.001)
    result = []
    for thresh in thresholds:
        y_hat=(confidence > thresh)
        y_hat=np.where(y_hat == True, 1, 0)
        result.append([fp_rate(y, y_hat), recall(y, y_hat)])
    return result


# to calculate the AUC depending on pairs from ROC
def auc(y: np.ndarray, confidence: np.ndarray) -> float:

    pair = roc_curve_pairs(y, confidence)
    area = 0
    for i in range(1, len(pair)):
        x_diff = abs(pair[i][0] - pair[i-1][0])
        area = area + (pair[i-1][1] + pair[i][1])/2 * x_diff
    return area


def cv_split(X: np.ndarray, y: np.ndarray, folds: int, stratified: bool = False):
    y_R = np.transpose(y)
    Data = np.column_stack((X,y_R))
    Data = Data[Data[:,-1].argsort()] # Sort Data by the last column
    Rows = len(X) # Calculate how many rows there is in the Array

    # For loop, I want the index of the row when I can split 0 and 1
    result_x =[]
    result_y =[]
    for i in range(Rows-1):
        if Data[i,-1] != Data[i+1, -1]:
            index =i

    # index is the index of the last 0 row
    # split the Data Array by the index row, into DataArray0 and DataArray1, 0 and 1 are labels
    Data_0 = Data[0:index+1,:]
    Data_1 = Data[index+1:Rows,:]
    Ran_Data0 = np.random.permutation(Data_0)
    Ran_Data1 = np.random.permutation(Data_1)

    # Split the Data0 array and Data1 array into 5 parts and merge them separately
    Split_D0 = np.array_split(Ran_Data0,folds,axis = 0)
    Split_D1 = np.array_split(Ran_Data1,folds,axis = 0)

    # Split is completed
    for i in range(folds):
        a = np.vstack((Split_D0[i], Split_D1[i]))
        result_x.append(a[:,0:len(X[0])])
        result_y.append(a[:,-1])

    return result_x,result_y

# normalization func 1
def normalization1(x: np.ndarray):
    mu = np.mean(x, axis = 0)
    sigma = np.std(x, axis = 0)
    return (x - mu) / sigma

# normalization func 2
def normalization2(x: np.ndarray):
    b = len(x[0])
    vector = []
    for v in x:
        vector.append(np.linalg.norm(v))
    max_val = max(vector)
    min_val = min(vector)
    res = (x - max_val)/((min_val-max_val)*np.sqrt(b))
    return res

# normalization func 3
def normalization3(x: np.ndarray):
    transformer = Normalizer().fit(x)  # fit does nothing.
    return transformer.transform(x)

# normalization func 4
def normalization4(x: np.ndarray):
    min_max_scaler = MinMaxScaler()
    X_train_minmax = min_max_scaler.fit_transform(x)
    return X_train_minmax

# pre processing function
def pre_processing(x: np.ndarray):
    number_of_samples = len(x)
    fix_feature_b = np.ones((number_of_samples, 1))
    x = np.column_stack((x, fix_feature_b))
    return x

# calculate the sensitivity
def cal_sensitivity(x):
    vector = []
    for v in x:
        vector.append(np.linalg.norm(v))
    sensitivity = max(vector) - min(vector)
    sensitivity = sensitivity/max(vector)
    return sensitivity

# for semi-supervised model: to split unlabeled data and labeled data
def split_unlabeled_data(datapath,epsilon,noise_type):
    # get the original data
    data = pd.read_csv(datapath,header=None) # 440data/volcanoes/volcanoes.data
    data = data.iloc[:, 2:]
    data = data.sample(frac=1)
    x = data.iloc[:, :-1]
    # adjudge whether it needs to add input noise
    if noise_type == 2:
        # calculate sensitivity
        sensitivity = cal_sensitivity(x)
        noise = np.random.laplace(0,sensitivity/epsilon,x.shape)
        x = x + noise
    y = data.iloc[:,-1]
    # x = normalization1(x)
    num_feature = x.shape[1]
    data = np.column_stack((x,y))
    data = pd.DataFrame(data)
    # assume 20% data is labeled and 80% have to be assigned pseudolabel
    labeled_index = int(0.2 * len(data))
    original_labeled = data[0:labeled_index]
    original_unlabeled = data[labeled_index:len(data)]
    # create the dataset that suits torch (I define it in torch_dataset.py)
    labeled_dataset = torch_dataset.dataset(original_labeled)
    unlabeled_dataset = torch_dataset.dataset(original_unlabeled)

    return labeled_dataset, unlabeled_dataset, original_labeled, original_unlabeled, y[labeled_index:len(data)],num_feature

# for semi-supervised model: student's training and testing data
def get_student_data(data, batch_size):
    # get the original data
    data = data.sample(frac=1)
    # 80% data for training, 20% data for testing
    train_index = int(0.8 * len(data))
    train_dataset = data[0:train_index]
    test_dataset = data[train_index:len(data)]
    # create the dataset that suits torch (I define it in torch_dataset.py)
    train_dataset = torch_dataset.dataset(train_dataset)
    test_dataset = torch_dataset.dataset(test_dataset)
    # load data
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                                   batch_size=batch_size,
                                                   shuffle=True)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                                   batch_size=batch_size,
                                                   shuffle=False)

    return train_loader, test_loader
