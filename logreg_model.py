import numpy as np
import util
from numba import jit
from scipy.optimize import fmin_bfgs

# input gaussian noise
@jit(nopython=True, fastmath = True)
def gaussian_input(x:np.ndarray, sensitivity, epsilon):
    std = sensitivity/epsilon
    gauss_array = np.random.normal(0.0, std,x.size)
    x += gauss_array
    return x

# input laplace noise
@jit(nopython=True, fastmath = True)
def laplace_val(sensitivity, epsilon):
    beta = sensitivity/epsilon
    random1 = np.random.random()
    random2 = np.random.random()
    if random1 <= 0.5:
        n_val = -beta * np.log(1-random2)
    else:
        n_val = beta * np.log(random2)
    return n_val

# input laplace noise by numpy laplace
def laplace_input(x:np.ndarray, sensitivity, epsilon):
    laplace_array = np.ones(x.size)
    for i in range(x.size):
        laplace_array[i] *= np.random.laplace(0,sensitivity/epsilon)
    laplace_array = laplace_array.reshape(len(x),len(x[0]))
    x += laplace_array
    return x

# add laplace noise to gradient
def gradient_noise(gradient, group_size, sigma, norm_bound):
    # clip gradient
    gradient = gradient/(max(1, (np.linalg.norm(gradient))**2/norm_bound))
    # add noise: sigma and norm bound decide the scale of noise
    gradient = 1/group_size * (gradient + np.random.laplace(0,sigma * norm_bound))
    return gradient

# calculate b when adding objective perturbing
def cal_b(epsilon, dimension):
    alpha = 1 # alpha is a normalizing constant
    d = dimension # dimension
    temp_b = np.random.gamma(d, 2/epsilon)
    b = (1/alpha) * np.random.exponential(1/(np.linalg.norm(temp_b)*epsilon*0.5),d)
    return b

# get b and delta when adding objective perturbing
def objective_perturbation(epsilon, noise_lamda, c, n):
    # c = 0.25 as paper mentioned
    epsilon_prime = epsilon - np.log(1+(2*c/n*noise_lamda)+(c**2/n**2 * noise_lamda**2))
    if epsilon_prime > 0:
        delta = 0
    else:
        delta = c/(n*(np.exp(epsilon/4)-1)) - noise_lamda
    b = cal_b(epsilon_prime, n)
    return b, delta

class Logreg():

    def __init__(self, num_features,lamda):
        self.num_features = num_features
        self._lamda = lamda
        self.weight_bias = np.zeros((num_features))

    # loss function
    def loss_func(self,w, x:np.ndarray, y:np.ndarray, lamda, b):
        n = self.dimension # number of features
        dz = -1*np.dot(np.transpose(w),np.transpose(x))*y
        dz = np.log(1+np.exp(dz))
        loss = -y*np.log(dz) - (1-y)*np.log(1-dz)
        return np.mean(loss) + (0.5*lamda * (np.linalg.norm(w) ** 2)) + (1/n)*np.dot(np.transpose(b),w)

    # gradient decent updates weight and bias (SGD)
    def gradient_decent_SGD(self, x, y):
        y_hat = util.sigmoid(np.multiply(self.weight_bias, x.T)) # GD uses dot, SGD uses multiply
        dz = y_hat - y
        d_weight_bias = ((np.multiply(dz, x)) - self._lamda * self.weight_bias) / len(x)
        return d_weight_bias

    # gradient decent updates weight and bias (GD)
    def gradient_decent_GD(self, x, y):
        y_hat = util.sigmoid(np.dot(self.weight_bias, x.T)) # GD uses dot, SGD uses multiply
        dz = y_hat - y
        d_weight_bias = ((np.dot(dz, x)) - self._lamda * self.weight_bias) / len(x)
        return d_weight_bias

    # fit function (when using SGD perturbing)
    def fit_SGD_noise(self, x: np.ndarray, y: np.ndarray, delta, epsilon, norm_bound, learning_rate):
        num_iterations = 50 # number of iteration
        sigma = np.sqrt(2*np.log(1.25/delta))/epsilon # decides the scale of noise

        for i in range(num_iterations):
            for j in range(len(x)):
                d_weight_bias = self.gradient_decent_SGD(x[j,:], y[j])
                # clip gradient
                gradient = d_weight_bias/(max(1, np.linalg.norm(d_weight_bias)/norm_bound))
                # add noise
                gradient = gradient + np.random.normal(0,sigma * norm_bound,size=gradient.shape) # mu, scale
                # gradient descent
                self.weight_bias = self.weight_bias - learning_rate * gradient

    # fit function (when using input and output perturbing)
    def fit_input_output_noise(self, x: np.ndarray, y: np.ndarray, learning_rate):
        num_iterations = 50
        for i in range(num_iterations):

            # this is for SGD with noise
            for j in range(len(x)):
                # this is for GD
                d_weight_bias = self.gradient_decent_GD(x, y)
                self.weight_bias = self.weight_bias - learning_rate * d_weight_bias

    # fit function (when using objective function perturbing)
    def fit_objective_noise(self, x:np.ndarray, y:np.ndarray, epsilon, noise_lamda):
        self.dimension = len(x[0])
        init_weight = np.random.randn(self.dimension)

        # setting c = 0.25 (as paper mentioned)
        b, delta = objective_perturbation(epsilon, noise_lamda, 0.25, self.dimension)
        d_weight = fmin_bfgs(self.loss_func,init_weight,args=(x,y,noise_lamda,b),disp=False)
        self.weight_bias = d_weight + (delta*np.linalg.norm(d_weight) ** 2)/2

    # predict function
    def predict(self, x: np.ndarray, threshold: float = 0.5):
        pred: list = list()
        result: np.ndarray = util.sigmoid(np.dot(self.weight_bias, x.T))

        for i in range(len(x)):
            if result[i] > threshold:
                pred.append(1)
            else:
                pred.append(0)
        return np.array(pred), result[0]

    # predict function
    def predict_IO(self, x: np.ndarray, noise_type, sensitivity, epsilon, threshold: float = 0.5, ):
        pred: list = list()
        result: np.ndarray = util.sigmoid(np.dot(self.weight_bias, x.T))
        if noise_type == 2:
            result += np.random.laplace(0,sensitivity/epsilon)

        for i in range(len(x)):
            if result[i] > threshold:
                pred.append(1)
            else:
                pred.append(0)
        return np.array(pred), result[0]

    # evaluate function
    def evaluation(self, x: np.ndarray, y: np.ndarray):
        y_hat, confidences = self.predict(x)
        return util.accuracy(y, y_hat)

    # evaluate function
    def evaluation_IO(self, x: np.ndarray, y: np.ndarray,noise_type, sensitivity, epsilon):
        y_hat, confidences = self.predict_IO(x,noise_type, sensitivity, epsilon)
        return util.accuracy(y, y_hat)
