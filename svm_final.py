from random import randint

import scipy.io as scio
import numpy as np
from sklearn import svm


def get_kernel(x_1, x_2):
    return np.dot(x_1.T, x_2)


def get_function(alpha, current_x, x_train, y_train, b_threshold):
    final_sum = 0
    for i in range(np.shape(x_train)[0]):
        final_sum += ((alpha[i, 0] * y_train[i, 0]) * get_kernel(x_train[i, :], current_x)) + b_threshold
    return final_sum


def get_random_not_i(i, max_val):
    random_val = randint(0, max_val - 1)
    while True:
        if random_val != i:
            break
        random_val = randint(0, max_val - 1)
    return random_val


def compute_and_clip_alpha_j(old_alpha_j, current_y_j, e_i, e_j, nu_threshold, current_L, current_H):
    new_alpha_j = old_alpha_j - ((current_y_j[0] * (e_i - e_j)) / nu_threshold)
    if new_alpha_j > current_H:
        return current_H
    elif current_L <= new_alpha_j <= current_H:
        return new_alpha_j
    elif new_alpha_j < current_L:
        return current_L
    else:
        return current_L


def compute_new_alpha_i(old_alpha_i, current_y_i, current_y_j, old_alpha_j, new_alpha_j):
    new_alpha_i = old_alpha_i + (np.dot(current_y_i, current_y_j) * (old_alpha_j - new_alpha_j))
    return new_alpha_i


def compute_new_b(b_1, b_2, new_alpha_i, new_alpha_j, c_regularization):
    if 0 < new_alpha_i < c_regularization:
        return b_1
    elif 0 < new_alpha_j < c_regularization:
        return b_2
    else:
        return (b_1 + b_2) / 2


def smo(c_regularization, tolerance, max_passes, x_train, y_train):
    alpha = np.zeros(np.shape(y_train))
    b_threshold = 0
    passes = 0
    while passes < max_passes:
        old_alpha_i = 0
        old_alpha_j = 0
        num_changed_alphas = 0
        for i in range(np.shape(x_train)[0]):
            current_x_i = x_train[i, :]
            current_y_i = y_train[i, :]
            e_i = np.dot(np.multiply(alpha, y_train).T, np.dot(x_train, current_x_i.T)) + b_threshold - current_y_i
            if (current_y_i * e_i < -tolerance and alpha[i, 0] < c_regularization) or (current_y_i * e_i > tolerance and alpha[i, 0] > 0):
                j = get_random_not_i(i, np.shape(x_train)[0])
                current_x_j = x_train[j, :]
                current_y_j = y_train[j, :]
                e_j = np.dot(np.multiply(alpha, y_train).T, np.dot(x_train, current_x_j.T)) + b_threshold - current_y_j
                old_alpha_i = alpha[i, 0]
                old_alpha_j = alpha[j, 0]
                current_L = 0
                current_H = 0
                if current_y_i[0] != current_y_j[0]:
                    current_L = max(0, old_alpha_j - old_alpha_i)
                    current_H = min(c_regularization, (c_regularization + (old_alpha_j - old_alpha_i)))
                else:
                    current_L = max(0, (old_alpha_i + old_alpha_j - c_regularization))
                    current_H = min(c_regularization, (old_alpha_i + old_alpha_j))
                if current_L == current_H:
                    continue
                nu_threshold = 2 * get_kernel(current_x_i, current_x_j) - get_kernel(current_x_i, current_x_i) - get_kernel(current_x_j, current_x_j)
                if nu_threshold >= 0:
                    continue
                new_alpha_j = compute_and_clip_alpha_j(old_alpha_j, current_y_j, e_i, e_j, nu_threshold, current_L, current_H)
                alpha[j, 0] = new_alpha_j
                if abs(new_alpha_j - old_alpha_j) < 0.00001:
                    continue
                new_alpha_i = compute_new_alpha_i(old_alpha_i, current_y_i, current_y_j, old_alpha_j, new_alpha_j)
                alpha[i, 0] = new_alpha_i
                b_1 = b_threshold - e_i - ((current_y_i[0] * (new_alpha_i - old_alpha_i)) * get_kernel(current_x_i, current_x_i)) - ((current_y_j[0] * (new_alpha_j * old_alpha_j)) * get_kernel(current_x_i, current_x_j))
                b_2 = b_threshold - e_j - ((current_y_i[0] * (new_alpha_i - old_alpha_i)) * get_kernel(current_x_i, current_x_j)) - ((current_y_j[0] * (new_alpha_j * old_alpha_j)) * get_kernel(current_x_j, current_x_j))
                b_threshold = compute_new_b(b_1, b_2, new_alpha_i, new_alpha_j, c_regularization)
                num_changed_alphas += 1
        if num_changed_alphas == 0:
            passes += 1
        else:
            passes = 0
    return alpha, b_threshold

file_reader = scio.loadmat("data.mat")
x_train = file_reader['X_trn']
y_train = file_reader['Y_trn']
x_test = file_reader['X_tst']
y_test = file_reader['Y_tst']
c_regularization = [1.0, 10.0, 20.0]
max_passes = 1000
tolerance = 0.0001

print(x_train.shape)
print(y_train.T[0])


def svm_smo(x_train, y_train, c_regularization, tolerance, max_passes):
    alpha_list = list()
    b_list = list()

    for i in range(3):
        new_y = y_train
        new_y = new_y.astype(np.int16)
        new_y[new_y != i] = -1
        new_y[new_y == i] = 1
        current_alpha, current_bias = smo(c_regularization, tolerance, max_passes, x_train, new_y)
        alpha_list.append(current_alpha)
        b_list.append(current_bias)
    return alpha_list, b_list


def runSVM(current_alpha, bias, x_train, y_train, x_test):
    val = 0
    i=0
    x_test = x_test.T
    for point in x_train:
        p_1 = current_alpha[i] * y_train[i]
        p_2 = x_train[i] * x_test
        out = p_1 * p_2
        val += out
        i +=1
    return val + bias


def svm_predict(alpha_list, b_list, x_train, y_train, x_test, y_test):
    acc=0
    i=0
    for val in y_test:
        predList = list()
        for label in range(3):
            current_prediction = runSVM(alpha_list[label], b_list[label],x_train, y_train, np.matrix(x_test[i]))
            predList.append(current_prediction)
        current_prediction = predList.index(max(predList))
        print('Predicted Label [' + str(current_prediction) + ']  --> Actual Label' + str(val))
        if current_prediction == val:
            acc+=1
        i+=1
    return acc / float(len(y_test))


def svm_custom(x_train, y_train, x_test, y_test, c_regularization, max_passes, tolerance):
    for each_c in c_regularization:
        current_alpha_list, current_b_list = svm_smo(x_train, y_train, each_c, tolerance, max_passes)
        print('******************************************************************')
        print("For the regularization parameter c : " + str(each_c) + ", The accuracy is : "  + str(svm_predict(current_alpha_list, current_b_list, x_train, y_train, x_test, y_test)))
        print('******************************************************************')

def get_accuracy(kernel_svm_predict, y_test):
    final_accuracy = 0
    for i in range(kernel_svm_predict.shape[0]):
        if kernel_svm_predict[i] == y_test[i]:
            final_accuracy += 1
    return final_accuracy / float(kernel_svm_predict.shape[0])


# svm_custom(x_train, y_train, x_test, y_test, c_regularization, max_passes, tolerance)
# print('******************************************************************')
linear_kernel_svm = svm.SVC(kernel='linear')
linear_kernel_svm.fit(x_train, y_train.T[0])
linear_kernel_svm_predict = linear_kernel_svm.predict(x_test)
print('Scikit Learn Package Linear Kernel Output : ' + str(linear_kernel_svm_predict))
print('Scikit Learn Package Actual Given  Output : ' + str(y_test.T))
print('Scikit Learn Package Linear Kernel Accuracy : ' + str(get_accuracy(linear_kernel_svm_predict, y_test)))
# print('******************************************************************')
# polynomial_kernel_svm = svm.SVC(kernel='poly')
# polynomial_kernel_svm.fit(x_train, y_train.T[0])
# polynomial_kernel_svm_predict = polynomial_kernel_svm.predict(x_test)
# print('Scikit Learn Package Polynomial Kernel Output : ' + str(polynomial_kernel_svm_predict))
# print('Scikit Learn Package Actual Given  Output : ' + str(y_test.T))
# print('Scikit Learn Package Polynomial Kernel Accuracy : ' + str(get_accuracy(polynomial_kernel_svm_predict, y_test)))
# print('******************************************************************')
# sigmoid_kernel_svm = svm.SVC(kernel='sigmoid')
# sigmoid_kernel_svm.fit(x_train, y_train.T[0])
# sigmoid_kernel_svm_predict = sigmoid_kernel_svm.predict(x_test)
# print('Scikit Learn Package Sigmoid Kernel Output : ' + str(sigmoid_kernel_svm_predict))
# print('Scikit Learn Package Actual Given  Output : ' + str(y_test.T))
# print('Scikit Learn Package Sigmoid Kernel Accuracy : ' + str(get_accuracy(sigmoid_kernel_svm_predict, y_test)))
# print('******************************************************************')