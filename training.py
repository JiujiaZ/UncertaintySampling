from tqdm import tqdm
import numpy as np

def train(sampler, data, label, test_data, test_label, theta):
    acc_list = list()
    z_list = list()
    sigma_list = list()
    full_acc_list = list()

    # initial accuracy
    acc = eval(theta, test_data, test_label)

    if sampler.method == 'perceptron':
        for x_t, y_t in tqdm(zip(data, label), total = len(label)):
        # for x_t, y_t in zip(data, label):
            if y_t * np.dot(x_t, theta) <=0 :
                theta  +=  y_t * x_t
                acc = eval(theta, test_data, test_label)
                acc_list.append(acc)
            full_acc_list.append(acc)
    else:
        for x_t, y_t in tqdm(zip(data, label), total=len(label)):
        # for x_t, y_t in zip(data, label):
            z_t, sigma_t = sampler.forward(x_t, theta)
            z_list.append(z_t)
            sigma_list.append(sigma_t)
            # print('sigma_t: ', sigma_t)
            if z_t == 1:
                # print('z_t: ', z_t)
                theta += sampler.lr * y_t * x_t * np.maximum(0, (1 - y_t * np.dot(theta, x_t)))
                acc = eval(theta, test_data, test_label)
                # print('acc: ', acc)
                acc_list.append(acc)
            full_acc_list.append(acc)

    return acc_list, z_list, sigma_list, full_acc_list


def eval(theta, data, label):
    pred = np.sign(np.matmul(data, theta))
    acc = (pred == label).sum() / len(label)
    return acc
