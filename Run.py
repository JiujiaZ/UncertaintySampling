import random
from data import *
from sampling import Sampler
from training import train

random.seed(10)

# global argument:
methods = ['adaptive', 'fixed', 'fixed', 'fixed', 'fixed', 'sgd', 'perceptron']
save_names = ['acc', 'z', 'sigma', 'full_acc']
rho = 3
epsilon = np.sqrt(rho-1)
mu = [0, 1, 2, 4, 10, 0, 0]
lr = 1


# generate data:
noise = 0.1
ratio = None
save_path = '/projectnb/aclab/jiujiaz/UncertaintySampling/Noise/'
save_path = '/Users/jiujiazhang/PycharmProjects/UncertaintySampling/Noise/'
data, label, theta_star_norm, min_dist = data_generator(n = 215000, d = 200, noise = noise)
train_data, train_label, test_data, test_label = train_test_split(data, label, ratio = ratio)
R = np.ceil(np.amax(np.linalg.norm(data, axis = 1)))

for i, method in enumerate(methods):

    print('method: ', method )
    sampler = Sampler(epsilon, mu = mu[i], lr = lr, R = R, rho = rho, method = method)
    print(sampler.method)
    theta = np.zeros(data.shape[1])
    save_infos = train(sampler, train_data, train_label, test_data, test_label, theta)

    for name, info in zip(save_names, save_infos):
        save_name = method + '_' + str(mu[i]) + '_' + name + '.npy'
        np.save(save_path+save_name, info)
