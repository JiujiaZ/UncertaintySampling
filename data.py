import numpy as np

def data_generator(n=215000, d=200, rho=3, noise = None):
    # linear separable synthetic data when noise is not None

    mean = np.zeros(d)
    cov = (1 / np.arange(1, d + 1)).reshape(-1, 1)
    cov = np.eye(d) * cov
    data = np.random.multivariate_normal(mean, cov, size=n)
    diff = 1

    while diff >= 0.05:
        theta = np.random.normal(size=d).reshape(-1, 1)
        y_hat = np.matmul(data, theta)
        pred = np.sign(y_hat)
        # balanced data:
        diff = ((pred == 1).sum() - (pred == -1).sum()) / n

    # non-linear seperable with specified noise:
    if noise is not None:
        flip_num = int(np.ceil(noise * n))
        flip_indx = np.random.choice(n, flip_num, replace=False)
        pred[flip_indx] = pred[flip_indx] * -1

    label = pred.reshape(-1)

    # scale problem return separating hyperplane norm
    min_dist = np.absolute(y_hat).min()
    scale = rho / min_dist
    theta_star = theta * scale

    return data, label, np.linalg.norm(theta_star), min_dist


def train_test_split(data, label, ratio=None):
    n = len(label)
    if ratio is not None:
        train_indx = np.random.randint(n, size=int(np.ceil(ratio * n)), dtype=int)
    else:  # paper split
        train_indx = np.random.randint(n, size=200000, dtype=int)

    train_data = data[train_indx, :]
    train_label = label[train_indx]
    test_data = data[~train_indx, :]
    test_label = label[~train_indx]

    return train_data, train_label, test_data, test_label