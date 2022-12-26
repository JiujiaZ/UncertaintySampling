import numpy as np

class Sampler:
    def __init__(self, epsilon, mu = 0, lr = 0, R = 0, rho = 3, method = 'adaptive'):
        self.epsilon = epsilon
        self.iter_accum = 0
        self.mu = mu
        self.method = method
        self.R = R
        self.rho = rho
        if method == 'fixed':
            self.lr = np.minimum(1/self.mu, (self.rho-1)/(1+self.mu)) / (self.R**2 * np.maximum(1,1/self.mu))
        else:
            self.lr = lr


    def adaptive_mu(self, data, iter):
        current_val = np.absolute(np.dot(data, iter))
        mu_t = 1 + 1/(np.sqrt(self.epsilon) + self.iter_accum)
        sigma_t = 1/(1+ mu_t * current_val )
        self.iter_accum += current_val
        self.mu = mu_t

        lr = np.minimum(1/mu_t, (self.rho-1)/(1+mu_t)) / np.maximum(1,1/mu_t)
        self.lr = lr / self.R**2
        return sigma_t

    def fixed_mu(self, data, iter):
        current_val = np.absolute(np.dot(data, iter))
        sigma_t = 1/(1+ self.mu * current_val )
        return sigma_t

    def forward(self, data, iter):
        if self.method == 'adaptive':
            sigma_t = self.adaptive_mu(data, iter)
        elif self.method == 'fixed':
            sigma_t = self.fixed_mu(data, iter)
        else:
            sigma_t = 1
            t = 1 / (self.lr **2)
            self.lr = 1 / np.sqrt(t + 1)

        # sample based on sigma_t
        z_t = np.random.binomial(1, sigma_t)

        return z_t, sigma_t
