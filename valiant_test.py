import numpy as np
from cvxpy import *
from scipy.misc import comb

def generate_dataset(p = 0.5, t = 2, size = 1000):
    '''
    Generates an array of binomial data with t trials and with prob success p.
    Size is the number of samples to draw. 
    '''
    noise = []
    num_noise = size
    factor = 3
    mean = 1/(factor * 2)
    for i in range(num_noise):
        noise.append(np.random.uniform()/factor - mean)

    dataset = []
    for i in range(len(noise)):
        dataset.append(np.random.binomial(n = t, p = p + noise[i], size = int(size / num_noise)))

    return np.reshape(np.array(dataset), (-1, ))

if __name__ == "__main__":

    #Set up for generating dataset
    np.random.seed(123)
    t = 2 #number of trials observed of each example
    m = 100 #size of the epsilon net
    p = 0.8 #underlying latent prob
    size = 100 #number of data points
    dataset = generate_dataset(t = t, p = p, size = size)
    n = dataset.shape[0]

    #Problem Setup
    beta_ks = np.zeros(t) #empirical estimates of moments
    for k in range(t):
        coef_ = 1./n * 1./comb(t, k+1)
        running_sum_ = 0
        for i in range(n):
            running_sum_ += comb(dataset[i], k+1)
        beta_ks[k] = (coef_ * running_sum_)
    beta_ks = Constant(beta_ks)

    x = Variable(m+1)
    beta_hat = [] #"true" moments that depend on the e-net variables
    for k in range(t):
        running_sum_ = 0
        for i in range(m+1):
            running_sum_ += x[i] * (i * 1.0 / m)**(k+1)
        beta_hat.append(running_sum_)
    beta_hat = bmat(beta_hat)

    #Quadratic Objective
    # objective = Minimize(sum_entries(power(beta_hat - beta_ks, 2)))
    #Linear Objective
    objective = Minimize(sum_entries(abs(beta_hat - beta_ks)))
    constraints = [x >= 0, sum_entries(x) == 1]
    prob = Problem(objective, constraints)
    result = prob.solve()

    soln = np.array(x.value)
    print(soln)
    print(np.argmax(soln)/m)




