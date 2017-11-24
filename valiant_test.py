import numpy as np
from cvxpy import *
from scipy.misc import comb

def generate_dataset(p = 0.4, t = 2, size = 1000):
    '''
    Generates an array of data where p's are close to 0.5.
    '''
    return np.random.binomial(n = t, p = p, size = size)


if __name__ == "__main__":

    #Set up for generating dataset
    np.random.seed(123)
    t = 3 #number of trials observe of each example
    m = 100 #size of the epsilon net
    p = 0.324 #underlying latent prob
    size = 1000 #number of data points
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




