from scipy import sparse
import numpy as np
from scipy.misc import comb
from cvxpy import *
from data_loader import load_partitions, valiant_preprocessing
from matplotlib import pyplot as plt



def estimate_parameters(data_matrix, csr_data, threshold = 100, by_movie = True):
    '''
    Return an array of latent parameters for 0/1 under
    descritized setting. 
    Arg:
    data_matrix numpy array after discretizing and summing over axis. 
    csr_data is the full discretized data. Needed for MLE to get # observations.
    '''
    #Reshape from [X, 1] to (X)
    data_matrix = np.reshape(data_matrix, data_matrix.shape[0])

    #Indicies threshold on data matrix to check which rows to use 
    #for which estimation technique.
    #To store number of observations per row (user/movie)
    non_zero_elements_array = np.zeros(csr_data.shape[0])
    for i in range(csr_data.shape[0]):
        non_zero_elements_array[i] = csr_data[i].indices.size

    valiant_indices = np.array(non_zero_elements_array < threshold)
    mle_indices = np.invert(valiant_indices)
    parameters = np.zeros_like(data_matrix, dtype = float)

    #For t <= threshold, use valiant.
    valiant_matrix = data_matrix[valiant_indices]
    parameters[valiant_indices] = valiant_estimation(valiant_matrix, threshold, plot = False)

    #For t > threshold, use MLE. 
    mle_matrix = data_matrix[mle_indices]
    parameters[mle_indices] = mle_estimation(mle_matrix, mle_indices, non_zero_elements_array)

    return parameters

def valiant_estimation(valiant_matrix, threshold, plot = False):
    '''
    Returns the latent parameter doing valiant estimation
    Returns the argmax probability in the CDF. 
    '''
    print("Valiant estimation for priors.")
    t = threshold #number of trials observed of each example
    m = 100 #size of the epsilon net
    n = valiant_matrix.shape[0] #number of data points
    print("Threshold: {0}, Size of Epsilon Net = {1}, Number of Points = {2}".format(t,m,n))

    #Problem Setup
    beta_ks = np.zeros(t) #empirical estimates of moments
    for k in range(t):
        coef_ = 1./n * 1./comb(t, k+1)
        running_sum_ = 0
        for i in range(n):
            running_sum_ += comb(valiant_matrix[i], k+1)
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
    objective = Minimize(sum_entries(power(beta_hat - beta_ks, 2)))
    #Linear Objective
    # objective = Minimize(sum_entries(abs(beta_hat - beta_ks)))
    constraints = [x >= 0, sum_entries(x) == 1]
    prob = Problem(objective, constraints)
    result = prob.solve()
    solution = np.array(x.value)

    ret_val = np.argmax(solution)/m

    #Plotting CDF logic
    if plot:
        running_sum = [0]
        for i in range(m):
            running_sum.append(running_sum[i] + solution[i])

        plt.figure()
        plt.plot(range(len(running_sum)), running_sum)
        plt.show()

    return ret_val


def mle_estimation(mle_matrix, mle_indices, non_zero_elements_array):
    '''
    MLE estimation given enough data per point.
    Arg:
        mle_indices informs which points to estimate on.
        non_zero_elements_array stores number observations for each row. 
        mle_matrix stores the sum of the bernoulli trials.
    '''
    print("Maximum Likelihood Estimation for Priors")

    #To store number of observations per row (user/movie)
    non_zero_elements_array = np.zeros(csr_data.shape[0])
    for i in range(csr_data.shape[0]):
        non_zero_elements_array[i] = csr_data[i].indices.size

    #need to recast [X, 1] to (X, )
    mle_indices = np.reshape(mle_indices, mle_indices.shape[0])

    #Need to recast int data to float for division
    mle_matrix = mle_matrix.astype(float)

    #Need to recast int data to float for division
    non_zero_elements_array = non_zero_elements_array.astype(float)

    #element-wise division of number of 1s by number of observations
    parameters = mle_matrix/non_zero_elements_array[mle_indices]

    return parameters



if __name__ == "__main__":
    a,_,_ = load_partitions()
    data_matrix, _, csr_data= valiant_preprocessing(a)
    print(estimate_parameters(data_matrix, csr_data))