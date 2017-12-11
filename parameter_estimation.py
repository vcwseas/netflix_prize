from scipy import sparse
import numpy as np
from scipy.misc import comb
from cvxpy import *
from data_loader import load_partitions, valiant_preprocessing



def estimate_parameters(data_matrix, csr_data, threshold = 10, by_movie = True):
    '''
    Return an array of latent parameters for 0/1 under
    descritized setting. 
    Assume data_matrix numpy array after discretizing and summing over axis. 
    Assume csr_data is the full discretized data.
    '''
    valiant_indices = np.array(data_matrix < threshold)
    mle_indices = np.invert(valiant_indices)
    parameters = np.zeros_like(data_matrix)

    #For t <= threshold, use valiant.
    # valiant_matrix = data_matrix[valiant_indices]
    # parameters[valiant_indices] = valiant_estimation(valiant_matrix, threshold)

    #For t > threshold, use MLE. 
    mle_matrix = data_matrix[mle_indices]
    parameters[mle_indices] = mle_estimation(mle_matrix, mle_indices, csr_data, by_movie)

    return parameters

def valiant_estimation(valiant_matrix, threshold):
    '''
    Returns the latent parameter doing valiant estimation
    Returns the argmax probability in the CDF. 
    '''
    print("Running valiant estimation.")
    t = threshold #number of trials observed of each example
    m = 100 #size of the epsilon net
    n = valiant_matrix.shape[0] #number of data points
    print("Threshold: {0}, m = {1}, n = {2}".format(t,m,n))

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
    # print (ret_val)

    return ret_val


def mle_estimation(mle_matrix, mle_indices, csr_data, by_movie):
    '''
    MLE estimation given enough data per point.
    If by_movie is true, then csr_data is already in correct format.
    Otherwise, if by_movie if false, then by_user, then csr_data needs to be transposed.
    MLE indices informs which points to estimate on. 
    '''
    non_zero_elements_array = np.zeros(mle_indices.shape[0])
    if not by_movie:
        csr_data = csr_data.transpose()

    for i in range(csr_data.shape[0]):
        non_zero_elements_array[i] = csr_data[i].indices.size
        print(non_zero_elements_array[i])

    print(mle_matrix.shape)
    print(non_zero_elements_array[mle_indices].shape)

    parameters = mle_matrix/non_zero_elements_array[mle_indices]
    return parameters



if __name__ == "__main__":
    a,_,_ = load_partitions()
    data_matrix, _, csr_data= valiant_preprocessing(a)
    print(estimate_parameters(data_matrix, csr_data))