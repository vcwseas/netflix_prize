from scipy import sparse
import numpy as np
from scipy.misc import comb
from cvxpy import *
from data_loader import load_dataset, valiant_preprocessing
from matplotlib import pyplot as plt



def estimate_parameters(data_matrix, csr_data, threshold = 10):
    '''
    Return an array of latent parameters for 0/1 under
    descritized setting. 
    Arg:
        data_matrix numpy array after discretizing and summing over axis. 
        csr_data is the full discretized data. Needed for MLE to get # observations.

    Return:
        parameters estimated
        solution to the valiant estimation problem 
    '''
    #Reshape from [X, 1] to (X)
    data_matrix = np.reshape(data_matrix, data_matrix.shape[0])

    #Indicies threshold on data matrix to check which rows to use 
    #for which estimation technique.
    #To store number of observations per row (user/movie)
    non_zero_elements_array = csr_data.getnnz(axis = 1)

    valiant_indices = np.array(non_zero_elements_array < threshold)
    mle_indices = np.invert(valiant_indices)
    parameters = np.zeros_like(data_matrix, dtype = float)

    #For t <= threshold, use valiant.
    valiant_matrix = data_matrix[valiant_indices]
    parameters[valiant_indices], solution = valiant_estimation(valiant_matrix, threshold, plot = False)

    #For t > threshold, use MLE. 
    mle_matrix = data_matrix[mle_indices]
    parameters[mle_indices] = mle_estimation(mle_matrix, mle_indices, non_zero_elements_array)

    return parameters, solution

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
        plt.plot(np.linspace(0, 1, m), running_sum[1:])
        plt.xlabel("Support")
        plt.xlabel("Cumulative Distribution Function")
        plt.title("Recovered CDF")
        plt.show()

    return ret_val, solution


def mle_estimation(mle_matrix, mle_indices, non_zero_elements_array):
    '''
    MLE estimation given enough data per point.
    Arg:
        mle_indices informs which points to estimate on.
        non_zero_elements_array stores number observations for each row. 
        mle_matrix stores the sum of the bernoulli trials.
    '''
    print("Maximum Likelihood Estimation for Priors")

    #need to recast [X, 1] to (X, )
    mle_indices = np.reshape(mle_indices, mle_indices.shape[0])

    #Need to recast int data to float for division
    mle_matrix = mle_matrix.astype(float)

    #Need to recast int data to float for division
    non_zero_elements_array = non_zero_elements_array.astype(float)

    #element-wise division of number of 1s by number of observations
    parameters = mle_matrix/non_zero_elements_array[mle_indices]

    return parameters

def load_priors(mp_filename = "movie_priors", up_filename = "user_priors"):
    '''
    Load prior data from .npz files.
    '''
    movie_priors = np.load(mp_filename)
    user_priors = np.load(up_filename)
    return movie_priors, user_priors

def create_priors(m_threshold = 50, u_threshold = 10):
    '''
    Create priors for full dataset. 
    '''
    data = load_dataset()
    data_matrix_movie, data_matrix_user, discretized_csr_data = valiant_preprocessing(data)
    movie_priors, movie_solution = estimate_parameters(data_matrix_movie, discretized_csr_data, threshold = m_threshold)
    with open("movie_priors" + str(m_threshold), "wb") as file:
        np.save(file, movie_priors)
    user_priors, user_solution = estimate_parameters(data_matrix_user, discretized_csr_data.transpose(), threshold = u_threshold)
    with open("user_priors" + str(u_threshold), "wb") as file:
        np.save(file, user_priors)
    return movie_solution, user_solution


def plot_cdfs(matrix, thresholds, title = "Recovered CDFs", filename = "CDF.svg"):
    '''
    Plotting logic for multiple thresholds
    '''
    plt.figure()
    for i in range(len(matrix)):
        m = matrix[i].shape[0]
        running_sum = [0]
        for j in range(m):
            running_sum.append(running_sum[j] + matrix[i][j])
        plt.plot(np.linspace(0, 1, m), running_sum[1:], label = "Threshold = {0}".format(thresholds[i]))
    plt.xlabel("Support")
    plt.xlabel("Cumulative Distribution Function")
    plt.title(title)
    plt.legend()
    plt.savefig(filename)

if __name__ == "__main__":

    #plotting logic
    mt = [60, 70, 80, 90, 100]
    ut = [5, 10, 15, 20, 25]
    ml = []
    ul = []
    for i in range(5):
        movie_solution, user_solution = create_priors(mt[i], ut[i])
        ml.append(movie_solution)
        ul.append(user_solution)
    plot_cdfs(ml, mt, title = "Recovered CDFs for Movies' Latent Priors", filename = "CDF_movies.svg")
    plot_cdfs(ul, ut, title = "Recovered CDFs for Users' Latent Priors", filename = "CDF_users.svg")
