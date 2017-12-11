import numpy as np
import scipy as sp
from scipy import sparse
from scipy import stats

def RMSE_vector(y_hat, y):

    # return vector RMSE

    return np.sqrt(np.sum(np.square(y_hat - y)/y.size))



def RMSE_sparse_mat(y_hat_sparse, y_sparse):

    # return sparse RMSE

    y_hat = y_hat_sparse.data
    y = y_sparse.data

    return np.sqrt(np.sum(np.square(y_hat - y) / y.size))