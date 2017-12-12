from scipy import sparse
import numpy as np
from data_loader import load_dataset, valiant_preprocessing
from parameter_estimation import load_priors
from matplotlib import pyplot as plt
from sklearn.metrics import *

def load_knn():
    '''
    Load nearest neighbours (equivalence classes) from data. 
        "KNN_matrix"
        "KNN_distances"
    Legacy problems: 
        Full numpy matrix with mostly zero entries. 
        Need to return only rows with non-zero entries.
        Matrices currently in text form.
    '''
    def cleanup_(matrix):
        #Scan across columns to check which rows have non-zero entries
        non_zero_row_indicies = np.all(matrix, axis = 1)
        return matrix[non_zero_row_indicies]

    knn_matrix = np.loadtxt("KNN_matrix")
    knn_distances = np.loadtxt("KNN_distances")

    #Legacy cleanup
    knn_matrix = cleanup_(knn_matrix)
    knn_distances = cleanup_(knn_distances)

    return knn_matrix, knn_distances

def predict(csr_data, movie_priors, user_priors, sample_size = 1000):
    '''
    Sample a sample_size number of datapoints from the discretized dataset
    Run classifier based on priors. 
    Return a vector of predictions (0,1) and the true labels and the scores
    '''
    l = []
    y = np.zeros(sample_size)
    n = csr_data.shape[0]
    y_hat = np.zeros(sample_size)
    scores = np.zeros(sample_size)

    for i in range(sample_size):
        row = np.random.randint(n)
        column = np.random.choice(csr_data[row].indices)
        l.append((row, column))
        y[i] = csr_data[row, column]

    for i in range(sample_size):
        y_hat[i], scores[i] = predict_(l[i][0], l[i][1], movie_priors, user_priors)

    return y_hat, y, scores

def predict_(movie, user, movie_priors, user_priors):
    '''
    Classifier 
    Where M and U are bernoulli priors. 
    Where R is the bernoulli rating. 
    Probability estimates from MLE over equivalence class.

    Arg:
        movie: index of movie in csr_data
        user: index of user in csr_data

    Return MAP prediction for movie rating for user. 
    '''
    #P(M = 1) and P(U = 1)
    m_prior = movie_priors[movie]
    u_prior = user_priors[user]

    #00, 01, 10, 11
    theta = np.zeros(4, dtype = float)
    theta[0] = (1. - m_prior) * (1. - u_prior)
    theta[1] = m_prior * (1. - u_prior)
    theta[2] = (1. - m_prior) * u_prior
    theta[3] = m_prior * u_prior

    soln = np.argmax(theta)
    retval = 0
    score = 0

    #MAP logic
    if soln == 3:
        retval = 1
    elif soln == 0:
        retval = 0
    else:
        if soln == 1:
            if m_prior > (1. - u_prior):
                retval = 1
            else:
                retval = 0
        else:
            if 1. - m_prior > u_prior:
                retval = 0
            else:
                retval = 1
    return retval, theta[soln]


def calculate_metrics(y_true, y_pred):
    '''
    Returns accurarcy, precision, recall and f1. 
    '''
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    return acc, prec, recall, f1

def plot_roc_curves(matrix, m_thresholds, u_thresholds, title = "Receiver-Operating Characteristic"):
    return


if __name__ == "__main__":
    np.random.seed(123)
    data = load_dataset()
    data_matrix_movie, data_matrix_user, discretized_csr_data = valiant_preprocessing(data)
    user_thresholds = [5, 10, 15, 20, 25]
    movie_thresholds = [60, 70, 80, 90, 100]
    for i in range(len(user_thresholds)):
        for j in range(len(movie_thresholds)):
            movie_priors, user_priors = load_priors("movie_priors"+str(movie_thresholds[j]), "user_priors"+str(user_thresholds[i]))
            y_pred, y_true, y_score = predict(discretized_csr_data, movie_priors, user_priors)
            print("Movie Threshold: {0}, User Threshold: {1}".format(movie_thresholds[j], user_thresholds[i]))
            print(confusion_matrix(y_true, y_pred))
            acc, prec, recall, f1 = calculate_metrics(y_true, y_pred)
            print("Accuracy: {0}, Precision: {1}, Recall: {2}, F1: {3}".format(acc, prec, recall, f1))


