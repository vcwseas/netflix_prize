import numpy as np
import scipy as sp
from scipy import sparse
from scipy import stats


def KNN_imp(KNN_matrix, users, dataset_n):

    y = np.array([])
    y_hat = np.array([])

    for i in range(0,100):
        print(i)

        user_movies = dataset_n[users[i]].indices
        user_ratings = dataset_n[users[i]].data
        y = np.append(y,user_ratings)

        neighbors = dataset_n[:,user_movies]
        neighbors = neighbors[KNN_matrix[users[i]],]
        neighbors_sum = neighbors.sum(axis=0)
        watched_movies = ((neighbors > 0)*1).sum(axis=0)
        pred = neighbors_sum/watched_movies
        y_hat = np.append(y_hat, pred)

    y_hat[np.isnan(y_hat)] = 3

    return np.sqrt(np.sum(np.square(y_hat - y)/y.size))

def KNN_imp_cweight(KNN_matrix, users, dataset_n):

    y = np.array([])
    y_hat = np.array([])
    weight = np.flip(np.arange(0,1,0.01), axis=0)

    for i in range(0,100):
        print(i)

        user_movies = dataset_n[users[i]].indices
        user_ratings = dataset_n[users[i]].data
        y = np.append(y,user_ratings)

        neighbors = dataset_n[:,user_movies]
        neighbors = neighbors[KNN_matrix[users[i]],]
        neighbors = np.multiply(neighbors,weight)
        neighbors_sum = neighbors.sum(axis=0)
        watched_movies = ((neighbors > 0)*1).sum(axis=0)
        watched_movies = np.multiply(watched_movies, weight)
        pred = neighbors_sum/watched_movies
        y_hat = np.append(y_hat, pred)

    y_hat[np.isnan(y_hat)] = 3

    return np.sqrt(np.sum(np.square(y_hat - y)/y.size)) # RMSE 0.93753881512488402


def KNN_imp_cweight(KNN_matrix, users, dataset_n):

    y = np.array([])
    y_hat = np.array([])
    weight = np.flip(np.arange(0,1,0.01), axis=0)

    for i in range(0,100):
        print(i)

        user_movies = dataset_n[users[i]].indices
        user_ratings = dataset_n[users[i]].data
        y = np.append(y,user_ratings)

        neighbors = dataset_n[:,user_movies]
        neighbors = neighbors[KNN_matrix[users[i]],]
        neighbors = np.transpose(np.multiply(np.transpose(neighbors.toarray()),weight))
        neighbors_sum = neighbors.sum(axis=0)
        watched_movies = ((neighbors > 0) * 1)
        watched_movies = np.transpose(np.multiply(np.transpose(watched_movies), weight))
        watched_movies = watched_movies.sum(axis=0)
        pred = neighbors_sum/watched_movies
        y_hat = np.append(y_hat, pred)

    y_hat[np.isnan(y_hat)] = 3

    return np.sqrt(np.sum(np.square(y_hat - y)/y.size)) # RMSE 0.93023783109686342


def KNN_imp_nweight(KNN_matrix, users, dataset_n):

    y = np.array([])
    y_hat = np.array([])
    weight = np.flip(np.arange(0,1,0.01), axis=0)
    weight2 = np.arange(0, 1, 0.01)

    for i in range(0,100):
        print(i)

        user_movies = dataset_n[users[i]].indices
        user_ratings = dataset_n[users[i]].data
        y = np.append(y,user_ratings)

        neighbors = dataset_n[:,user_movies]
        neighbors = neighbors[KNN_matrix[users[i]],]
        neighbors = neighbors.toarray()
        neighbors2 = 5-neighbors
        neighbors = np.transpose(np.multiply(np.transpose(neighbors),weight))
        neighbors2 = np.transpose(np.multiply(np.transpose(neighbors2),weight2))
        neighbors = neighbors + neighbors2
        neighbors_sum = neighbors.sum(axis=0)
        watched_movies = ((neighbors > 0) * 1)
        watched_movies = watched_movies.sum(axis=0)
        pred = neighbors_sum/watched_movies
        y_hat = np.append(y_hat, pred)

    y_hat[np.isnan(y_hat)] = 3

    return np.sqrt(np.sum(np.square(y_hat - y)/y.size)) # RMSE 1.4163138462255374


def KNN_imp_bin(KNN_matrix, users, dataset_n):

    y = np.array([])
    y_hat = np.array([])

    for i in range(0,100):
        print(i)

        user_movies = dataset_n[users[i]].indices
        user_ratings = dataset_n[users[i]].data
        y = np.append(y,user_ratings)

        neighbors = dataset_n[:,user_movies]
        neighbors = neighbors[KNN_matrix[users[i]],]
        neighbors = neighbors.toarray()
        neighbors_t = (neighbors > 3)*2
        neighbors_f = np.logical_and(neighbors <= 3, neighbors > 0)*1
        neighbors = neighbors_t + neighbors_f
        neighbors_sum = neighbors.sum(axis=0)
        watched_movies = ((neighbors > 0) * 1)
        watched_movies = watched_movies.sum(axis=0)
        pred = neighbors_sum/watched_movies
        y_hat = np.append(y_hat, pred)

    y_hat = (y_hat > 1.5) * 1
    y_hat[np.isnan(y_hat)] = 0
    y = (y > 3) * 1

    return np.sum(y_hat==y)/y.size # Accuracy 0.75115504620184803
