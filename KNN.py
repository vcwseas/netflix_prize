import numpy as np
import scipy as sp
from scipy import sparse
from scipy import stats

def KNN(k, dataset, users):

    #   Dataset is a sparse row matrix with column corresponding to user, row corresponding to movie
    #   k is number of nearest neighbors we are interested in finding

    nearest_neighbors = np.zeros([dataset.shape[0], k])
    distances = np.zeros([dataset.shape[0], k])


    for user in users:
        user_movies = dataset[user]
        user_movies_indices = user_movies.indices
        neighbor_data = dataset[:, user_movies_indices]
        distance_i = np.zeros([dataset.shape[0]])
        print(user)

        j = 0
        for neighbor in range(0, dataset.shape[0]-1):
            neighbor_data_i = neighbor_data[neighbor]
            if neighbor_data_i.size > 5 and neighbor != user:
                print(j)
                other_user_movies = neighbor_data_i.toarray()
                watched_movies = ~(other_user_movies == 0) * 1
                user_movies_array = user_movies.toarray()[[0], user_movies_indices]
                user_movies_array = user_movies_array * watched_movies
                distance_i[j] = np.linalg.norm(user_movies_array - other_user_movies) / watched_movies.sum()
            j = j + 1

        distance_i[np.where(distance_i == 0)] = np.inf
        top_neighbors = distance_i.ravel().argsort()
        top_distances = distance_i[top_neighbors]
        nearest_neighbors[user, :] = top_neighbors[0:k]
        distances[user, :] = top_distances[0:k]

    return nearest_neighbors, distances

dataset = sp.sparse.load_npz('data_matrix.npz')
dataset_c = dataset.tocsc(copy=True)
dataset_n = dataset_c.transpose()

user_index = dataset_n.indices
freq = sp.stats.itemfreq(user_index)
users = user_index[freq[:, 1].argsort()]
users = users[0:100]


k = 100
nn, nn_distances = KNN(k, dataset_n, users)
np.savetxt("KNN_matrix", nn, fmt='%d')
np.savetxt("KNN_distances", nn_distances, fmt='%d')
