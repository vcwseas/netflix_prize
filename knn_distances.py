import numpy as np
import scipy as sp
from scipy import sparse

'''
def find_test_scores(data_train, index_c_train, index_r_train, user_id, test_movies):

    # separate to just data for user
    [user_index, ] = np.where(index_r_train == user_id)
    user_movies = index_c_train[user_index]
    user_ratings = data_train[user_index]
    print('a')


    # Grab data corresponding to movies user watched - grabs only data points where movie was a movie user watched
    others_index_bool = np.isin(index_c_train, user_movies)  # This is messing up runtime
    other_movies = np.extract(others_index_bool, index_c_train)
    other_users = np.extract(others_index_bool, index_r_train)
    other_ratings = np.extract(others_index_bool, data_train)
    print('b')

    '''
'''
    # Find counts of movies watched by each user
    others, counts = np.unique(other_users, return_counts=True)

    # Separate data to just that of users who watched all movies of interested user
    others = others[np.where(counts == user_movies.size)]
    others_index_bool = np.in1d(other_users, others)
    other_movies2 = np.extract(others_index_bool, other_movies)
    other_users2 = np.extract(others_index_bool, other_users)
    other_ratings2 = np.extract(others_index_bool, other_ratings)
    '''
'''

    # Get counts to ensure only top neighbors are considered
    others, counts = np.unique(other_users, return_counts=True)
    others_ind = counts.ravel().argsort()[::-1]
    others = others[others_ind]
    print('c')

    # Initialize
    distances = np.zeros(25)
    test_scores = np.zeros((25, test_movies.size))
    sorted_ind = user_movies.ravel().argsort()
    user_ratings = user_ratings[sorted_ind]
    j = 0
    print('d')


    # Calculate distance measures
    for i in np.unique(others[1:25]):

        other_ratings_i = other_ratings[np.where(other_users == i)]
        other_movies_i = other_movies[np.where(other_users == i)]
        sorted_ind_i = other_movies_i.ravel().argsort()
        other_ratings_i_sorted = other_ratings_i[sorted_ind_i]

        sorted_ind = user_movies.ravel().argsort()
        user_ratings_i = user_ratings[sorted_ind]
        user_movies_i = user_movies[sorted_ind]

        user_ratings_i = user_ratings_i[np.isin(user_movies_i,other_movies_i)]

        distances[j] = np.linalg.norm(user_ratings_i - other_ratings_i_sorted)/user_ratings_i.size

        for k in range(0,np.size(test_movies)):
            index_single = np.where(index_c_train == test_movies[k])
            user_single = index_r_train[index_single]
            if i in user_single:
                rating_single = (data_train[index_single])
                rating_single = rating_single[np.where(user_single==i)]
                test_scores[j][k] = rating_single
            else:
                test_scores[j][k] = np.nan

        j = j + 1
    test_rating = np.zeros(np.size(test_movies))
    for l in range(0,np.size(test_movies)):

        index_comp = ~np.isnan(test_scores[:,l])
        test_rating[l] = np.vdot(distances[index_comp]/distances[index_comp].sum(), test_scores[index_comp,l])
    return test_scores, test_rating


# Main Function
training = sp.sparse.load_npz('data_matrix')


# Get the data from sparse crs matrix
data_train = training.data
index_c_train = training.indices
index_r_train = training.indptr

temp = index_r_train[1:] - index_r_train[0:-1]
temp2 = np.arange(0, np.size(temp))
index_r_train = np.repeat(temp2, temp)

# Seperate to train and test set
indices = np.random.randint(95862516, size=100)
index_r_test_set = index_r_train[indices]
index_r_train_set = np.delete(index_r_train, indices)

index_c_test_set = index_c_train[indices]
index_c_train_set = np.delete(index_c_train, indices)

data_test_set = data_train[indices]
data_train_set = np.delete(data_train, indices)

data_test_set_hat = np.zeros(np.size(data_test_set))
a = 1

# Test
for user in np.unique(index_r_test_set):
    print(a)
    a = a + 1
    user_ind = np.where(index_r_test_set == user)
    user_test_movies = index_c_test_set[user_ind]
    unused, test_ratings = find_test_scores(data_train_set, index_c_train_set, index_r_train_set, user_id=user, test_movies=user_test_movies)
    data_test_set_hat[user_ind] = test_ratings

data_test_set_hat[np.isnan(data_test_set_hat)] = 3
mse = np.mean((data_test_set - data_test_set_hat)**2)
print(mse)
print(data_test_set_hat)
'''




def find_test_scores(training, user, test_movies):

    user_movies = training[user]
    train_movies = user_movies.indices
    train_other_users = training[:, train_movies]
    print('a')

    test_movies = test_movies.indices
    test_other_users = training[:, test_movies]

    test_scores = test_other_users.toarray()
    distance = np.zeros(training.shape[0])
    print('b')

    test_users_watched = np.where(test_other_users.toarray().sum(axis=1) > 0)[0]

    for i in test_users_watched:
        if train_other_users[i].size > 100 and i != user:

            other_user_movies = train_other_users[i].toarray()
            watched_movies = ~(other_user_movies==0)*1
            user_movies_array = user_movies.toarray()[[0],train_movies]
            user_movies_array = user_movies_array*watched_movies
            distance[i] = np.linalg.norm(user_movies_array-other_user_movies)/watched_movies.sum()

    testing_rating = np.dot(np.transpose(distance), test_scores)/np.sum(distance)
    print('c')

    return testing_rating


# Main Function
training = sp.sparse.load_npz('data_matrix')

# Separate to train and test set
indices = np.random.randint(training.size, size=1000)
indices_not = np.delete(range(0,training.size),indices)

training_split = training.copy()
testing_split = training.copy()
testing_split.data[indices_not] = 0
training_split.data[indices] = 0
testing_split.eliminate_zeros()
training_split.eliminate_zeros()

testing_split_c = testing_split.tocsc(copy=True)

test_predictions = np.zeros(testing_split.data.size)
print('created data')

# Test
for user in np.unique(testing_split_c.indices):
    user_test_movies = testing_split[user]
    if user_test_movies.size > 0:
        test_rating = find_test_scores(training=training_split,  user=user, test_movies=user_test_movies)
        test_predictions[np.where(testing_split_c.indices == user)] = test_rating # Cheating a bit by assuming ordered users in data array

test_predictions[np.isnan(test_predictions)] = 3
mse = np.mean((testing_split.data - test_predictions)**2)
print(mse)

