
import numpy as np
import csv
import os
from scipy import sparse
import pickle

def load_ratings():
    cwd = os.getcwd()
    dir_path = os.path.join(cwd, "training_set")
    data_matrix = sparse.lil_matrix((17770, 480189), dtype =np.int8)
    userid_indicies_mapping = {}
    file_count = 0
    index = 0
    for file in os.listdir(dir_path):
        if file.endswith(".txt"): #sanity check
            file_count += 1
            filepath = os.path.join(dir_path, file)
            print("Processed %d/17770" % file_count)
            with open(filepath) as csvfile:
                reader = csv.reader(csvfile, delimiter = ",")
                for row in reader:
                    if len(row) == 1: #indicates movie id
                        movieid = np.int32(row[0][:-1].strip()) - 1
                    else:
                        userid = np.int32(row[0].strip())
                        rating = np.int8(row[1].strip())
                        if userid in userid_indicies_mapping:
                            data_matrix[movieid, userid_indicies_mapping[userid]] = rating
                        else:
                            userid_indicies_mapping[userid] = index
                            index += 1
                            data_matrix[movieid, userid_indicies_mapping[userid]] = rating

    if file_count == 17770:
        print("Correct count of files processed!")
    else:
        print("ERROR: Incorrect count of files processed!")

    data_matrix = data_matrix.tocsr(copy = False)
    return  data_matrix, userid_indicies_mapping

def create_partition():

    #Load data
    data_matrix = sparse.load_npz("data_matrix.npz")

    #Shuffle data
    np.random.seed(123)
    indicies = np.arange(data_matrix.shape[0])
    np.random.shuffle(indicies)
    data_matrix = data_matrix[indicies]

    #Indicies of training, dev and test sets
    train_index = int(indicies.shape[0] * 0.9)
    dev_index = int(indicies.shape[0] * 0.95)
    training_indicies = indicies[:train_index]
    dev_indicies = indicies[train_index:dev_index]
    test_indicies = indicies[dev_index:]

    #Create data partitions
    train_partiton = data_matrix[training_indicies]
    dev_partition = data_matrix[dev_indicies]
    test_partition = data_matrix[test_indicies]

    #We require the indicies to preserve mapping to original userids; otherwise this information is lost
    #Dump to hard disk
    sparse.save_npz("train_partiton", train_partiton)
    sparse.save_npz("dev_partition", dev_partition)
    sparse.save_npz("test_partition", test_partition)
    pickle.dump(training_indicies, open("training_indicies", "wb"))
    pickle.dump(dev_indicies, open("dev_indicies", "wb"))
    pickle.dump(dev_indicies, open("dev_indicies", "wb"))

    print("Created Partitions and dumped to Hard Disk!")

    return


def load_partitions():
    '''
    Note that partitions are in movie by user format. 
    That is, each row is the ratings given to a movie by each user.
    '''
    train_partiton = sparse.load_npz("train_partiton.npz")
    dev_partition = sparse.load_npz("dev_partition.npz")
    test_partition = sparse.load_npz("test_partition.npz")

    return train_partiton, dev_partition, test_partition

def valiant_preprocessing(dataset, threshold = 3):
    '''
    Convert ratings from 0-5 to 0, 1 with discretization. 
    Everything strictly greater than the threshold is a 1. 
    Everything below and including the threshold is a 0.
    '''
    dataset[dataset < threshold + 1] = 0 #< comparison is inefficient?
    dataset[dataset > threshold] = 1
    return dataset
            
if __name__ == '__main__':
    data_matrix, userid_indicies_mapping = load_ratings()
    sparse.save_npz("data_matrix", data_matrix)
    pickle.dump(userid_indicies_mapping, open("userid_indicies_mapping", "wb"))
    print("Loaded Data into Sparse CSR Matrix; Loaded User IDs into bijective mapping with indicies in Matrix; Dumped!")
    create_partition()


