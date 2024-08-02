import os
import sys
from sklearn.datasets import make_blobs

sys.path.insert(1, os.getcwd() + "/src")
from signal_processing.iotcupid import IoTCupidProcessing

import matplotlib.pyplot as plt
def test_iotcupid_cmeans_clustering():
    X, _ = make_blobs(n_samples=300, centers=3, cluster_std=0.5, random_state=0)
    

    cntr, u, c, score = IoTCupidProcessing.fuzzy_cmeans_w_elbow_method(X.T, 5, 0.05, 1.5, 3, 100)
    
    print(score)
    print(c)
    print(cntr)
    plt.scatter(X[:,0], X[:,1])
    plt.scatter(cntr[0,:], cntr[1,:])
    plt.show()

    #assert c == 3

if __name__ == "__main__":
    test_iotcupid_cmeans_clustering()