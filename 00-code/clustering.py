# -*- coding: utf-8 -*-
"""
Created on Fri Mar 27 17:25:55 2015

@author: cah
"""

from sklearn.cluster import AffinityPropagation
from sklearn.cluster import DBSCAN
from sklearn.cluster import MiniBatchKMeans
from sklearn import cross_validation
import cPickle
from sklearn import metrics
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import time
from sklearn.metrics import silhouette_score


def kmeans_test():
    
    savefile = open('traindata.pkl', 'rb')
    (x_train, y_train, t1) = cPickle.load(savefile)
    savefile.close()
#    x_train, t1, y_train, t2 = cross_validation.train_test_split(
#        x_train, y_train, test_size=0.10, random_state=42)   
    savefile = open('testdata.pkl', 'rb')
    (x_test, t1, name1) = cPickle.load(savefile)
    savefile.close()
    
    for clus in np.arange(2,100):
#        clus = 1000
        mbk = MiniBatchKMeans(init='k-means++', n_clusters=clus, batch_size=128,
                          n_init=10, max_no_improvement=10, verbose=0)
        mbk.fit(x_train)
        silhouette_avg = silhouette_score(x_train, mbk.labels_)
        print (clus,silhouette_avg)
    
    data1 = mbk.labels_
    filename = 'train_kmeans' + str(clus)
    savefile = open(filename+'.pkl', 'wb')
    cPickle.dump((data1, [], []),savefile,-1)
    savefile.close()     
    
    
def affin_test():
    savefile = open('traindata.pkl', 'rb')
    (x_train, y_train, t1) = cPickle.load(savefile)
    savefile.close()
    
     
    x_train, X_valid, y_train, y_valid = cross_validation.train_test_split(
        x_train, y_train, test_size=0.9, random_state=42)    
    
    
    labels_true = y_train 
    
    x_train = StandardScaler().fit_transform(x_train)
    af = AffinityPropagation(preference=-50).fit(x_train)
    cluster_centers_indices = af.cluster_centers_indices_
    labels = af.labels_
    
    n_clusters_ = len(cluster_centers_indices)
    
    print('Estimated number of clusters: %d' % n_clusters_)
    print("Homogeneity: %0.3f" % metrics.homogeneity_score(labels_true, labels))
    print("Completeness: %0.3f" % metrics.completeness_score(labels_true, labels))
    print("V-measure: %0.3f" % metrics.v_measure_score(labels_true, labels))
    print("Adjusted Rand Index: %0.3f"
          % metrics.adjusted_rand_score(labels_true, labels))
    print("Adjusted Mutual Information: %0.3f"
          % metrics.adjusted_mutual_info_score(labels_true, labels))
    print("Silhouette Coefficient: %0.3f"
          % metrics.silhouette_score(x_train, labels, metric='sqeuclidean'))
    
def dbscan_test():
    savefile = open('traindata.pkl', 'rb')
    (x_train, y_train, t1) = cPickle.load(savefile)
    savefile.close()
#    x_train, t1, y_train, t2 = cross_validation.train_test_split(
#        x_train, y_train, test_size=0.10, random_state=42)   
    savefile = open('testdata.pkl', 'rb')
    (x_test, t1, name1) = cPickle.load(savefile)
    savefile.close()
    
    norm1 = StandardScaler().fit(x_train)
    x_train = norm1.transform(x_train)
    x_test = norm1.transform(x_test)
    
    x_t1 = np.vstack((x_train, x_test))
    
    eps1=7
    db = DBSCAN(eps=eps1, min_samples=5).fit(x_t1)
    
    core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
    core_samples_mask[db.core_sample_indices_] = True
    labels = db.labels_    
    
#    labels_true = y_train

        
      # Number of clusters in labels, ignoring noise if present.
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    n_ofclusters_ = 1.0-(1.0*np.sum(1.0*(labels == -1)))/len(labels)
    print('Estimated number of clusters: %d' % n_clusters_)
    print('Estimated in cluster ratio: %f' % n_ofclusters_)
#    print("Homogeneity: %0.3f" % metrics.homogeneity_score(labels_true, labels))
#    print("Completeness: %0.3f" % metrics.completeness_score(labels_true, labels))
#    print("V-measure: %0.3f" % metrics.v_measure_score(labels_true, labels))
#    print("Adjusted Rand Index: %0.3f"
#          % metrics.adjusted_rand_score(labels_true, labels))
#    print("Adjusted Mutual Information: %0.3f"
#          % metrics.adjusted_mutual_info_score(labels_true, labels))
#    print("Silhouette Coefficient: %0.3f"
#          % metrics.silhouette_score(x_train, labels))  
    
    data1 = labels[:x_train.shape[0]]
    filename = 'traindbscan' + str(eps1)
    savefile = open(filename+'.pkl', 'wb')
    cPickle.dump((data1, [], []),savefile,-1)
    savefile.close()           

def tsne_test():  
    savefile = open('traindata.pkl', 'rb')
    (x_train, y_train, t1) = cPickle.load(savefile)
    savefile.close()
    x_train, X_valid, y_train, y_valid = cross_validation.train_test_split(
        x_train, y_train, test_size=0.90, random_state=42)  
        
    x_train = StandardScaler().fit_transform(x_train) 
    x_train = np.asarray(x_train, dtype=np.float64)
    color = 1.0*y_train/np.max(y_train)
    model = TSNE(n_components=2, random_state=0)    
    d2_x_train = model.fit_transform(x_train)  
    plt.scatter(d2_x_train[:, 0], d2_x_train[:, 1], c=color, cmap=plt.cm.Spectral)   
    plt.show()
    pause(10)
    
        
if __name__ == '__main__':
#    tsne_test()
#    dbscan_test()        
#    affin_test()    
    kmeans_test()
        
        
        