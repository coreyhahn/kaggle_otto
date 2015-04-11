# -*- coding: utf-8 -*-
"""
Created on Sat Mar 21 20:56:47 2015

@author: cah
"""

from numpy import genfromtxt
import numpy as np
import cPickle

datastore = '/home/cah/Desktop/otto/'

def gen_train():
    labeldata = genfromtxt(datastore + '' + 'train.csv', delimiter=',' \
    ,names=True,dtype=[np.float32 for x in np.arange(94)].append('S64')) 
    
    numofrows = labeldata.size
    class1 = np.zeros((numofrows),dtype=np.float32)
    name1 = np.zeros((numofrows),dtype=np.int32)
    data1 = np.zeros((numofrows,93),dtype=np.float32)
    for n in np.arange(numofrows):
        class1[n] = np.int32(float(labeldata[n][-1][-1]))
        name1[n] = labeldata[n][0]
        for m in np.arange(1,94):
            data1[n,m-1] = labeldata[n][m]
        
    filename = 'traindata'
    savefile = open(filename+'.pkl', 'wb')
    cPickle.dump((data1, class1, name1),savefile,-1)
    savefile.close()       
    print 'done'


def gen_test():
    labeldata = genfromtxt(datastore + '' + 'test.csv', delimiter=',' \
    ,names=True,dtype=[np.float32 for x in np.arange(93)].append('S64')) 
    
    numofrows = labeldata.size
    class1 = np.zeros((numofrows),dtype=np.float32)
    name1 = np.zeros((numofrows),dtype=np.int32)
    data1 = np.zeros((numofrows,93),dtype=np.float32)
    for n in np.arange(numofrows):
        class1[n] = 0 #np.int32(float(labeldata[n][-1][-1]))
        name1[n] = labeldata[n][0]
        for m in np.arange(1,94):
            data1[n,m-1] = labeldata[n][m]
        
    filename = 'testdata'
    savefile = open(filename+'.pkl', 'wb')
    cPickle.dump((data1, class1, name1),savefile,-1)
    savefile.close()       
    print 'done'

def binarize():
    savefile = open('traindata.pkl', 'rb')
    (x_train, class1, name1) = cPickle.load(savefile)
    savefile.close()
    savefile = open('testdata.pkl', 'rb')
    (x_test, t2, name2) = cPickle.load(savefile)
    savefile.close()
    
    feat1 = {}  
    len1 = 0
    p = 0
    x1 = np.vstack((x_train,x_test))
    for n in np.arange(x1.shape[1]):
        s1 = set(x1[:,n])
        len1 += len(s1)
        for m in s1:
            feat1[1000*n+m] = p
            p += 1
        
        print (n, len(s1))        
    
    x_new = np.zeros((x_train.shape[0],len1),dtype=np.uint8)
    for n in np.arange(x_train.shape[0]):
        for m in np.arange(x_train.shape[1]):
            key1 = m*1000+x_train[n,m]
            x_new[n,feat1[key1]] = 1
            
    x_train_new = x_new
    
    x_new = np.zeros((x_test.shape[0],len1),dtype=np.uint8)
    for n in np.arange(x_test.shape[0]):
        for m in np.arange(x_test.shape[1]):
            key1 = m*1000+x_test[n,m]
            x_new[n,feat1[key1]] = 1

    x_test_new = x_new
    
    filename = 'traindata_b'
    savefile = open(filename+'.pkl', 'wb')
    cPickle.dump((np.uint8(x_train_new), class1, name1),savefile,-1)
    savefile.close()   

    filename = 'testdata_b'
    savefile = open(filename+'.pkl', 'wb')
    cPickle.dump((np.uint8(x_test_new), [], name2),savefile,-1)
    savefile.close()  








if __name__ == '__main__':
#    gen_train()
#    gen_test()
    binarize()