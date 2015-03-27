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













if __name__ == '__main__':
    gen_train()
#    gen_test()