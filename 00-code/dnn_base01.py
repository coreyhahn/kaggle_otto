# -*- coding: utf-8 -*-
"""
Created on Wed Mar 25 22:28:25 2015

@author: cah
"""
import cPickle
import numpy as np
from sklearn import cross_validation
from sklearn.metrics import classification_report
from sklearn.metrics import zero_one_loss
from lasagne import layers
from lasagne.updates import nesterov_momentum
from lasagne.updates import adagrad, adadelta
from nolearn.lasagne import NeuralNet
from lasagne.nonlinearities import softmax
from lasagne.nonlinearities import sigmoid
import theano
from sklearn import preprocessing as pp
from sklearn.decomposition import PCA
import kaggle_csv as kcsv

def shuffle(*arrays):
    p = np.random.permutation(len(arrays[0]))
    return [array[p] for array in arrays]
        
class BatchIterator(object):
    def __init__(self, batch_size):
        self.batch_size = batch_size

    def __call__(self, X, y=None):
        self.X, self.y = X, y
        return self
        
 
    def __iter__(self):
        n_samples = self.X.shape[0]
        bs = self.batch_size
        self.X, self.y = shuffle(self.X, self.y)
        for i in range((n_samples + bs - 1) // bs):
            sl = slice(i * bs, (i + 1) * bs)
            Xb = self.X[sl]
            if self.y is not None:
                yb = self.y[sl]
            else:
                yb = None
            yield self.transform(Xb, yb)

    def transform(self, Xb, yb):
        return Xb, yb




class AdjustVariable(object):
    def __init__(self, name, start=0.03, stop=0.001):
        self.name = name
        self.start, self.stop = start, stop
        self.ls = None

    def __call__(self, nn, train_history):
        if self.ls is None:
            self.ls = np.linspace(self.start, self.stop, nn.max_epochs)

        epoch = train_history[-1]['epoch']
        new_value = np.float32(self.ls[epoch - 1])
        getattr(nn, self.name).set_value(new_value)
        

if __name__ == "__main__":
    savefile = open('traindata.pkl', 'rb')
    (x_train, y_train, t1) = cPickle.load(savefile)
    savefile.close()
    
    savefile = open('testdata.pkl', 'rb')
    (x_test, t1, name1) = cPickle.load(savefile)
    savefile.close()
        
    scaler = pp.StandardScaler().fit(x_train)
    x_train_scale = scaler.transform(x_train)
    x_test_scale = scaler.transform(x_test)
    
    wpca = PCA(whiten=True)
    wpca.fit(x_train)
    x_train_wpca = wpca.transform(x_train)
    x_test_wpca = wpca.transform(x_test)
    
    x_train = np.hstack((x_train, x_train_scale, x_train_wpca))
    x_test = np.hstack((x_test, x_test_scale, x_test_wpca))
    
    x_train = np.asarray(x_train,dtype=np.float32)
    y_train = np.asarray(y_train, dtype='int32')-1
    
    x_train, x_test, y_train, y_test = cross_validation.train_test_split(
                      x_train, y_train, test_size=0.1, random_state=42)
    
    clf = NeuralNet(
        layers=[('input',layers.InputLayer),                
                ('hidden0', layers.DenseLayer),
                ('dropout0', layers.DropoutLayer), 
                ('hidden1', layers.DenseLayer),
                ('dropout1', layers.DropoutLayer), 
#                ('hidden2', layers.DenseLayer),
#                ('dropout2', layers.DropoutLayer), 
#                ('hidden3', layers.DenseLayer),
                ('output', layers.DenseLayer)],        
        # layer parameters:
        batch_iterator_train=BatchIterator(batch_size=512),
        input_shape=(None, x_train.shape[1]),  # 96x96 input pixels per batch
        hidden0_num_units=1024,  # number of units in hidden layer
        hidden1_num_units=1024,  # number of units in hidden layer
        hidden2_num_units=1024,  # number of units in hidden layer
#        hidden3_num_units=256,  # number of units in hidden layer
        output_nonlinearity=softmax,  # output layer uses identity function
        output_num_units=len(set(y_train)),  # 30 target values
        dropout0_p=0.5,
        dropout1_p=0.5,
        eval_size=.2,
        dropout2_p=0.5,
        # optimization method:
#        update=adagrad,
#        update_learning_rate=.1,
        update=nesterov_momentum,
        update_learning_rate=theano.shared(np.float32(0.01)),
        update_momentum=theano.shared(np.float32(0.9)),
#        
        on_epoch_finished=[
            AdjustVariable('update_learning_rate', start=0.01, stop=0.00001),
            AdjustVariable('update_momentum', start=0.9, stop=0.999),
            ],
        regression=False,  # flag to indicate we're dealing with regression problem
        max_epochs=100,  # we want to train this many epochs
        verbose=1        
        )                 
#    clf = dbn([x_train.shape[1],1024,1024,256,len(set(y_train))])      

    clf.fit(x_train,y_train)          
    
    if 1:
        y_pred = clf.predict(x_test)
        print "Accuracy:", zero_one_loss(y_test, y_pred)
        print "Classification report:"
        print classification_report(y_test, y_pred)
    else:
        ypred = clf.predict_proba(x_test)
        y_str = ['Class_1','Class_2','Class_3','Class_4','Class_5','Class_6','Class_7','Class_8','Class_9']
        kcsv.print_csv(ypred, name1, y_str,indexname='id')
                          
    