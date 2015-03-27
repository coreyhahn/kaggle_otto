# -*- coding: utf-8 -*-
"""
Created on Sun Mar 22 15:54:49 2015

@author: cah
"""
from sklearn.cross_validation import cross_val_score
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
import cPickle
import kaggle_csv as kcsv
from sklearn.metrics import make_scorer
import scipy.optimize

def ada_boost_search():
    savefile = open('traindata.pkl', 'rb')
    (x_train, y_train, t1) = cPickle.load(savefile)
    savefile.close()
    savefile = open('testdata.pkl', 'rb')
    (x_test, t1, name1) = cPickle.load(savefile)
    savefile.close()
    
    x_train = np.asarray(x_train,dtype=np.float32)
    y_train = np.asarray(y_train, dtype='int32')-1   
    
    xopt = scipy.optimize.fmin(func=minfunc_wrapper, x0=[1,.1,1],args=(x_train,y_train),disp=True,retall=True,xtol=.5)
    
    print xopt

def minfunc_wrapper(x,*args):
    [nest,lr,md] = x
    nest = np.int32(nest*20)
    md = np.int32(md*8)
    x_train = args[0]
    y_train = args[1]
    clf = GradientBoostingClassifier(n_estimators=nest, learning_rate=lr, max_depth=md, random_state=0)
    multiclass_log_loss = make_scorer(score_func=logloss_mc, greater_is_better=False, needs_proba=True)
    scores = cross_val_score(clf, x_train, y_train, n_jobs=8, cv=5,scoring=multiclass_log_loss)
    print (nest, lr, md, scores.mean())
    return -scores.mean()
    
def ada_boost():
    savefile = open('traindata.pkl', 'rb')
    (x_train, y_train, t1) = cPickle.load(savefile)
    savefile.close()
#    savefile = open('testdata.pkl', 'rb')
#    (x_test, t1, name1) = cPickle.load(savefile)
#    savefile.close()
    
    X = np.reshape(X,(X.shape[0],-1))
    y = np.asarray(y, dtype='int32')
    X_train, X_valid, y_train, y_valid = cross_validation.train_test_split(
    X, y, test_size=0.1, random_state=42)
    
    x_train = np.asarray(x_train,dtype=np.float32)
    y_train = np.asarray(y_train, dtype='int32')-1   
    
    nest = 200
    lr = .08
    md = 9
#    clf1 = DecisionTreeClassifier(max_depth=2)
#    clf = AdaBoostClassifier(clf1, n_estimators=200, learning_rate=.25)
    clf = GradientBoostingClassifier(n_estimators=nest, learning_rate=lr, max_depth=md, random_state=0)
#    clf = RandomForestClassifier(n_estimators=200) #.81
#    clf = ExtraTreesClassifier(n_estimators=1000, max_depth=None, min_samples_split=10, random_state=0,n_jobs=8) #.81
#    clf = KNeighborsClassifier(15)
    if 0:
        clf.fit(x_train, y_train)
        ypred = clf.predict_proba(x_test)
        y_str = ['Class_1','Class_2','Class_3','Class_4','Class_5','Class_6','Class_7','Class_8','Class_9']
        kcsv.print_csv(ypred, name1, y_str,indexname='id')
        print (nest, lr, md) 
    
    if 1:
        multiclass_log_loss = make_scorer(score_func=logloss_mc, greater_is_better=False, needs_proba=True)
        scores = cross_val_score(clf, x_train, y_train, n_jobs=8, cv=5,scoring=multiclass_log_loss)
        print scores
        print (nest, lr, md, scores.mean())  
    
def logloss_mc(y_true, y_prob, epsilon=1e-15):
    """ Multiclass logloss
    This function is not officially provided by Kaggle, so there is no
    guarantee for its correctness.
    """
    # normalize
    y_prob = y_prob / y_prob.sum(axis=1).reshape(-1, 1)
    y_prob = np.maximum(epsilon, y_prob)
    y_prob = np.minimum(1 - epsilon, y_prob)
    # get probabilities
    y = [y_prob[i, j] for (i, j) in enumerate(y_true)]
    ll = - np.mean(np.log(y))
    return ll

    
if __name__ == '__main__':
#    ada_boost()
    ada_boost_search()