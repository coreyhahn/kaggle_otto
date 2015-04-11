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

def fun_wrap(x,*arg):
    gg = (x[0]-5)**2 + (x[1]-5)**2 + (x[2]-5)**2
    print (x,gg)
    return gg

def searchfun():
#    xopt = scipy.optimize.fmin(func=fun_wrap, x0=[1,.1,1],disp=True,retall=True,xtol=[1,.02,1])
#    xopt = scipy.optimize.fmin_bfgs(f=fun_wrap, x0=[1,1,1],epsilon=[1,.01,1],disp=True,retall=True)
    xopt = scipy.optimize.brute(func=fun_wrap,ranges=((0,10),(0,10),(0,10)),disp=True)
#    xopt = scipy.optimize.brute(func=fun_wrap,ranges=(slice(0,10,1),slice(0,10,.01),slice(0,10,1)),disp=True)
    print xopt

def search_wrap(x,args):
    x_train = args[0]
    y_train = args[1]
    clf = GradientBoostingClassifier(n_estimators=x[0], learning_rate=1.0*x[1]/10000, max_depth=x[2], random_state=0)
    multiclass_log_loss = make_scorer(score_func=logloss_mc, greater_is_better=False, needs_proba=True)
    scores = cross_val_score(clf, x_train, y_train, n_jobs=8, cv=5,scoring=multiclass_log_loss)
    
#    scores = (x[0]-54)**2 + (x[1]/100-.124)**2 + (x[2]-5.4)**2
    
    print (x[0], x[1], x[2], -scores.mean()) #scores) #
    return -scores.mean()  #scores #
    
def step_wise_search():
    savefile = open('traindata.pkl', 'rb')
    (x_train, y_train, t1) = cPickle.load(savefile)
    savefile.close()
    
    x_train = np.asarray(x_train,dtype=np.float32)
    y_train = np.asarray(y_train, dtype='int32')-1 
    
    
    init1 = [194,1000,6]
    steps = (1,1,1)
    bounds = ((0,220), (0,2000),(1,20))
    points = {}
    
    dim1 = len(init1)
    #initialize
    curpoint = init1
    score = search_wrap(x=curpoint,args=(x_train,y_train))
    points[hash(tuple(curpoint))] = score    
    best_score = np.Inf
    new_best_score = score
    
    while(best_score> new_best_score):
        best_score = new_best_score
        scoredim = np.zeros((2,dim1))
        for n in np.arange(dim1):
            nextpoint = curpoint[:]
            nextpoint[n] += np.int(1*steps[n])
            if curpoint_valid_check(nextpoint,bounds):
                if hash(tuple(nextpoint)) in points:
                    score = points[hash(tuple(nextpoint))]
                else:
                    score = search_wrap(x=nextpoint,args=(x_train,y_train))
                    points[hash(tuple(nextpoint))] = score
                
            else:
                score = np.Inf
            scoredim[0,n] = score    
            
            
            if score > best_score:
                nextpoint = curpoint[:]
                nextpoint[n] += np.int(-1*steps[n])
                if curpoint_valid_check(nextpoint,bounds):
                    if hash(tuple(nextpoint)) in points:
                        score = points[hash(tuple(nextpoint))]
                    else:
                        score = search_wrap(x=nextpoint,args=(x_train,y_train))
                        points[hash(tuple(nextpoint))] = score
                else:
                    score = np.Inf            
                scoredim[1,n] = score
            else:
                scoredim[1,n] = np.Inf
                
        bestind = np.argmin(scoredim)
        if bestind >= dim1:
            curpoint[bestind%dim1] += np.int(-1*steps[bestind%dim1])
        else:
            curpoint[bestind] += np.int(1*steps[bestind])
        new_best_score = np.min(scoredim)
        print ('Current Point', curpoint)
        print '--'
            
    print ('Best Point', curpoint, new_best_score)
    
    
def curpoint_valid_check(curpoint, bounds):
    for n in np.arange(len(curpoint)):
        if curpoint[n] < bounds[n][0]:
            return False
        if curpoint[n] > bounds[n][1]:
            return False   
    
    return True

def ada_boost_search():
    savefile = open('traindata.pkl', 'rb')
    (x_train, y_train, t1) = cPickle.load(savefile)
    savefile.close()
    savefile = open('testdata.pkl', 'rb')
    (x_test, t1, name1) = cPickle.load(savefile)
    savefile.close()
    
    x_train = np.asarray(x_train,dtype=np.float32)
    y_train = np.asarray(y_train, dtype='int32')-1   
    
    xopt = scipy.optimize.fmin(func=minfunc_wrapper, x0=[1,.1,1],args=(x_train,y_train),disp=True,retall=True,xtol=[1,.02,1])
    
    print xopt

def minfunc_wrapper(x,*args):
    [nest,lr,md] = x
    nest = np.int32(np.round(20**nest))
    md = np.int32(np.round(md*8))
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
    savefile = open('testdata.pkl', 'rb')
    (x_test, t1, name1) = cPickle.load(savefile)
    savefile.close()
    
#    X_train, X_valid, y_train, y_valid = cross_validation.train_test_split(
#    X, y, test_size=0.1, random_state=42)
    
    x_train = np.asarray(x_train,dtype=np.float32)
    y_train = np.asarray(y_train, dtype='int32')-1   
    
    nest = 190
    lr = .1
    md = 6
#    clf1 = DecisionTreeClassifier(max_depth=2)
#    clf = AdaBoostClassifier(clf1, n_estimators=200, learning_rate=.25)
    clf = GradientBoostingClassifier(n_estimators=nest, learning_rate=lr, max_depth=md, random_state=0)
#    clf = RandomForestClassifier(n_estimators=200) #.81
#    clf = ExtraTreesClassifier(n_estimators=1000, max_depth=None, min_samples_split=10, random_state=0,n_jobs=8) #.81
#    clf = KNeighborsClassifier(15)
    if 1:
        clf.fit(x_train, y_train)
        ypred = clf.predict_proba(x_test)
        y_str = ['Class_1','Class_2','Class_3','Class_4','Class_5','Class_6','Class_7','Class_8','Class_9']
        kcsv.print_csv(ypred, name1, y_str,indexname='id')
        print (nest, lr, md) 
    
    if 0:
        multiclass_log_loss = make_scorer(score_func=logloss_mc, greater_is_better=True, needs_proba=True)
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

def bern():
    from sklearn.naive_bayes import BernoulliNB
    
    savefile = open('traindata.pkl', 'rb')
    (x_train, y_train, t1) = cPickle.load(savefile)
    savefile.close()
    savefile = open('testdata.pkl', 'rb')
    (x_test, t1, t1) = cPickle.load(savefile)
    savefile.close()
    
    y_train = y_train-1
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
    
    x_new = np.zeros((x_train.shape[0],len1))
    for n in np.arange(x_train.shape[0]):
        for m in np.arange(x_train.shape[1]):
            key1 = m*1000+x_train[n,m]
            x_new[n,feat1[key1]] = 1
    
    x_train = np.hstack((x_train,x_new))        
    print 'done'    
#    clf = BernoulliNB()
#    clf = RandomForestClassifier(n_estimators=200)
    clf = GradientBoostingClassifier(n_estimators=100, learning_rate=.1, max_depth=8, random_state=0)
    multiclass_log_loss = make_scorer(score_func=logloss_mc, greater_is_better=False, needs_proba=True)
    scores = cross_val_score(clf, x_new, y_train, n_jobs=8, cv=5,scoring=multiclass_log_loss)
    
    print scores


if __name__ == '__main__':
#    ada_boost()
#    ada_boost_search()
#    searchfun()
    step_wise_search()
#    bern()