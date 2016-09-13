import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import dtree as dt
import sys


class AdaBoostClassifier(dt.DecisionTreeClassifier):
    
    def _treeScore(self, err, p):
        return np.log((1.-err)/err) + np.log(p-1)
    
    
    def _boostSamples(self, w):
        m = len(w)
        i = np.random.choice(m, size=m, replace=True, p=w)
        oob = [x for x in xrange(m) if x not in i]
        return i, oob
    
    
    def _trainAdaBoost(self, X, y, nIters=100, maxDepth=2):
        trees = [None]*nIters
        oobs = [None]*nIters
        alpha = np.zeros(nIters)
        model = dt.Model()
        model.classes = pd.Series(y.unique())
        
        m = len(y)
        w = np.ones(m) * 1. / m
        
        for k in xrange(nIters):
            sys.stdout.write("\rIteration #%d" % k)
            sys.stdout.flush()
            
            i, oobs[k] = self._boostSamples(w)
            trees[k] = self._buildTree(X.iloc[i], y.iloc[i], maxDepth=maxDepth, minLeafSize=1, randomFeatrue=True)
            
            h = X.apply(lambda row: self._predictTree(row, trees[k]), axis=1)
            err = w.dot(h!=y)
            if err > 0.5:
                print 'Error %.4f at %d' %(err, k)
            
            alpha[k] = self._treeScore(err, len(y.unique()))
            
            w[np.where(h!=y)] = w[np.where(h!=y)] * np.exp(alpha[k])
            w = w / sum(w)
        
        model.trees = trees
        model.oobs = oobs
        model.alpha = alpha
        return model

    
    def train(self, X, y, option):
        nIters = option.nIters
        maxDepth = option.maxDepth
        model = self._trainAdaBoost(X, y, nIters=nIters, maxDepth=maxDepth)
        return model    

    
    def predict(self, row, model):
        trees = model.trees
        classes = model.classes
        alpha = model.alpha
        
        o = len(trees)
        p = len(classes)
        
        alpha = alpha.reshape((o, 1)) 
        c =  np.array(classes.apply(hash)).reshape((1, p)) #1x2
        t_arr = [hash(self._predictTree(row, trees[k])) for k in xrange(o)]
        t = np.array(t_arr).reshape((o, 1)) #50x1
        T = alpha * (t == np.ones((o, 1)).dot(c)) #50x2
        return model.classes[np.argmax(sum(T))]
    

    def oobErrorHist(self, X, y, model):
        trees = model.trees
        oobs = model.oobs
        alpha = model.alpha
        classes = model.classes
        m = len(y)
        o = len(trees)
        p = len(classes)
        err_hist = np.zeros(o)
        
        c = np.array(classes.apply(hash)).reshape((1, p)) #1x2
        T = np.zeros((m, p))
        for k in xrange(o):        
            t = np.empty(m) * np.nan
            t[oobs[k]] = X.iloc[oobs[k]].apply(lambda row: hash(self._predictTree(row, trees[k])), axis=1) #891
            T = T + alpha[k] * (t.reshape((m, 1)) == np.ones((m, 1)).dot(c)) #891x2
            h = classes[np.argmax(T, axis=1)]
            err_hist[k] = np.mean(h!=y)
        
        return err_hist
        
        
    def errorHist(self, X, y, model):
        trees = model.trees
        alpha = model.alpha
        classes = model.classes
        m = len(y)
        o = len(trees)
        p = len(classes)
        err_hist = np.zeros(o)
        
        c = np.array(classes.apply(hash)).reshape((1, p)) #1x2
        T = np.zeros((m, p))
        for k in xrange(o):        
            t = np.empty(m) * np.nan
            t = X.apply(lambda row: hash(self._predictTree(row, trees[k])), axis=1) #891
            T = T + alpha[k] * (t.reshape((m, 1)) == np.ones((m, 1)).dot(c)) #891x2
            h = classes[np.argmax(T, axis=1)]
            err_hist[k] = np.mean(h!=y)
        
        return err_hist


    def score(self, X, y, model):
        h = X.apply(lambda row: self.predict(row, model), axis= 1)
        return np.mean(h==y)
        
        
#100, 1, 0.78947
#100, 2, 0.77990
#500, 1, 0.76077
#200, 8, 0.75598


