# coding: utf-8
# Random Forest Classifier from Scratch

import pandas as pd
import numpy as np
import sys
import dtree as dt


class RandomForestClassifier(dt.DecisionTreeClassifier):
    
    ## Bagging Samples
    def _bagSamples(self, m):
        b = np.random.choice(m, size=m, replace=True)
        oob = [i for i in xrange(m) if i not in b]
        return b, oob
    
    
    # ## Random Features Selection
    def _buildForest(self, X, y, nIters=100, maxDepth=50, minLeafSize=1):
        trees = []
        oobs = []
        
        for k in xrange(nIters):
            sys.stdout.write("\rIteration #%d" % k)
            sys.stdout.flush()
            
            b, oob = self._bagSamples(len(y))
            X_k = X.iloc[b]
            y_k = y.iloc[b]
            tree_k = self._buildTree(X_k, y_k, maxDepth=maxDepth, minLeafSize=minLeafSize, randomFeatrue=True)
            trees.append(tree_k)
            oobs.append(oob)
            
        return trees, oobs
    
    
    # ## Vote Classifier Majority
    def _predictForest(self, row, trees):
        p = []
        for k in xrange(len(trees)):
            p.append(self._predictTree(row, trees[k]))
    
        return self._voteMajority(p)
    
    
    def train(self, X, y, option):
        nIters = option.nIters
        maxDepth = option.maxDepth
        minLeafSize = option.minLeafSize
        trees, oobs = self._buildForest(X, y, nIters=nIters, maxDepth=maxDepth, minLeafSize=minLeafSize)
        model = dt.Model()
        model.trees = trees
        model.oobs = oobs
        model.classes = pd.Series(y.unique())
        return model
    

    def predict(self, row, model):
        trees = model.trees
        return self._predictForest(row, trees)
    
    
    def oobErrorHist(self, X, y, model):
        trees = model.trees
        oobs = model.oobs
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
            T = T + 1.*(t.reshape((m, 1)) == np.ones((m, 1)).dot(c)) #891x2
            h = classes[np.argmax(T, axis=1)]
            err_hist[k] = np.mean(h!=y)
        
        return err_hist
        
        
    def errorHist(self, X, y, model):
        trees = model.trees
        o = len(trees)
        err_hist = np.zeros(o)
 
        for k in xrange(o):
            h = X.apply(lambda row: self._predictForest(row, trees[0:k+1]), axis=1)                
            err_hist[k] = np.mean(h!=y)
                
        return err_hist


    def score(self, X, y, model):
        trees = model.trees
        h = X.apply(lambda row: self._predictForest(row, trees), axis= 1)
        return np.mean(h==y)