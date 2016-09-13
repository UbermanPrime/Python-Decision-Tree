import numpy as np
import pandas as pd

class KFold:
    X_train = None
    X_test = None
    y_tran = None
    y_test = None

def sample(X, y, nFolds=3):
    m = len(X)
    i = np.random.permutation(np.arange(m))
    foldSize = m / nFolds
    kf = [None]*nFolds

    i_from = 0
    for f in xrange(nFolds-1):
        i_to = i_from + foldSize
        kf[f] = KFold()
        kf[f].X_test = X.iloc[i[i_from:i_to]]
        kf[f].y_test = y.iloc[i[i_from:i_to]]
        kf[f].X_train = X.iloc[np.setdiff1d(i, i[i_from:i_to])]
        kf[f].y_train = y.iloc[np.setdiff1d(i, i[i_from:i_to])]
        i_from = i_to

    kf[f+1] = KFold()
    kf[f+1].X_test = X.iloc[i[i_from:]]
    kf[f+1].y_test = y.iloc[i[i_from:]]
    kf[f+1].X_train = X.iloc[np.setdiff1d(i, i[i_from:])]
    kf[f+1].y_train = y.iloc[np.setdiff1d(i, i[i_from:])]
    return kf
      

def train(kf, classifier, option):
    nFolds = len(kf)
    models = [None]*nFolds
    for f in xrange(nFolds):
        models[f] = classifier.train(kf[f].X_train, kf[f].y_train, option)
        
    return models
    
        
def predict(kf, classifier, models):
    nFolds = len(models)
    hs = [None]*nFolds
    for f in xrange(nFolds):
        hs[f] = kf[f].X_test.apply(lambda row: classifier.predict(row, models[f]), axis=1)
        
    return pd.concat(hs).sort_index()


def score(X, y, classifier, option, nFolds=3):
    kf = sample(X, y, nFolds=nFolds)
    models = train(kf, classifier, option)
    h = predict(kf, classifier, models)
    return np.mean(h==y)


def errorHist(X, y, classifier, option, nFolds=3):
    o = option.nIters
    err_hists = np.zeros((nFolds, o))
    kf = sample(X, y, nFolds=nFolds)
    models = train(kf, classifier, option)
    for f in xrange(nFolds):
        err_hists[f] = classifier.errorHist(kf[f].X_test, kf[f].y_test, models[f])
    
    return err_hists.mean(axis=0)
    
