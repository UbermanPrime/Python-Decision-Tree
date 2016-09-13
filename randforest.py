import pandas as pd
import numpy as np
from collections import defaultdict
import dtree as dt

def bagSamples(m):
    b = np.random.choice(m, size=m, replace=True)
    oob = [i for i in range(m) if i not in b]
    return b, oob


def buildForest(X, y, nTrees=100, maxDepth=50, minLeafSize=1):
    trees = []
    oobMap = defaultdict(list)
    
    for k in range(nTrees):
        print "iteration #%d" % k 
        b, oob = bagSamples(len(y))
        X_k = X.iloc[b]
        y_k = y.iloc[b]
        tree_k = dt.buildTree(X_k, y_k, maxDepth=maxDepth, minLeafSize=minLeafSize, randomFeatrue=True)
        trees.append(tree_k)
        for i in oob:
            oobMap[i].append(k)
            
    return trees, oobMap

    
def predictForest(row, trees):
    p = []
    for k in range(len(trees)):
        p.append(dt.predictTree(row, trees[k]))
    
    return dt.voteMajority(p)


def oobScore(X, y, oobMap, trees, maxTreeNum=None):
    p = X.apply(lambda row: 
                    predictForest(row, 
                            [trees[k] for k in oobMap[row.name]],
                            maxTreeNum), 
                axis=1)
    return np.mean(p==y*1.)*100.
    
