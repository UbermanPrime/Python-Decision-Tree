# coding: utf-8
# # C4.5 Decision Tree

import pandas as pd
import numpy as np
from scipy.stats import mode

# A structure to store information of a node in a tree 
class Tree:    
    prediction = None # return prediction if the node is a leaf
    feature = None # feature for splitting at a node
    splitPoint = None # class or threshold of the feature for splitting
    left = None # a point to left tree node or leaf 
    right = None # a point to right tree node or leaf 
    entropy = None # store entropy for analysis
    nCount  = None # size of the node or leaft
    distribution = None # distribution of target values at the node, e.g. [2 8]



class Option:
    nIters = 0
    maxDepth = 0
    minLeafSize = 0

    def __init__(self, nIters=10, maxDepth=4, minLeafSize=1):
        self.nIters = nIters
        self.maxDepth = maxDepth
        self.minLeafSize = minLeafSize

        
class Model:
    trees = []
    oobs = []
    classes = None
    alpha = None

    def __init__(self, trees=[], oobs=[], classes=None, alpha=None):
        self.trees = trees
        self.oobs = oobs
        self.classes = classes
        self.alpha = alpha


class DecisionTreeClassifier:
    # entropy funciton to support multiple classes
    # y takes Pandas Series or 1D Numpy array
    def _entropy(self, y):
        dummy, count = np.unique(y.values, return_counts=True)
        p = 1.*count/len(y)
        h = sum(-p*np.log2(p))
        return h
    
    
    def _informationGain(self, y, y_left, y_right):
        H_mother = self._entropy(y)
        H_left = self._entropy(y_left)
        H_right = self._entropy(y_right)
        H_childern = 1.*(len(y_left) * H_left + len(y_right) * H_right) / len(y)
        return H_mother - H_childern
    
    
    # test if an input feature is categorical or not based on its data type
    # input data X can be Pandas DataFrame, or a row in DataFrame
    def _isCategorical(self, X, feature):
        dtype = type(X[feature])
        if dtype == pd.core.series.Series:
            dtype = X[feature].dtype
    
        return (dtype==str or dtype==object)
    
    
    # find all possible split points for a feature of both categorical or numeric
    def _allSplitPoints(self, X, y, feature):
        # if catgegorical, just returns all unique classes
        if self._isCategorical(X, feature):
            return X[feature].unique()
        
        # if numeric, sort on input feature, return thresholds where change of classes on y occurs 
        thresholds = []
        view = pd.concat([X[feature], y], axis=1).copy()
        view.sort_values(feature, inplace=True)
        iterator = view.iterrows()
        last_row = iterator.next()[1]
        for dummy, row in (view.iterrows()):
            if row[y.name] != last_row[y.name]:
                thresholds.append((row[feature]+last_row[feature])/2.)
            last_row = row
        return thresholds
    
    
    # common function for all feature values comparision in the package
    def _compare(self, X, feature, splitPoint):
        # if feature is categorical, test if it falls in the splitPoint as class
        if self._isCategorical(X, feature):
            return X[feature] == splitPoint
        
        # if features is numeric, test if it's within the splitPoint as threshold
        return X[feature] <= splitPoint
    
    
    # find the best split point for a feature based on information gain
    def _findBestSplitPoint(self, X, y, feature):
        best_ig = 0
        best_splitPoint = None
        
        # loop for all possible split points of a feature
        for splitPoint in self._allSplitPoints(X, y, feature):
            y_left = y.loc[self._compare(X,feature,splitPoint)]
            y_right = y.loc[np.logical_not(self._compare(X,feature,splitPoint))]
            ig = self._informationGain(y, y_left, y_right)
            # test if the current ig is better than previous, beware rounding
            if ig - best_ig > 0.0001:
                best_ig = ig
                best_splitPoint = splitPoint
    
        # return None, if no information gain at all
        return (best_splitPoint, best_ig)
    
    
    # find the best feature to split at a node
    # randomFeature is reserved for random forest
    def _findBestFeature(self, X, y, randomFeatrue=False):
        best_feature = None
        best_splitPoint = None
        best_ig = 0
        
        # for random forest, randomly pick sqrt(n) features to test
        if randomFeatrue:
            n = int(len(X.columns)**0.5)
            columns = np.random.choice(X.columns, n, replace=False)
        else:
            # otherwise, test all features
            columns = X.columns
    
        for feature in columns:
            # for each feature, find it's best split point and ig
            splitPoint, ig = self._findBestSplitPoint(X, y, feature)
            # test if the current ig is better than previous, beware rounding
            if ig - best_ig > 0.0001:
                best_ig = ig
                best_feature = feature
                best_splitPoint = splitPoint
    
        # return None if no information gain at all
        return (best_feature, best_splitPoint)
    
    
    # split dataset (X,y) given a feature and split pont to split
    def _splitData(self, X, y, feature, splitPoint):
        # the splitted dataset should be returned as reference, for performance consideration
        # beware the Pandas characteristic, always use ".loc[]"
        X_left = X.loc[self._compare(X,feature,splitPoint)]
        y_left = y.loc[self._compare(X,feature,splitPoint)]
        X_right = X.loc[np.logical_not(self._compare(X,feature,splitPoint))]
        y_right = y.loc[np.logical_not(self._compare(X,feature,splitPoint))]
        return (X_left, y_left, X_right, y_right)
    
    
    def _voteMajority(self, y):
        # beware the weird characteristic of Pandas mode function
        # use Scipy instead
        return mode(y)[0][0]


    # build tree recursively, return the node built
    # X takes Pandas DataFrame, elements can be int, float or string
    # y takes Pandas Series, elements can be int, float or string
    # depth was used internally, can be ignored
    # maxDepth is a model parameter to control the maximum depth of the tree
    # minLeafSize is another model parameter, which the leaf size to prevent overfitting
    # randomFeature is reserved for random forest, when on, random features are selected to split
    def _buildTree(self, X, y, depth=0, maxDepth=50, minLeafSize=5, randomFeatrue=False):
    #    print "depth #%d" % depth    
        tree = Tree()
        tree.entropy = self._entropy(y)
        tree.ncount = y.count()
        tree.distribution = y.value_counts(sort=False).values
    
        # stop building when node size decreased to minimum leaf size limit
        if len(y)<=minLeafSize:
            tree.prediction = self._voteMajority(y)
            return tree
    
        # stop building when all target values are the same
        if y.max()==y.min():
            tree.prediction = self._voteMajority(y)
            return tree
    
        # stop building when the node depth reaches the maxmium tree depth limit
        if depth==maxDepth:
            tree.prediction = self._voteMajority(y)
            return tree
    
        # try to find the best feature and split point at this node
        feature, splitPoint = self._findBestFeature(X, y, randomFeatrue)
        
        # stop building when no information gain for any splits
        if feature==None:
            tree.prediction = self._voteMajority(y)
            return tree
    
        tree.feature = feature
        tree.splitPoint = splitPoint
        
        # split the node dataset (X,y) into two, and call recursively
        (X_left, y_left, X_right, y_right) = self._splitData(X, y, feature, splitPoint)
        depth = depth + 1
        tree.left=self._buildTree(X_left, y_left, depth, 
                            maxDepth=maxDepth, minLeafSize=minLeafSize, 
                            randomFeatrue=randomFeatrue)
        tree.right=self._buildTree(X_right, y_right, depth, 
                            maxDepth=maxDepth, minLeafSize=minLeafSize, 
                            randomFeatrue=randomFeatrue)
        
        # return the tree node built
        return tree
    
    
    # return the prediction for an instance of a built tree
    # row takes a row from a Pandas DataFrame like X
    # tree is a built tree returned from the buildTree function
    def _predictTree(self, row, tree):
        if tree.prediction != None:
            return tree.prediction
    
        if self._compare(row, tree.feature, tree.splitPoint):
            return self._predictTree(row, tree.left)
        
        return self._predictTree(row, tree.right)

    def train(self, X, y, option):
        maxDepth = option.maxDepth
        minLeafSize = option.minLeafSize
        tree = self._buildTree(X, y, maxDepth=maxDepth, minLeafSize=minLeafSize, randomFeatrue=False)
        return Model(trees=[tree])
        
            
    def predict(self, row, model):
        tree = model.trees[0]
        return self._predictTree(row, tree)

    # score a tree model, returns accuracy in percentage
    # X is input that takes Pandas DataFrame
    # y is target that takes Pandas Series
    def score(self, X, y, model):
        tree= model.trees[0]
        h = X.apply(lambda x: self._predictTree(x, tree), axis=1)
        return np.mean(h==y)

# End class DTreeClassifier    
