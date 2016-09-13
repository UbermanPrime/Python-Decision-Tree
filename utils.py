# coding: utf-8
# # C4.5 Decision Tree

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from graphviz import Digraph
from uuid import uuid4
import re
import dtree as dt
import rforest as rf
import adaBoost as adb
import crossvalidator as cv    
    
def drawGraph(graph, tree):
    node_id = uuid4().hex
    if tree.prediction != None:        
        graph.node(node_id, shape="box", 
                 label="%s\nentropy = %.4f\nsampels = %d\ny %s" 
                 % (tree.prediction, tree.entropy, tree.ncount, 
                    tree.distribution))
        return node_id
    
    graph.node(node_id, shape="box", 
             label="%s | %s\nentropy = %.4f\nsamples = %d\ny %s" 
             % (tree.feature, tree.splitPoint, tree.entropy, tree.ncount, 
                tree.distribution))
    left_id = drawGraph(graph, tree.left)
    graph.edge(node_id, left_id, label="left")
    right_id = drawGraph(graph, tree.right)
    graph.edge(node_id, right_id, label="right")
    return node_id
    
    
def drawTree(model, title=''):
    graph = Digraph(comment=title)
    tree = model.trees[0]
    drawGraph(graph, tree)
    return graph 

# uncomment to draw your tree
#tree = trainTitanic(maxDepth=3, minLeafSize=1)
#graph = drawTree(tree, title='Titanic Survival Descision Tree')
#graph


def prepareTitanic(filename):
    df = pd.read_csv(filename)
    df["Title"] = df["Name"].apply(lambda name: re.search(' ([A-Za-z]+)\.', name).group(1))
    df["Age"] = df["Age"].groupby(df["Title"]).transform(lambda age: age.fillna(age.median()))
    df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode().iloc[0])
    df["HasCabin"] = df["Cabin"].apply(lambda x: "true" if pd.notnull(x) else "false")
    df["FamilySize"] = df["SibSp"] + df["Parch"]
    return df


def trainTitanic(classifier, option):
    # load the traning data
    df = prepareTitanic("train.csv")
    X = df[["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked", "Title", "HasCabin", "FamilySize"]]
    y = df["Survived"]
    # train a tree
    model = classifier.train(X, y, option)
    # score the accuracy
    accuracy = classifier.score(X, y, model)[0]
    print ("Accuracy %.2f"%accuracy)

    return model


def submitTitanic(classifier, model):
    df_test = prepareTitanic("test.csv")
    X_test = df_test[["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked", "Title", "HasCabin", "FamilySize"]]
    p_test = X_test.apply(lambda row: classifier.predict(row, model), axis=1)
    
    submission = pd.DataFrame({"PassengerId": df_test["PassengerId"], "Survived": p_test})
    submission.to_csv("dtree.csv", index=False)
    return

#uncomment to run
#tree = trainTitanic(maxDepth=3, minLeafSize=1)
#submitKaggle(tree)

def tuneMaxDepth(X, y, classifier, nIters=100, nFolds=3, minLeafSize=1):
    maxDepths = [1, 2, 3, 4]
    rfOpt = dt.Option(nIters=nIters, minLeafSize=1)
    for maxDepth in maxDepths:
        rfOpt.maxDepth = maxDepth
        err_hist = cv.errorHist(X, y, classifier, rfOpt, nFolds=3)
        plt.plot(xrange(nIters), err_hist, label="maxDepth=%d"%maxDepth)
        
    plt.title("CV Error vs No. of Trees")
    plt.xlabel("No. of Trees")
    plt.ylabel("CV Error")
    plt.legend()
    plt.show()
    return
    
    