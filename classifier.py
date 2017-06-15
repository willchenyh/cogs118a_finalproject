#!/usr/bin/env python2
#
# Example to classify faces.
# Brandon Amos
# 2015/10/11

import time

start = time.time()

#import argparse
#import cv2
import os
import pickle
import sys

#from operator import itemgetter

import numpy as np
np.set_printoptions(precision=2)
import pandas as pd

#import openface

#from sklearn.pipeline import Pipeline
#from sklearn.lda import LDA
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from sklearn.grid_search import GridSearchCV
from sklearn.mixture import GMM
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split,  KFold
#fileDir = os.path.dirname(os.path.realpath(__file__))
#modelDir = os.path.join(fileDir, '..', 'models')
#dlibModelDir = os.path.join(modelDir, 'dlib')
#openfaceModelDir = os.path.join(modelDir, 'openface')


def train(classfier, data, labelsNum, nClasses,):
    print("Loading embeddings.")
    fname = "{}/labels.csv".format(workDir)
    labels = data[:,0]
    embeddings = data[:,1:]
    labelsNum = labels.tolist()
    print("Training for {} classes.".format(nClasses))
    if classifier == 'LinearSvm':
        clf = SVC(C=1, kernel='linear', probability=True)
    elif classifier == 'GridSearchSvm':
        print("""
        Warning: In our experiences, using a grid search over SVM hyper-parameters only
        gives marginally better performance than a linear SVM with C=1 and
        is not worth the extra computations of performing a grid search.
        """)
        param_grid = [
            {'C': [1, 10, 100, 1000],
             'kernel': ['linear']},
            {'C': [1, 10, 100, 1000],
             'gamma': [0.001, 0.0001],
             'kernel': ['rbf']}
        ]
        clf = GridSearchCV(SVC(C=1, probability=True), param_grid, cv=5)
    # ref:
    # http://scikit-learn.org/stable/auto_examples/classification/plot_classifier_comparison.html#example-classification-plot-classifier-comparison-py
    elif classifier == 'DecisionTree':  # Doesn't work best
        clf = DecisionTreeClassifier(max_depth=20)


    clf.fit(embeddings, labelsNum)

    fName = "{}/classifier.pkl".format(workDir)
    print("Saving classifier to '{}'".format(fName))
    with open(fName, 'w') as f:
        pickle.dump((le, clf), f)

"""
"""
def infer(X,Y,multiple=False,verbose=True):
    classifierModel = "{}/classifier.pkl".format(workDir)
    with open(classifierModel, 'rb') as f:
        if sys.version_info[0] < 3:
                (le, clf) = pickle.load(f)
        else:
                (le, clf) = pickle.load(f, encoding='latin1')


    # TODO Store testing represenations in folder 
    f_x = clf.predict(X)
    error = np.sum(Y[:,0] != f_x) / float(len(Y))

    print "\tTesting error is {}".format(error)
    return error
    # reps = getRep(img, multiple)
    # if len(reps) > 1:
    #     print("List of faces in image from left to right")
    # for r in reps:
    #     rep = r[1].reshape(1, -1)
    #     bbx = r[0]

    #     start = time.time()
    #     predictions = clf.predict_proba(rep).ravel()
    #     maxI = np.argmax(predictions)
    #     person = le.inverse_transform(maxI)
    #     confidence = predictions[maxI]

    #     if verbose:
    #         print("Prediction took {} seconds.".format(time.time() - start))
    #     if multiple:
    #         print("Predict {} @ x={} with {:.2f} confidence.".format(person.decode('utf-8'), bbx,
    #                                                                  confidence))
    #     else:
    #         print("Predict {} with {:.2f} confidence.".format(person.decode('utf-8'), confidence))

    #     # match prediction with label
    #     # Sum up 

if __name__ == '__main__':
    workDir = "./training-embeddings"
    print("Loading embeddings.")
    fname = "{}/labels.csv".format(workDir)
    labels = pd.read_csv(fname, header=None).as_matrix()[:, 0:1]
    fname = "{}/reps.csv".format(workDir)
    embeddings = pd.read_csv(fname, header=None).as_matrix()
    le = LabelEncoder().fit(labels)
    labelsNum = le.transform(labels)
    nClasses = len(le.classes_)

    print embeddings.shape
    print labels.shape, embeddings.shape
    data = np.append(labels,embeddings,axis=1)
    Y = data[:,0:1]
    X = data[:,1:]

    # Split dataset
    # Train on generated embeddings
    splits = [.20,.40,.50,.60,.80]
    for split in splits:
        print "----------------------------------------------------"
        print "[{},{}] [Train,Test] Split".format(  int(100-(split*100)),\
                                                int((split*100)))
        X_train, X_test, Y_train, Y_test = train_test_split(X,Y,
                                      test_size=split,random_state=42)
        classifier = "LinearSvm"
        data = np.append(Y_train,X_train,axis=1)
        train(classifier,data,labelsNum,nClasses)

        test_error = infer(X_test,Y_test)
        print "----------------------------------------------------"



