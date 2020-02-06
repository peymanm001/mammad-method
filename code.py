from random import randint
from random import uniform
import numpy as np
import pandas as pd
from sklearn import preprocessing
from cvxopt import matrix as cvxopt_matrix
from cvxopt import solvers as cvxopt_solvers
from sklearn import metrics
import matplotlib.pyplot as plt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as lda
from sklearn.svm import SVC

def cvxopt_solve(X,y,mems):
    C = 10
    m,n = X.shape
    y = y.reshape(-1,1) * 1.
    X_dash = y * X
    H = np.dot(X_dash , X_dash.T) * 1.
    
    #Converting into cvxopt format - as previously
    P = cvxopt_matrix(H)
    q = cvxopt_matrix(-np.ones((m, 1)))
    G = cvxopt_matrix(np.vstack((np.eye(m)*-1,np.eye(m))))
    h = cvxopt_matrix(np.hstack((np.zeros(m), mems * C)))
    A = cvxopt_matrix(y.reshape(1, -1))
    b = cvxopt_matrix(np.zeros(1))
    
    #Run solver
    sol = cvxopt_solvers.qp(P, q, G, h, A, b)
    alphas = np.array(sol['x'])
    #==================Computing and printing parameters===============================#
    w = ((y * alphas).T @ X).reshape(-1,1)
    S = (((C * mems) > alphas.T) > 1e-4).flatten()
    b = y[S] - np.dot(X[S], w)
    b = np.mean(b)
    
    return (w.flatten(),b)
def readDataset(name):
    if name == 'heart.data':
        dataframe = pd.read_csv(name,header=-1,delim_whitespace=True)
    else:
        dataframe = pd.read_csv(name, delimiter=',', header=-1)
    return dataframe

def testLeaveOneOut(original_features,original_labels,original_mems):
    correct_predicted0 = 0
    overall0 = 0
    for test_index in range(len(original_features)):
        test_pattern = original_features[test_index]
        test_label = original_labels[test_index]
        test_mem = original_mems[test_index]
        features = np.delete(original_features, [test_index], axis=0)
        labels = np.delete(original_labels, [test_index], axis=0)
        mems = np.delete(original_mems, [test_index], axis=0)
        (w1,b1) = cvxopt_solve(features,labels,mems)
        y = np.sign(np.dot(np.transpose(w1),test_pattern)+b1)
        if y == test_label:
            correct_predicted0+=1
        overall0 +=1
    
    return correct_predicted0/overall0

def lda_(values,class1,class2,labels,features):
    
    
    #use fisher for finding best line
    clf = lda()
    clf.fit(features,labels)
    w_ = clf.coef_
    b_ = clf.intercept_
    
    class1_projected = np.dot(class1,w_.T)+b_
    class2_projected = np.dot(class2,w_.T)+b_   
    class1_mean_projected = np.mean(class1_projected)
    class2_mean_projected = np.mean(class2_projected)

    class1_dist_to_mean = class1_projected - class1_mean_projected
    class2_dist_to_mean = class2_projected - class2_mean_projected
    
    means_dist = class2_mean_projected - class1_mean_projected
    
    mems1 = (means_dist - class1_dist_to_mean)/means_dist 
    mems1[mems1 > 1] = 1
    mems1[mems1 < 0] = 0
    mems2 = (means_dist + class2_dist_to_mean)/means_dist
    mems2[mems2 > 1] = 1
    mems2[mems2 < 0] = 0
    
    mems = np.concatenate((mems1,mems2))
    

    return testLeaveOneOut(features,labels,mems.ravel())
def ourFuzzy(values,class1,class2,labels,features):
    
    
    class1_mean = np.mean(class1,axis = 0)
    class2_mean = np.mean(class2,axis = 0)
    line = class2_mean - class1_mean
    line = line / np.linalg.norm(line)
    class1_mean_projected = np.dot(np.transpose(class1_mean),line)
    class2_mean_projected = np.dot(np.transpose(class2_mean),line)
    class1_projected = np.dot(class1,line)
    class2_projected = np.dot(class2,line)
    

    class1_dist_to_mean = class1_projected - class1_mean_projected
    class2_dist_to_mean = class2_projected - class2_mean_projected
    
    means_dist = class2_mean_projected - class1_mean_projected
    
    mems1 = (means_dist - class1_dist_to_mean)/means_dist 
    mems1[mems1 > 1] = 1
    mems1[mems1 < 0] = 0
    mems2 = (means_dist + class2_dist_to_mean)/means_dist
    mems2[mems2 > 1] = 1
    mems2[mems2 < 0] = 0
    
    mems = np.concatenate((mems1,mems2))
    

    return testLeaveOneOut(features,labels,mems.ravel())

def original(values,class1,class2,labels,features):
    
    mems = np.ones(len(values))

    return testLeaveOneOut(features,labels,mems)
    
if __name__ == "__main__":
    df = readDataset('ionosphere.data')
    values = df.values
    labels = values[:,-1]
    labels[labels==2]=-1
    features = values[:,:-1]
    class1 = values[values[:,-1] == 1]
    class2 = values[values[:,-1] == -1]
    class1 = class1[:,:-1]
    class2 = class2[:,:-1]



    acc = lda_(values,class1,class2,labels,features)
    lda_file = open('lda.txt',"w")
    lda_file.write("accuracy:"+str(acc))
    lda_file.close

    acc = original(values,class1,class2,labels,features)
    original_file = open('original.txt',"w")
    original_file.write("accuracy:"+str(acc))
    original_file.close


    acc = ourFuzzy(values,class1,class2,labels,features)
    fuzzy_file = open('our_fuzzy.txt',"w")
    fuzzy_file.write("accuracy:"+str(acc))
    fuzzy_file.close

    
    
    
    
    
