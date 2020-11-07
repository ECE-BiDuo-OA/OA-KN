"""
Self made library by
    SEGARD Neil
    JOGARAJAH Kishor
"""

import matplotlib.pyplot as plt
import numpy as np
from numpy import genfromtxt

np.set_printoptions(suppress=True)

class regression():
    def __init__(self, K, N, J): #creation des Xtrain, Xtest, Ytrain, Ytest

        my_data = genfromtxt('data.csv', delimiter=',')

        t=my_data.T[0]
        z=list(my_data.T[1])

        self.K=K #number of test set
        self.N=N #number of features
        self.J=J #size of a prediction

        X=[]
        Y=[]

        for i in range(len(z) - self.N - (self.J-1)):
            X.append(z[i: i + self.N])
            Y.append(z[i + self.N: i + self.N + self.J])


        self.I = len(X) - self.K #number of training set

        self.Xtrain=np.asarray(X[:self.I])
        self.Ytrain=np.asarray(Y[:self.I])

        self.Xtest=np.asarray(X[self.I:])
        self.Ytest=np.asarray(Y[self.I:])

        #add a column of ones for the bias
        ones=np.ones((self.I, 1))
        self.Xtrain=np.concatenate((ones, self.Xtrain), axis=1)

        ones=np.ones((self.K, 1))
        self.Xtest=np.concatenate((ones, self.Xtest), axis=1)

    def error(self, Yp):
        E=np.sum(np.square(Yp.T-self.Ytrain))/2
        return E

    def trainBGD(self, alpha=0.2):
        theta = np.random.rand(self.N+1)*2-1

        E=11
        while E>10:
            h=np.dot(theta,self.Xtrain.T)
            g=h-self.Ytrain

            for n in range(self.N+1):
                sum=0
                for i in range(self.I):
                    sum += g[i] * self.Xtrain[i][n]

                theta[n] = theta[n] - alpha * sum

            h=np.dot(theta, self.Xtrain.T)
            E= self.error(h)

        return theta

    def trainSGD(self, alpha=0.2):
        theta = np.random.rand(self.N+1)*2-1

        E=11
        while E>10:
            h=np.dot(theta,self.Xtrain.T)
            g=h-self.Ytrain

            for n in range(self.N+1):
                i = np.random.randint(1, self.I)
                theta[n] = theta[n] - alpha * g[i] * self.Xtrain[i][n]

            h=np.dot(theta,self.Xtrain.T)
            E=self.error(h)

        return theta

    def trainCFS(self):
        theta = np.dot(np.dot(np.linalg.inv(np.dot(self.Xtrain.T, self.Xtrain)), self.Xtrain.T), self.Ytrain)

        return theta

    def predict(self,theta):
        Yp=np.dot(theta.T,self.Xtrain.T)
        return Yp, self.error(Yp)
