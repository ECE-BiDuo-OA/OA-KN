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

        self.t=my_data.T[0]
        z=list(my_data.T[1])

        self.zMax=np.max(z)
        z /= self.zMax

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
        E=np.sum(np.square(Yp-self.Ytrain.T[0]))/2
        return E

    def trainBGD(self, targetError=10, alpha=0.001, showError=False):
        theta = np.random.rand(self.N+1)*2-1

        E=targetError+1
        while E>targetError:
            for n in range(self.N+1):
                h=np.dot(theta,self.Xtrain.T)
                g=h-self.Ytrain.T[0]

                sum=0
                for i in range(self.I):
                    sum += g[i] * self.Xtrain[i][n]

                theta[n] = theta[n] - alpha * sum

            h = np.dot(theta, self.Xtrain.T)
            E = self.error(h)
            if showError: print("Error:",E)

        return theta

    def trainSGD(self, targetError=10, alpha=0.2, showError=False):
        theta = np.random.rand(self.N+1)*2-1

        E=targetError+1
        while E>targetError:
            for n in range(self.N+1):
                h=np.dot(theta,self.Xtrain.T)
                g=h-self.Ytrain.T[0]

                i = 4#np.random.randint(1, self.I)
                theta[n] = theta[n] - alpha * g[i] * self.Xtrain[i][n]

            h=np.dot(theta,self.Xtrain.T)
            E=self.error(h)
            if showError: print("Error:",E)

        return theta

    def trainCFS(self):
        theta = np.dot(np.dot(np.linalg.inv(np.dot(self.Xtrain.T, self.Xtrain)), self.Xtrain.T), self.Ytrain)

        return theta

    def predict(self,theta):
        Yp=np.dot(theta.T,self.Xtrain.T)
        return Yp, self.error(Yp)

    def plotQ6(self, YpBGD, YpSGD, YpCFS):
        t = self.t[:self.I]
        Y = self.Ytrain.T[0]
        plt.plot(t,Y,"g")
        plt.plot(t,YpBGD,"y", label="BGD")
        plt.plot(t,YpSGD,"b", label="SGD")
        plt.plot(t,YpCFS,"r", label="CFS")
        plt.legend()
        plt.show()

    def plotQ7(self, YpCFS):
        plt.plot(self.t[-self.J:],self.Ytest[0],"g")
        for i in range(self.K-1):
            a = -self.K+i+1
            start=-self.J + a

            plt.plot(self.t[start:a], YpCFS.T[i], "r")
        plt.plot(self.t[-self.J:], YpCFS.T[self.K-1], "r")

        plt.show()
