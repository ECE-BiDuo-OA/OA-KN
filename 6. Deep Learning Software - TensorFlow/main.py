"""
Made by
    SEGARD Neil
    JOGARAJAH Kishor
"""

import matplotlib.pyplot as plt
import numpy as np
from numpy import genfromtxt

np.set_printoptions(suppress=True)

##Getting data
my_data = genfromtxt('data.csv', delimiter=',',names=True)

t=my_data["t"]
z=my_data["z"]

##Q1 Plotting data
#plt.plot(t,z)
#plt.show()

##Q2 Splitting the data
K = 10
N = 150
J = 1

X = []
Y = []

for i in range(len(z) - N - (J-1)):
    X.append(z[i: i + N])
    Y.append(z[i + N: i + N + J])


I = len(X) - K #number of training set

Xtrain=np.asarray(X[:I])
Ytrain=np.asarray(Y[:I])

Xtest=np.asarray(X[I:])
Ytest=np.asarray(Y[I:])