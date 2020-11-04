import matplotlib.pyplot as plt
import numpy as np
from numpy import genfromtxt

np.set_printoptions(suppress=True)

#Getting data
print("Getting Data...", end="")
my_data = genfromtxt('data.csv', delimiter=',')

t=my_data.T[0]
z=my_data.T[1]
z=np.round(z,2)#A SUPPRIMER
print("Done")

##Plot
print("Q1 Printing plot...", end="")
plt.plot(t,z)
#plt.show()
print("Done")


##Variables
K=3 #number of test set
N=150 #number of features
J=1 #size of a prediction

print("Q2 Generating sets...", end="")
X=[]
Y=[]

for i in range(len(z) - N):
    X.append(z[i:i+N])
    Y.append(z[i+N])


I=len(X) - K #number of training set

Xtrain=np.asarray(X[:I])
Ytrain=np.asarray(Y[:I])

Xtest=np.asarray(X[I:])
Ytest=np.asarray(Y[I:])

#add a column of ones for the bias
ones=np.ones((I, 1))
Xtrain=np.concatenate((ones, Xtrain), axis=1)

ones=np.ones((K, 1))
Xtest=np.concatenate((ones, Xtest), axis=1)

print("Done")


##BGD
theta = np.random.rand(N+1)*2-1
#theta = np.random.randint(-5,5,N+1) c'est pire
alpha=0.0001

E=11
while E>10:
    h=np.dot(theta,Xtrain.T)
    g=h-Ytrain

    for n in range(N+1):
        sum=0
        for i in range(I):
            sum += g[i]*Xtrain[i][n]

        theta[n] = theta[n] - alpha * sum

    h=np.dot(theta,Xtrain.T)
    E=np.sum(np.square(h-Ytrain))/2
    print(E)

print(E)











