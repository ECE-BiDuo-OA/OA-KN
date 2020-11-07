import matplotlib.pyplot as plt
import numpy as np
from numpy import genfromtxt

np.set_printoptions(suppress=True)

#Getting data
print("Getting Data...", end="")
my_data = genfromtxt('data.csv', delimiter=',')

t=my_data.T[0]
z=list(my_data.T[1])
print("Done")

##Plot
"""
print("\nQ1 Printing plot...", end="")
plt.plot(t,z)
plt.show()
print("Done")
"""

##Variables
K=3 #number of test set
N=150 #number of features
J=1 #size of a prediction

print("\nQ2 Generating sets...", end="")
X=[]
Y=[]

for i in range(len(z) - N - (J-1)):
    X.append(z[i:i+N])
    Y.append(z[i+N:i+N+J])


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
def BGD():
    theta = np.random.rand(N+1)*2-1
    alpha=0.2

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
        #print(E)

    return theta

##SGD
def SGD():
    theta = np.random.rand(N+1)*2-1
    alpha=0.2

    E=11
    while E>10:
        h=np.dot(theta,Xtrain.T)
        g=h-Ytrain

        for n in range(N+1):
            i = np.random.randint(1, I)
            theta[n] = theta[n] - alpha * g[i] * Xtrain[i][n]

        h=np.dot(theta,Xtrain.T)
        E=np.sum(np.square(h-Ytrain))/2
        #print(E)

    return theta

##CFS
def CFS():
    theta = np.dot(np.dot(np.linalg.inv(np.dot(Xtrain.T, Xtrain)), Xtrain.T), Ytrain)

    return theta

def computeError(Yp):
    E=np.sum(np.square(Yp.T-Ytrain))/2
    return E

"""
##Q3
print("\nQ3 BGD, computing... ",end="")
thetaOptBGD=BGD()
YpBGD=np.dot(thetaOptBGD.T,Xtrain.T)
print("Done")

E=computeError(YpBGD)
print("\nError ",E)

print("\nOptimal value for the parameters:")
print(thetaOptBGD)

##Q4
print("\nQ4 SGD, computing... ",end="")
thetaOptSGD=SGD()
YpSGD=np.dot(thetaOptSGD.T,Xtrain.T)
print("Done")

E=computeError(YpSGD)
print("\nError ",E)

print("\nOptimal value for the parameters:")
print(thetaOptSGD)
"""
##Q5
print("\nQ5 CFS, computing... ",end="")
thetaOptCFS=CFS()
YpCFS=np.dot(thetaOptCFS.T,Xtrain.T)
print("Done")

E=computeError(YpCFS)
print("\nError ",E)

print("\nOptimal value for the parameters:")
print(thetaOptCFS)

##Plot Q6
print("\nQ6 Plotting prediction on the original dataset ...", end="")
plt.plot(t[:I],Ytrain.T[0],"g")
#plt.plot(t[:I],YpBGD,"y")
#plt.plot(t[:I],YpSGD,"b")
plt.plot(t[:I],YpCFS[0],"r")
plt.show()
print("Done")

##Plot Q7
print("\nQ7 Plotting prediction on the test dataset ...", end="")

#YpBGD=np.dot(thetaOptBGD.T,Xtest.T)
#YpSGD=np.dot(thetaOptSGD.T,Xtest.T)
YpCFS=np.dot(thetaOptCFS.T,Xtest.T)

plt.plot(t[-J:],Ytest[0],"g")
#plt.plot(t[-K:],YpBGD,"y")
#plt.plot(t[-K:],YpSGD,"b")
for i in range(K-1):
    plt.plot(t[-J-K+i+1:-K+i+1],YpCFS.T[i],"r")
plt.plot(t[-J:],YpCFS.T[K-1],"r")

plt.show()
print("Done")







