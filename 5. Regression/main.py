"""
Made by
    SEGARD Neil
    JOGARAJAH Kishor
"""

import matplotlib.pyplot as plt
import numpy as np
from numpy import genfromtxt
from lib import regression #importing our self made library

np.set_printoptions(suppress=True)

##Plot
"""
print("\nQ1 Printing plot...", end="")
plt.plot(t,z)
plt.show()
print("Done")
"""

print("\nQ2 Generating sets")
prog = regression(3, 150, 1)#K N J



##Q3
print("\nQ3 BGD")
thetaOptBGD=prog.trainBGD()
YpBGD, err = prog.predict(thetaOptBGD)
print("\nError ",err)

print("\nOptimal value for the parameters:")
print(thetaOptBGD)

##Q4
print("\nQ4 SGD")
thetaOptSGD=prog.trainSGD()
YpSGD, err = prog.predict(thetaOptSGD)
print("\nError ",err)

print("\nOptimal value for the parameters:")
print(thetaOptSGD)

##Q5
print("\nQ5 CFS")
thetaOptCFS=prog.trainCFS()
YpCFS, err = prog.predict(thetaOptCFS)
print("\nError ",err)

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







