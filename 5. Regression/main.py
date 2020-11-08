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

print("\nQ2 Generating sets")
prog = regression(3, 150, 1)#K N J

##Q3
print("\nQ3 BGD")
thetaOptBGD=prog.trainBGD(alpha=0.00001, showError=True, targetError=40)
YpBGD, err = prog.predict(thetaOptBGD)
print("\nError ",err)

print("\nOptimal value for the parameters:")
print(thetaOptBGD)

##Q4
print("\nQ4 SGD")
thetaOptSGD=prog.trainSGD(alpha=0.0001, showError=True, targetError=40)
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
print(thetaOptCFS.T)

##Plot Q6
print("\nQ6 Plotting prediction on the original dataset ...", end="")
prog.plotQ6(YpBGD, YpSGD, YpCFS[0])
print("Done")

##Plot Q7
print("\nQ7 Plotting prediction on the test dataset ...", end="")

prog30 = regression(3, 150, 30)#K N J
thetaOptCFS30=prog30.trainCFS()
YpCFS30, err = prog30.predict(thetaOptCFS30)
print("\nError ",err)

print("\nOptimal value for the parameters:")
print(thetaOptCFS30)

prog30.plotQ7(YpCFS30)

print("Done")



## COMMENTS
print("\n\nREPORT\n\nFor the homework, we've created our self made library, you can see it in the lib.py file.\nWe can adjust parameters when calling functions from this library.\nYou can hide the error when training by setting showError to False.\n\nBGD & SGD are quite similar, but the SGD is faster and more efficient than the BGD\nThe CFS method is much more precise and faster to compute.\n\nWith J=1, the prediction is better than J=30.")





