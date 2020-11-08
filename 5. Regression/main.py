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
thetaOptBGD=prog.trainBGD(alpha=0.001, showError=True)
YpBGD, err = prog.predict(thetaOptBGD)
print("\nError ",err)

print("\nOptimal value for the parameters:")
print(thetaOptBGD)

exit(0)
##Q4
print("\nQ4 SGD")
thetaOptSGD=prog.trainSGD(alpha=0.001, showError=True)
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
YpBGD=YpCFS[0] ###############A SUPPRIMER
YpSGD=YpCFS[0] ###############A SUPPRIMER
#prog.plotQ6(YpBGD, YpSGD, YpCFS)
print("Done")

##Plot Q7
print("\nQ7 Plotting prediction on the test dataset ...", end="")

prog30 = regression(3, 150, 30)#K N J
thetaOptCFS=prog30.trainCFS()
YpCFS, err = prog30.predict(thetaOptCFS)
print("\nError ",err)

print("\nOptimal value for the parameters:")
print(thetaOptCFS)

prog30.plotQ7(YpCFS)

print("Done")







