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
plt.plot(t,z)
plt.show()

