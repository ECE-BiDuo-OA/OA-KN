import matplotlib.pyplot as plt
import numpy as np
from numpy import genfromtxt

np.set_printoptions(suppress=True)

print("Getting Data...", end="")
my_data = genfromtxt('data.csv', delimiter=',')

t=my_data.T[0]
z=my_data.T[1]
print("Done")


print("Printing plot...", end="")
plt.plot(t,z)
plt.show()
print("Done")