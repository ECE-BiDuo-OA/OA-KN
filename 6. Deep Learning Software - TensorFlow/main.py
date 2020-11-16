"""
Made by
    SEGARD Neil
    JOGARAJAH Kishor
"""


import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from numpy import genfromtxt
import os
import math

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
np.set_printoptions(suppress=True)

print("Currently using tensorflow {}".format(tf.__version__))
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

##Getting data
my_data = genfromtxt('https://raw.githubusercontent.com/ECE-BiDuo-OA/OA-KN/main/6.%20Deep%20Learning%20Software%20-%20TensorFlow/data.csv', delimiter=',',names=True)

t=my_data["t"]
z=my_data["z"]


##Getting data Microsoft
"""
my_data = genfromtxt('https://raw.githubusercontent.com/ECE-BiDuo-OA/OA-KN/main/6.%20Deep%20Learning%20Software%20-%20TensorFlow/microsoft.csv', delimiter=',',names=True)

z=my_data["Close"]
t=list(range(len(z)))
"""

##Q1 Plotting data
plt.plot(t,z)
plt.show()

##Q2 Splitting the data
K = 200
N = 150
J = 1
training_epochs=500
learning_rate=0.00000005#0.000001

X = []
Y = []

for i in range(len(z) - N - (J-1)):
    X.append(z[i: i + N])
    Y.append(z[i + N: i + N + J])

I = len(X) - K #number of training set

Xtrain=np.asarray(X[:I])
Ytrain=np.asarray(Y[:I]).reshape(I)

Xtest=np.asarray(X[I:])
Ytest=np.asarray(Y[I:]).reshape(K)

##Q3 Preparation
# Sets up the input and output nodes as placeholders
X = tf.compat.v1.placeholder(tf.float32, shape=[N], name='placeholder_X')
Y = tf.compat.v1.placeholder(tf.float32, name='placeholder_Y')

# Defines the model as y = theta*X
def model(X, theta):
    return tf.reduce_sum(tf.multiply(X, theta))

def error(pred,real):
    return np.sum(np.square(pred-real))

# Sets up the weights variable
theta = tf.Variable([0.0]*150, name="weights")

# Defines the cost function
y_model = model(X, theta)
cost = tf.square(Y-y_model)

# Defines the operation that will be called on each iteration of the learning algorithm
train_op = tf.compat.v1.train.GradientDescentOptimizer(learning_rate).minimize(cost)

sess = tf.compat.v1.Session()
init = tf.compat.v1.global_variables_initializer()
sess.run(init)

##Training
print("\nTraining...")
for epoch in range(training_epochs):
    for (x, y) in zip(Xtrain, Ytrain):
        # Updates the model parameter(s)
        sess.run(train_op, feed_dict={X: x, Y: y})

    if (epoch+1)%20==0:
        theta_val = sess.run(theta)
        y_pred = np.dot(Xtrain,theta_val)
        print("Epoch {:3d}/{}: Error = {:.3f}".format(epoch+1,training_epochs,error(y_pred,Ytrain)))

# Obtains the final parameter value
theta_val = sess.run(theta)
print("\nOptimal theta:")
print(theta_val)

sess.close()

##Q4 Plotting the regression on the original dataset
y_pred = np.dot(Xtrain,theta_val)

plt.plot(t[:I],Ytrain)
plt.plot(t[:I],y_pred, "r")
plt.show()

##Q5 Plotting the regression on the test dataset
y_pred = np.dot(Xtest,theta_val)
print("\nError on the test set = {}".format(error(y_pred,Ytest)))

plt.plot(t[N+I:],Ytest)
plt.plot(t[N+I:],y_pred, "r")
plt.show()

##Q6 Plotting the regression on the test dataset
print("\n\nREPORT\nCFS is more accurate and faster than tensorflow.\nTensorflow is like to be more powerful than CFS but in our case (small problem) CFS is better.")


