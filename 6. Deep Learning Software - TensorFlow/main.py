#!pip install tensorflow==1.15.4
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

##Q1 Plotting data
#plt.plot(t,z)
#plt.show()

##Q2 Splitting the data
K = 10
N = 150
J = 1
training_epochs=10
learning_rate=0.00001

X = []
Y = []

for i in range(len(z) - N - (J-1)):
    X.append(z[i: i + N])
    Y.append(z[i + N: i + N + J])


I = len(X) - K #number of training set

Xtrain=np.asarray(X[:I])
Ytrain=np.asarray(Y[:I]).reshape(1097)

Xtest=np.asarray(X[I:])
Ytest=np.asarray(Y[I:]).reshape(10)

##Preparing the truc
# Sets up the input and output nodes as placeholders
X = tf.compat.v1.placeholder(tf.float32, shape=[N], name='placeholder_X')
Y = tf.compat.v1.placeholder(tf.float32, name='placeholder_Y')

# Defines the model as y = theta*X
def model(X, theta):
    return tf.multiply(X, theta)

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
for epoch in range(training_epochs):
    for (x, y) in zip(Xtrain, Ytrain):
        # Updates the model parameter(s)
        sess.run(train_op, feed_dict={X: x, Y: y})

    print(sess.run(theta)[0])

# Obtains the final parameter value
theta_val = sess.run(theta)

# Closes the session
#

#print(theta_val)

##Plotting the regression
"""
tensorList=[]
for x in Xtrain:
    tensorList.append(tf.convert_to_tensor(x, dtype=tf.float32))
y_pred = sess.run(tensorList)
"""
#y_pred = sess.run(tf.convert_to_tensor(Xtrain))

X2 = tf.compat.v1.placeholder(tf.float32, shape=(1097,N), name='placeholder_X2')
X3 = tf.compat.v1.placeholder(tf.float32, shape=(1097), name='placeholder_X3')

y_pred = tf.compat.v1.placeholder(tf.float32, name="placeholder_y_pred")
y_pred2 = tf.compat.v1.placeholder(tf.float32,shape=(1097), name="placeholder_y_pred2")

try: 1#sess.run(y_pred, feed_dict={X:Xtrain[0]})
except ValueError as e: print(e)

try: 1#sess.run(y_pred, feed_dict={X2:Xtrain})
except ValueError as e: print(e)

try: 1#sess.run(y_pred2, feed_dict={X2:Xtrain})
except ValueError as e: print(e)

try: sess.run(y_pred, feed_dict={X3:Xtrain})
except ValueError as e: print(e)

try: sess.run(y_pred2, feed_dict={X3:Xtrain})
except ValueError as e: print(e)

softmax_tensor = sess.graph.get_tensor_by_name('final_ops/softmax:0')

     predictions = self.sess.run(softmax_tensor, {'Placeholder:0': self.tfimages})


"""
plt.plot(t,z)
plt.plot(t[-I:],y_pred)
plt.show()
"""
#sess.close()








