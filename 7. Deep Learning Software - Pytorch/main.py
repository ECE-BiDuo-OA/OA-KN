"""
Made by the team :
    Kishor JOGARAJAH
    Neil SEGARD
"""

import numpy as np

import torch
import matplotlib.pyplot as plt
from torch.autograd import Variable
import torch.nn.functional as F
#%matplotlib inline

def getData(file):
    data = np.loadtxt(fname = file)
    X=data[:,:3]
    YData=data[:,3:]

    #Formatting Y
    Y=[]
    YUnique=np.unique(YData)

    for y in YData:
        Y.append((YUnique==y[0])*1)

    Y=np.asarray(Y)

    return X, Y, YUnique, YData

############# INITIALIZING #############
#Getting X and Y
X, Y, YUnique, Ydata = getData("https://raw.githubusercontent.com/ECE-BiDuo-OA/OA-KN/main/7.%20Deep%20Learning%20Software%20-%20Pytorch/data_ffnn_3classes.txt")
K=10
learning_rate=0.02
nbEpoch=500

############# Q1 #############
print("Q1")
print("We have {} different categories\n".format(len(YUnique)))


############# Q2&3 #############
print("\nQ2&3")

torch.manual_seed(1) # reproducible
# sample data preparation
Ydata=Ydata.reshape(len(Ydata))
x = torch.tensor(X).type(torch.FloatTensor)
y = torch.tensor(Ydata).type(torch.LongTensor)

# torch need to train on Variable, so convert sample features to Variable
x, y = Variable(x), Variable(y)

#plt.scatter(x.data.numpy()[:,0], x.data.numpy()[:,1], c=y.data.numpy(), s=100)
#plt.show()


class Net(torch.nn.Module):
    def __init__(self, n_feature, n_hidden, n_output):
        super(Net, self).__init__()
        self.hidden = torch.nn.Linear(n_feature, n_hidden) # hidden layer
        self.out = torch.nn.Linear(n_hidden, n_output) # output layer

    def forward(self, x):
        x = torch.sigmoid(self.hidden(x)) # activation function for hidden layer
        # x = F.sigmoid(self.hidden(x)) # activation function for hidden layer
        x = self.out(x)
        return x

net = Net(n_feature=3, n_hidden=K, n_output=3) # define the network
# net.double()
#print(net) # Neural network architecture

# Loss and optimizer
# Softmax is internally computed.
# Set parameters to be updated
optimizer = torch.optim.SGD(net.parameters(), lr=learning_rate)
loss_func = torch.nn.CrossEntropyLoss() # the target label is not an one-hotted.
#print(optimizer)
#print(loss_func)


# turn the interactive mode on
plt.ion()



for t in range(nbEpoch):
    out = net(x) # input x and predict based on x
    loss = loss_func(out, y) # must be (1. nn output, 2. target)
    optimizer.zero_grad() # clear gradients for next train
    loss.backward() # backpropagation, compute gradients
    optimizer.step() # apply gradients

    if t % 10 == 0:
        # show learning process
        _, prediction = torch.max(F.softmax(out),1)
        pred_y = prediction.data.numpy().squeeze()
        target_y = y.data.numpy()
        error = sum(pred_y != target_y)
        accuracy = 1 - error/len(X)

        print("Epoch={:3d}, Error={:2d}, Accuracy={:.2f} %".format(t, error, accuracy*100))

        if accuracy==1.0:
          break

plt.ioff()


############# Q4 #############
print("\nQ4")
out = net(x) # input x and predict based on x
_, prediction = torch.max(F.softmax(out),1)
pred_y = prediction.data.numpy().squeeze()
target_y = y.data.numpy()

for p,t in zip(pred_y,target_y):
  print("Target: {}    Predicted: {}    {}".format(t, p, "MATCH" if p==t else "NOT MATCH"))


############# Q5 #############
print("\nQ5")
XTest=[[2, 2, -3],[3, 4, 3],[4.5, 1.5, 0]]
xtext = torch.tensor(XTest).type(torch.FloatTensor)
xtext = Variable(xtext)

out = net(xtext) # input x and predict based on x
_, prediction = torch.max(F.softmax(out),1)
pred_y = prediction.data.numpy().squeeze()

print(" X1   X2   X3     Y (predicted)")
for xt, yp in zip(XTest, pred_y):
    print("{:< 5.1f}{:< 5.1f}{:< 5.1f}   {:d}".format(xt[0],xt[1],xt[2],yp))