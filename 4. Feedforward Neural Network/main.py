import numpy as np

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

    return X,Y,YUnique

def sigmoid(x):
    return 1/(1+np.exp(-x))

def fwp(X,V,W):
    # Forward Propagation
    I=len(X)

    #Computing Xb
    ones=np.ones((I, 1))
    Xb=np.concatenate((ones, X), axis=1)

    #Computing Xbb
    Xbb=np.dot(Xb,V)

    #Computing F
    F=np.apply_along_axis(sigmoid, 0, Xbb)

    #Computing Fb
    Fb=np.concatenate((ones, F), axis=1)

    #Computing Fbb
    Fbb=np.dot(Fb,W)

    #Computing Yp (Y predicted)
    Yp=np.apply_along_axis(sigmoid, 0, Fbb)

    return Yp,F,Fb,Xb

def error(Y,Yp):
    return np.sum(np.square(Yp - Y))

def bp(V,W,av,aw,Y,Yp,Fb,Xb,K):
    I=len(Y)
    N=len(X[0])
    J=len(Y[0])

    #BACK Propagation
    H=(Yp-Y)*Yp*(1-Yp)

    #Computing the new W
    for k in range(0,K+1):
        for j in range(0,J):
            dEdW=0
            for i in range(0,I):
                dEdW += H[i][j] *Fb[i][k]

            W[k][j] = W[k][j] - aw * dEdW


    #Computing the new V
    for n in range(0,N+1):
        for k in range(0,K):
            dEdV=0
            for i in range(0,I):
                for j in range(0,J):
                    dEdV += H[i][j] * W[k][j] * F[i][k] * (1 - F[i][k]) * Xb[i][n]

            V[n][k] = V[n][k] - aw * dEdV


    return V,W

############# SETTINGS #############
K=5
nbEpoch=500
av=0.06
aw=av

############# INITIALIZING #############
#Getting X and Y
X, Y, YUnique = getData("data_ffnn.txt")

N=len(X[0]) #number of neuron in the input layer
J=len(Y[0]) #number of neuron in the output layer

############# LEARNING #############
#Generating V and W randomly
V=np.random.uniform(-1,1,(N+1,K)) #We add 1 to N to have a bias
W=np.random.uniform(-1,1,(K+1,J))



for epoch in range(nbEpoch):
    # Forward Propagation
    Yp, F,Fb,Xb = fwp(X,V,W)

    #Computing Error
    E = error(Y,Yp)

    print(E)

    #Back Propagation
    V,W = bp(V,W,av,aw,Y,Yp,Fb,Xb,K)




XTest=[[ 3.9601,  1.4057,  0.4019]]

R,_,_,_=fwp(XTest,V,W)

R=np.apply_along_axis(np.round, 0, R)

for i in range(len(XTest)):
    rCateg=R[i]
    r=sum(YUnique*rCateg)
    print(XTest[i], " \t", rCateg, " \t", r)











