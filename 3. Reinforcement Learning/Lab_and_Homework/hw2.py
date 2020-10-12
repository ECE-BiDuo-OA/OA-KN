"""
Made by the team :
    Kishor JOGARAJAH
    Neil SEGARD
"""

import numpy as np

def chooseDirection():
    n = np.random.rand()
    if n < 0.8:#straight
        return 0
    elif 0.8 <= n and n < 0.9:#left
        return 1
    elif 0.9 <= n:#right
        return 2
        
#Parameters
gamma = 0.99

#Creating the map
states = np.ones((3, 4))
states = states * (-0.02)
states[0,3] = +1
states[1,3] = -1
states[1,1] = -10

print(states)

#Making policy (where to go for each state)
#Actions : N=0, S=1, E=2, W=3
policy = np.random.randint(0,4,(3, 4))#random juste pour les tests faudra calculer

print(policy)


##Value iteration
values=np.zeros((3, 4))

for i in range(10):#a changer en "jusqu'a convergence" avec un while
    value=1#a suivre... mdr








