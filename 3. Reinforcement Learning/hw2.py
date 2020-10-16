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
R = np.ones((3, 4))
R = R * (-0.02)
R[0,3] = +1
R[1,3] = -1
R[1,1] = -10

print(R)
print()

#Making policy (where to go for each state)
policy = np.random.randint(0,4,(3, 4))#random juste pour les tests faudra calculer

def newState(originRow, originCol, action):
    print(action)
    if action == 0: #North
        row = originRow + 1
    elif action == 1: #South
        row = originRow - 1
    elif action == 2: #East
        col = originCol + 1
    elif action == 3: #West
        col = originCol - 1
    else:
        col = originCol
        row = originRow

    if row > 2 : row = 2
    if row < 0 : row = 0
    if col > 3 : col = 3
    if col < 0 : col = 0

    return row, col

def possibleStates(row, col):
    list=[]

    for action in range(4):
        list.append(newState(row, col, action))

    return list

##Value iteration
values=np.zeros((3, 4))

for col in range(4):
    for row in range(3):
        r = R[row][col]

        #pour chaque action
        #Actions : N=0, S=1, E=2, W=3
        pList=[]
        for action in range(4):
            caseNorth, caseSouth, caseEast, caseWest = possibleStates(row, col)

            if action == 0: #North
                p = 0.8 * value[caseNorth[0],caseNorth[1]] \
                  + 0.1 * value[caseEast[0],caseEast[1]] \
                  + 0.1 * value[caseWest[0],caseWest[1]]
            elif action == 1: #Sud
                p = 0.8 * value[caseSouth[0],caseSouth[1]] \
                  + 0.1 * value[caseEast[0],caseEast[1]] \
                  + 0.1 * value[caseWest[0],caseWest[1]]
            elif action == 2: #East
                p = 0.8 * value[caseEast[0],caseEast[1]] \
                  + 0.1 * value[caseNorth[0],caseNorth[1]] \
                  + 0.1 * value[caseSouth[0],caseSouth[1]]
            elif action == 3: #West
                p = 0.8 * value[caseWest[0],caseWest[1]] \
                  + 0.1 * value[caseNorth[0],caseNorth[1]] \
                  + 0.1 * value[caseSouth[0],caseSouth[1]]

            pList.append(p)

        max = np.max(pList)

        value[row][col] = r + gamma * max











