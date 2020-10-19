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


def newState(originRow, originCol, action):
    col = originCol
    row = originRow

    if action == 0: #North
        row = originRow - 1
    elif action == 1: #South
        row = originRow + 1
    elif action == 2: #East
        col = originCol + 1
    elif action == 3: #West
        col = originCol - 1

    if row > 2 : row = 2
    if row < 0 : row = 0
    if col > 3 : col = 3
    if col < 0 : col = 0

    if row ==1 and col == 1:
        col = originCol
        row = originRow

    return row, col

def possibleStates(row, col):
    list=[]

    #Actions : N=0, S=1, E=2, W=3
    for action in range(4):
        list.append(newState(row, col, action))

    return list

##Value iteration
print("##### Value iteration #####")
values=np.zeros((3, 4))
fixedCells=[(0,3),(1,3),(1,1)]
values[0,3]=+1
values[1,3]=-1
values[1,1]=-10
valuesOld=np.ones((3, 4))

#Finding optimal Value function
while np.sum(abs(valuesOld-values)>0.00001) != 0 :
    valuesOld=np.copy(values)
    for col in range(4):
        for row in range(3):
            if (row,col) not in fixedCells:
                r = R[row][col]

                caseNorth, caseSouth, caseEast, caseWest = possibleStates(row, col)

                pList=[]
                #North
                pList.append(0.8 * values[caseNorth[0],caseNorth[1]] \
                           + 0.1 * values[caseEast[0],caseEast[1]] \
                           + 0.1 * values[caseWest[0],caseWest[1]])

                #Sud
                pList.append(0.8 * values[caseSouth[0],caseSouth[1]] \
                           + 0.1 * values[caseEast[0],caseEast[1]] \
                           + 0.1 * values[caseWest[0],caseWest[1]])

                #East
                pList.append(0.8 * values[caseEast[0],caseEast[1]] \
                           + 0.1 * values[caseNorth[0],caseNorth[1]] \
                           + 0.1 * values[caseSouth[0],caseSouth[1]])

                #West
                pList.append(0.8 * values[caseWest[0],caseWest[1]] \
                           + 0.1 * values[caseNorth[0],caseNorth[1]] \
                           + 0.1 * values[caseSouth[0],caseSouth[1]])

                maxi = np.max(pList)

                values[row][col] = r + gamma * maxi



print("Values")
print(values)
print()

#Finding optimal Policy
policy=np.ones((3, 4))*-1

for col in range(4):
    for row in range(3):
        if (row,col) not in fixedCells:
            caseNorth, caseSouth, caseEast, caseWest = possibleStates(row, col)

            pList=[]
            #North
            pList.append(0.8 * values[caseNorth[0],caseNorth[1]] \
                        + 0.1 * values[caseEast[0],caseEast[1]] \
                        + 0.1 * values[caseWest[0],caseWest[1]])

            #Sud
            pList.append(0.8 * values[caseSouth[0],caseSouth[1]] \
                        + 0.1 * values[caseEast[0],caseEast[1]] \
                        + 0.1 * values[caseWest[0],caseWest[1]])

            #East
            pList.append(0.8 * values[caseEast[0],caseEast[1]] \
                        + 0.1 * values[caseNorth[0],caseNorth[1]] \
                        + 0.1 * values[caseSouth[0],caseSouth[1]])

            #West
            pList.append(0.8 * values[caseWest[0],caseWest[1]] \
                        + 0.1 * values[caseNorth[0],caseNorth[1]] \
                        + 0.1 * values[caseSouth[0],caseSouth[1]])

            policy[row][col] = np.argmax(pList)

print("Policy")
print("Actions : N=0, S=1, E=2, W=3")
print(policy)
print()



##Policy iteration
print("##### Policy iteration #####")
policy = np.random.randint(0,4,(3, 4))
for row,col in fixedCells:
    policy[row][col]=-1


for col in range(4):
    for row in range(3):
        if (row,col) not in fixedCells:
            r = R[row][col]
            caseNorth, caseSouth, caseEast, caseWest = possibleStates(row, col)
            action = policy[row][col]
            assert action != -1

            if action == 0: #North
                p = 0.8 * values[caseNorth[0],caseNorth[1]] \
                  + 0.1 * values[caseEast[0],caseEast[1]] \
                  + 0.1 * values[caseWest[0],caseWest[1]]
            elif action == 1: #Sud
                p = 0.8 * values[caseSouth[0],caseSouth[1]] \
                  + 0.1 * values[caseEast[0],caseEast[1]] \
                  + 0.1 * values[caseWest[0],caseWest[1]]
            elif action == 2: #East
                p = 0.8 * values[caseEast[0],caseEast[1]] \
                  + 0.1 * values[caseNorth[0],caseNorth[1]] \
                  + 0.1 * values[caseSouth[0],caseSouth[1]]
            elif action == 3: #West
                p = 0.8 * values[caseWest[0],caseWest[1]] \
                  + 0.1 * values[caseNorth[0],caseNorth[1]] \
                  + 0.1 * values[caseSouth[0],caseSouth[1]]

            values[row][col] = r + gamma * p

            pList=[]
            #North
            pList.append(0.8 * values[caseNorth[0],caseNorth[1]] \
                        + 0.1 * values[caseEast[0],caseEast[1]] \
                        + 0.1 * values[caseWest[0],caseWest[1]])

            #Sud
            pList.append(0.8 * values[caseSouth[0],caseSouth[1]] \
                        + 0.1 * values[caseEast[0],caseEast[1]] \
                        + 0.1 * values[caseWest[0],caseWest[1]])

            #East
            pList.append(0.8 * values[caseEast[0],caseEast[1]] \
                        + 0.1 * values[caseNorth[0],caseNorth[1]] \
                        + 0.1 * values[caseSouth[0],caseSouth[1]])

            #West
            pList.append(0.8 * values[caseWest[0],caseWest[1]] \
                        + 0.1 * values[caseNorth[0],caseNorth[1]] \
                        + 0.1 * values[caseSouth[0],caseSouth[1]])

            policy[row][col] = np.argmax(pList)




print("Values")
print(values)
print()

print("Policy")
print("Actions : N=0, S=1, E=2, W=3")
print(policy)