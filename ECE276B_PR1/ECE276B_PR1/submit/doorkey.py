import numpy as np
import gym
from utils import *
from example import example_use_of_gym_env
import copy

MF = 0 # Move Forward
TL = 1 # Turn Left
TR = 2 # Turn Right
PK = 3 # Pickup Key
UD = 4 # Unlock Door
IFINITY=999999

def action2state(pres,currs):
    #Convert state sequence to action
    if(np.array_equal(currs['position'], pres['position'] + pres['orientation'])):
        return MF
    if(np.array_equal(currs['orientation'], np.array([[0, 1], [-1, 0]]).dot(pres['orientation']))):
        return TL
    if (np.array_equal(currs['orientation'],np.array([[0, -1], [1, 0]]).dot(pres['orientation']))):  # Turn right
        return TR
    if (currs['keyDoor'][0][0] == pres['keyDoor'][0][0]+1):
        return PK
    if (currs['keyDoor'][1][0] == pres['keyDoor'][1][0]+1):
        return UD

def define_state(env,info):
    #Generate state space
    height=env.height
    width=env.width
    goal=info['goal_pos']
    orientation=[np.array([[1],[0]]),np.array([[0],[1]]), np.array([[-1],[0]]),np.array([[0],[-1]])] #right, up, left, down
    state_space = []
    goalIndex=[]
    keyDoor=[np.array([[1],[0]]),np.array([[1],[1]]), np.array([[0],[0]])]
    for i in range(width):
        for j in range(height):
            for k in keyDoor:
                for l in orientation:
                    state = {}
                    state['position']=np.array([[i],[j]])
                    state['keyDoor']=k
                    state['orientation']=l
                    if(env.grid.get(i,j)==None):
                        state_space.append(state)
                    elif(env.grid.get(i,j).type!='wall'):
                        state_space.append(state)
                    if(i==goal[0] and j==goal[1]):
                        goalIndex.append(len(state_space)-1)

    return state_space, goalIndex

def step_action(preState,action,door,key):
    #Get the next state given the current state and the action
    currentState=copy.deepcopy(preState)
    if action==0: #Move Forward
        if(currentState['position'][0][0]==door[0] and currentState['position'][1][0]==door[1] and currentState['keyDoor'][1][0]!=1):
            return currentState
        currentState['position']=currentState['position']+currentState['orientation']
    if action==1: #Turn Left
        currentState['orientation']=np.array([[0,1],[-1,0]]).dot(currentState['orientation'])
    if action==2: #Turn right
        currentState['orientation']=np.array([[0,-1],[1,0]]).dot(currentState['orientation'])
    if action==3: #Pick up the key
        if(currentState['position'][0][0]+currentState['orientation'][0][0]==key[0] and currentState['position'][1][0]+currentState['orientation'][1][0]==key[1]):
            currentState['keyDoor'][0][0]=1
    if action==4: #Open Door
        if(currentState['position'][0][0]+currentState['orientation'][0][0]==door[0] and currentState['position'][1][0]+currentState['orientation'][1][0]==door[1] and currentState['keyDoor'][0][0]==1):
            currentState['keyDoor'][1][0]=1
    return currentState

def get_optimal_action(policy,info,stateSpace):
    #Get the optimal action
    arrPolicy=np.asarray(policy)
    startState={}
    startState['position']=info['init_agent_pos'].reshape(2,1)
    startState['keyDoor']=np.array([[0],[0]])
    startState['orientation']=info['init_agent_dir'].reshape(2,1)
    for i in range(len(stateSpace)):
        if equal2state(startState,stateSpace[i]):
            index = i
            break
    action=[]
    action.append(index)
    tempAction=index
    for i in range(len(policy)-1,-1,-1):
        tempAction=arrPolicy[i][tempAction]
        action.append(tempAction)
    optimal_action=[]
    for j, k in zip(action,action[1:]):
        if(equal2state(stateSpace[j],stateSpace[k])):
            continue
        optimal_action.append(action2state(stateSpace[j],stateSpace[k]))
    return optimal_action


def equal2state(s1,s2):
    #Determine if two state are equal
    if(np.array_equal(s1['position'],s2['position']) and np.array_equal(s1['keyDoor'],s2['keyDoor']) and np.array_equal(s1['orientation'],s2['orientation'])):
        return 1
    else:
        return 0

def get_cost(preState, nextState,door,key):
    #Get cost between two states
    for i in range(5):
        possStep=step_action(preState, i, door, key)
        if(equal2state(nextState,possStep)):
            return 1
    return IFINITY

def cost_matrix(env,info,state_space):
    numState=len(state_space)
    # Door, Key position
    door=info['door_pos']
    key=info['key_pos']
    costMatrix=np.zeros((numState,numState))
    for i in range(numState):
        for j in range(numState):
                if(i==j): #If the current state equal to next state
                    costMatrix[i,j]=0
                else:
                    #Get cost from i state to j state
                    costMatrix[i,j]=get_cost(state_space[i],state_space[j],door,key)
    return costMatrix

def doorkey_problem(env,info):
    stateSpace, goalIndex=define_state(env,info)
    costMatrix=cost_matrix(env,info,stateSpace)
    T=len(stateSpace)-1

    value=IFINITY*np.ones((len(stateSpace),T))
    policy=[]
    #Step 1: Initialize the terminal state cost
    for i in goalIndex:
        value[i,-1]=0

    #Step 2: Iterate backwards in time
    for j in range(T-2,-1,-1):
        Q=np.zeros((len(stateSpace),len(stateSpace)))
        Q=costMatrix+value[:,j+1]
        policy.append(np.argmin(Q,axis=1))
        value[:,j]=np.amin(Q, axis=1)
        if(np.array_equal(value[:,j],value[:,j+1])):
            break
    policy=np.asarray(policy)
    # Get the optimal action with initial state
    optimal_action=get_optimal_action(policy,info,stateSpace)
    return value, policy.T, optimal_action

def main():
    env_path = './envs/doorkey-6x6-direct.env'
    env, info = load_env(env_path) # load an environment
    value, policy, action = doorkey_problem(env,info) # find the optimal action sequence
    draw_gif_from_seq(action, load_env(env_path)[0]) # draw a GIF & save
    return value, policy, action

if __name__ == '__main__':
    value, policy, action=main()
    print('The optimal sequence is :', action)