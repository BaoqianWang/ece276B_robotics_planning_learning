import numpy as np
import gym
from utils import *
from example import example_use_of_gym_env
import copy

ST = 5 # Stay Static
MF = 0 # Move Forward
TL = 1 # Turn Left
TR = 2 # Turn Right
PK = 3 # Pickup Key
UD = 4 # Unlock Door
IFINITY=999999

def define_state(env,info):
    height=env.height
    width=env.width
    goal=info['goal_pos']
    orientation=[np.array([[1],[0]]),np.array([[0],[1]]), np.array([[-1],[0]]),np.array([[0],[-1]])] #right, up, left, down
    state_space = []
    goalIndex=[]
    keyDoor=[np.array([[1],[0]]),np.array([[1],[1]]), np.array([[0],[0]]),np.array([[0],[1]])]
    for i in range(width):
        for j in range(height):
            for k in keyDoor:
                for l in orientation:
                    state = {}
                    state['position']=np.array([[i],[j]])
                    state['keyDoor']=k
                    state['orientation']=l
                    #if(env.grid.get(i,j)==None):
                        #state_space.append(state)
                    #elif(env.grid.get(i,j).type!='wall'):
                    state_space.append(state)
                    if(i==goal[0] and j==goal[1]):
                        goalIndex.append(len(state_space)-1)

    return state_space, goalIndex

def get_optimal_action(policy,info,stateSpace):
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
    if(np.array_equal(s1['position'],s2['position']) and np.array_equal(s1['keyDoor'],s2['keyDoor']) and np.array_equal(s1['orientation'],s2['orientation'])):
        return 1
    else:
        return 0

def isWall(x,y,env):
    if(env.grid.get(x,y)==None):
        return 0
    if(env.grid.get(x,y).type=='wall'):
        return 1
    return 0

def get_cost(state, action,env,info):
    door=info['door_pos']
    key=info['key_pos']
    if(isWall(state['position'][0][0], state['position'][1][0],env)):
        return IFINITY
    if(action==0):
        isDoorNoO=(state['position'][0][0]+state['orientation'][0][0]==door[0] and state['position'][1][0]+state['orientation'][1][0]==door[1] and state['keyDoor'][1][0]==0)
        if ( isWall(state['position'][0][0]+state['orientation'][0][0],state['position'][1][0]+state['orientation'][1][0],env) or isDoorNoO):
            return IFINITY
        else:
            return 1
    if(action==1):
        return 1
    if(action==2):
        return 1
    if(action==3):
        if(state['position'][0][0] + state['orientation'][0][0]==key[0] and state['position'][1][0] + state['orientation'][1][0]== key[1] and state['keyDoor'][0][0]==0):
            return 1
        else:
            return IFINITY
    if(action==4):
        isDoorNoO=(state['position'][0][0]+state['orientation'][0][0]==door[0] and state['position'][1][0]+state['orientation'][1][0]==door[1] and state['keyDoor'][1][0]==0)
        if (isDoorNoO and state['keyDoor'][0][0]==1):
            return 1
        else:
            return IFINITY
    if(action==5):
        return 0

def stage_cost_matrix(env,info,stateSpace,actionSpace):
    numState=len(stateSpace)
    numAction=len(actionSpace)
    # Door, Key position
    costMatrix=np.zeros((numState,numAction))
    for i in range(numState):
        for j in range(numAction):
                    #Get cost from i state to j state
                    costMatrix[i,j]=get_cost(stateSpace[i],actionSpace[j],env,info)
    return costMatrix

def motion_model(state_i,action_j,stateSpace):
    currentState = copy.deepcopy(stateSpace[state_i])
    if action_j == 0:  # Move Forward
        currentState['position'] = currentState['position'] + currentState['orientation']
    if action_j == 1:  # Turn Left
        currentState['orientation'] = np.array([[0, 1], [-1, 0]]).dot(currentState['orientation'])
    if action_j == 2:  # Turn right
        currentState['orientation'] = np.array([[0, -1], [1, 0]]).dot(currentState['orientation'])
    if action_j == 3:  # Pick up the key
        currentState['keyDoor'][0][0] = 1
    if action_j == 4:  # Open Door
        currentState['keyDoor'][1][0] = 1
    for i in  range(len(stateSpace)):
        if(equal2state(currentState,stateSpace[i])):
            index=i
    return index, currentState


def doorkey_problem(env,info):
    stateSpace, goalIndex=define_state(env,info)
    actionSpace=[0,1,2,3,4,5]

    costMatrix=stage_cost_matrix(env,info,stateSpace,actionSpace)
    T=len(stateSpace)-1

    value=IFINITY*np.ones((len(stateSpace),T))
    policy=[]
    #Step 1: Initialize the terminal state cost
    for i in goalIndex:
        value[i,-1]=0

    #Step 2: Iterate backwards in time
    for j in range(T-2,-1,-1):
        Q=np.zeros((len(stateSpace),len(actionSpace)))
        for i in range(len(stateSpace)):
            for k in range(len(actionSpace)):
                indexNext, nextState=motion_model(i,k,stateSpace)
                Q=costMatrix[i,k]+value[:,indexNext]
        #policy[:,j]= np.argmin(Q,axis=1)
        policy.append(np.argmin(Q,axis=1))
        value[:,j]=np.amin(Q, axis=1)
        if(np.array_equal(value[:,j],value[:,j+1])):
            break
    policy=np.asarray(policy)
    # Get the optimal action with initial state
    #optimal_action=get_optimal_action(policy,info,stateSpace)

    return costMatrix, policy, value

def main():
    env_path = './envs/doorkey-5x5-normal.env'
    env, info = load_env(env_path) # load an environment
    costMatrix,policy,value = doorkey_problem(env,info) # find the optimal action sequence
    #action=[1,3,2,4,0,0,2,0]
    #draw_gif_from_seq(action, load_env(env_path)[0]) # draw a GIF & save
    return costMatrix, policy, value

if __name__ == '__main__':
    costMatrix, policy, value =main()
    #main()