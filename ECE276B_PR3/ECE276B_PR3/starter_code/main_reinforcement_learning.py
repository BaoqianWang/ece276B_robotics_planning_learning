"""
==================================
Inverted pendulum animation class
==================================

Adapted from the double pendulum problem animation.
https://matplotlib.org/examples/animation/double_pendulum_animated.html
"""
import heapq
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.stats import multivariate_normal
import random
import itertools
import scipy.interpolate
import copy
import random

class EnvAnimate:
    '''
    Initialize Inverted Pendulum Animation Settings
    '''
    def __init__(self):       
        pass
        
    def load_random_test_trajectory(self,):
        # Random trajectory for example
        self.theta = np.linspace(-np.pi, np.pi, self.t.shape[0])
        self.u = np.zeros(self.t.shape[0])

        self.x1 = np.sin(self.theta)
        self.y1 = np.cos(self.theta)

        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(111,autoscale_on=False, xlim=(-2,2), ylim=(-2,2))
        self.ax.grid()
        self.ax.axis('equal')
        plt.axis([-2, 2, -2, 2])

        self.line, = self.ax.plot([],[], 'o-', lw=2)
        self.time_template = 'time = %.1fs \nangle = %.2frad\ncontrol = %.2f'
        self.time_text = self.ax.text(0.05, 0.8, '', transform=self.ax.transAxes)
        pass
    
    '''
    Provide new rollout trajectory (theta and control input) to reanimate
    '''
    def load_trajectory(self, theta, u):
        """
        Once a trajectory is loaded, you can run start() to see the animation
        ----------
        theta : 1D numpy.ndarray
            The angular position of your pendulum (rad) at each time step
        u : 1D numpy.ndarray
            The control input at each time step
            
        Returns
        -------
        None
        """
        self.theta = theta
        self.x1 = np.sin(theta)
        self.y1 = np.cos(theta)
        self.u = u
        
        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(111,autoscale_on=False, xlim=(-2,2), ylim=(-2,2))
        self.ax.grid()
        self.ax.axis('equal')
        plt.axis([-2, 2,-2, 2])
        self.line, = self.ax.plot([],[], 'o-', lw=2)
        self.time_template = 'time = %.1fs \nangle = %.2frad\ncontrol = %.2f'
        self.time_text = self.ax.text(0.05, 0.9, '', transform=self.ax.transAxes)
        pass
    
    # region: Animation
    # Feel free to edit (not necessarily)
    def init(self):
        self.line.set_data([], [])
        self.time_text.set_text('')
        return self.line, self.time_text

    def _update(self, i):
        thisx = [0, self.x1[i]]
        thisy = [0, self.y1[i]] 
        self.line.set_data(thisx, thisy)
        self.time_text.set_text(self.time_template % (self.t[i], self.theta[i], self.u[i]))
        return self.line, self.time_text

    def start(self):
        print('Starting Animation')
        print()
        # Set up plot to call animate() function periodically
        self.ani = FuncAnimation(self.fig, self._update, frames=range(len(self.x1)), interval=25, blit=True, init_func=self.init, repeat=False)
        self.ani.save('test_trajectory.gif')
        plt.show()
    # endregion: Animation


class InvertedPendulum(EnvAnimate):
    def __init__(self,dt,vmax,umax,n1,n2,nu,a,b,sigma,k,r,gamma):
        EnvAnimate.__init__(self,)
        # Change this to match your discretization
        # Usually, you will need to load parameters including
        # constants: dt, vmax, umax, n1, n2, nu
        # parameters: a, b, σ, k, r, γ
        self.dt = dt
        self.vmax=vmax
        self.umax=umax
        self.n1=n1 #The discretization should be chosen carefully
        self.n2=n2
        self.nu=nu
        self.a=a
        self.b=b
        self.sigma=sigma
        self.k=k
        self.r=r
        self.gamma=gamma
        self.t = np.arange(0.0, 20.0, self.dt)
        #self.load_random_test_trajectory()
        #Initialize states and control inputs
        self.x1=0
        self.x2=0
        self.u=0
        self.state_x1 = [j for j in np.linspace(0, 2*np.pi, self.n1, endpoint=True)]
        self.state_x2_h1 = [i for i in np.linspace(0,self.vmax,self.n2/2,endpoint=True)]
        self.state_x2_h2=  [i for i in np.linspace(-self.vmax,0,self.n2/2,endpoint=False)]
        self.state_x2=self.state_x2_h2+self.state_x2_h1
        self.state_space = list(itertools.product(self.state_x1, self.state_x2))
        self.action_space=[k for k in np.linspace(-self.umax,self.umax,self.nu,endpoint=True)]

        
    # TODO: Implement your own VI and PI algorithms
    # Feel free to edit/delete these functions
    def l_xu(self, x1, x2, u):
        # Stage cost
        cost=(1-np.exp(-self.k*np.cos(x1)-self.k)+(self.r/2)*u**2)*self.dt
        return cost


    def env_step(self,state,action):
        reward = self.l_xu(state[0], state[1], action)
        mean_x2 = state[1] + self.dt * ((self.a * np.sin(state[0]) - self.b * state[1]) + action)
        if mean_x2>self.vmax:
            mean_x2=self.vmax
        if mean_x2<-self.vmax:
            mean_x2=-self.vmax

        mean_x1=state[0] + state[1] * self.dt
        if mean_x1>2 * np.pi:
            mean_x1=np.remainder(mean_x1,2*np.pi)
        if mean_x1<0:
            mean_x1=np.remainder(mean_x1,2*np.pi)

        mean = np.array([mean_x1, mean_x2])
        covariance = self.sigma.dot(self.sigma.T) * self.dt
        pd = multivariate_normal(mean=[mean[0], mean[1]],
                                 cov=[[covariance[0][0], covariance[0][1]], [covariance[1][0], covariance[1][1]]])

        ret = []
        total_prob = 0
        prob_next_state = pd.pdf(self.state_space)

        # Evaluate six highest probability states
        filtered_state_index = heapq.nlargest(6, range(len(prob_next_state)), prob_next_state.__getitem__)
        index=random.choice(filtered_state_index)
        ret=(self.state_space[index], reward)
        return ret

    def env_transition(self,state,action):
        reward=self.l_xu(state[0],state[1],action)
        mean_x2=state[1] + self.dt * ((self.a * np.sin(state[0]) - self.b * state[1]) + action)
        if mean_x2>self.vmax:
            mean_x2=self.vmax
        if mean_x2<-self.vmax:
            mean_x2=-self.vmax
        mean = np.array([np.remainder(state[0] + state[1] * self.dt,2*np.pi), mean_x2])
        covariance = self.sigma.dot(self.sigma.T) * self.dt

        pd = multivariate_normal(mean=[mean[0], mean[1]],
                                 cov=[[covariance[0][0], covariance[0][1]], [covariance[1][0], covariance[1][1]]])

        ret=[]
        total_prob=0
        prob_next_state=pd.pdf(self.state_space)

        #Evaluate six highest probability states
        filtered_state_index=heapq.nlargest(6, range(len(prob_next_state)), prob_next_state.__getitem__)
        # Normalize the probability
        for i in filtered_state_index:
            total_prob+=prob_next_state[i]
        for i in filtered_state_index:
            ret.append((prob_next_state[i]/total_prob,self.state_space[i],reward))
        return ret


    def Q_learning(self):
        Q_value=1000*np.ones((len(self.state_space),len(self.action_space)))
        policy=dict.fromkeys(self.state_space,0)
        box = self.state_space.index((0,0))
        MAX_STEPS=500000
        GAMMA=0.8
        steps=0
        epsilon=0.5
        current_state = (0, 0)
        while(steps<=MAX_STEPS):
            steps+=1
            print('num_step', steps)
            if(np.random.rand()>epsilon):
                best_action_index=np.argmin(Q_value[box,:])
                best_action=self.action_space[best_action_index]
            else:
                best_action=random.choice(self.action_space)
                best_action_index=self.action_space.index(best_action)

            ret=self.env_step(current_state,best_action)
            next_state=ret[0]
            reward=ret[1]
            print(best_action,current_state,next_state)
            next_box = self.state_space.index(next_state)
            Q_value[box, best_action_index] = Q_value[box, best_action_index] + 0.5 * (reward + GAMMA * np.min(Q_value[next_box,:]) - Q_value[box, best_action_index])
            box=copy.deepcopy(next_box)
            current_state=copy.deepcopy(next_state)

        for state in self.state_space:
            box=self.state_space.index(state)
            best_action_index = np.argmin(Q_value[box, :])
            best_action = self.action_space[best_action_index]
            policy[state]=best_action

        return Q_value, policy


    def one_step_look_ahead(self,state,V_x):
        # The value of taking all actions in the state
        A=dict.fromkeys(self.action_space,0)

        for action in self.action_space:
            for prob, next_state, reward in self.env_transition(state,action):
                A[action] += prob * (reward + self.gamma * V_x[next_state])
        return A

    def value_iteration(self):
        V_x=dict.fromkeys(self.state_space, 0)
        VI_policy = dict.fromkeys(self.state_space,0)
        episodeV=[]
        episodeV.append(copy.deepcopy(V_x))
        #Start value iteration
        while True:
            delta = 0
            for state in self.state_space:
                A=self.one_step_look_ahead(state,V_x)
                best_action_value = min(A.values())
                delta = max(delta, np.abs(best_action_value - V_x[state]))
            # Update the value function. Ref: Sutton book eq. 4.10.
                V_x[state] = best_action_value
            # Check if we can stop
            episodeV.append(copy.deepcopy(V_x))
            print('Value Iteration Error',delta)
            if delta < 0.02:
                break

        for state in self.state_space:
            # One step lookahead to find the best action for this state
            A = self.one_step_look_ahead(state, V_x)
            best_action = min(A, key=lambda k: A[k])
            # Always take the best action
            VI_policy[state] = best_action
        return V_x, VI_policy,episodeV

    def policy_evaluation(self,PI_policy):
        # Start with a random (all 0) value function
        V_x = dict.fromkeys(self.state_space, 0)
        while True:
            delta = 0
            # For each state, perform a "full backup"
            for state in self.state_space:
                v = 0
                # For each action, look at the possible next states...
                for prob, next_state, reward in self.env_transition(state,PI_policy[state]):
                    # Calculate the expected value
                    v += prob * (reward + self.gamma * V_x[next_state])
                # How much our value function changed (across any states)
                delta = max(delta, np.abs(v - V_x[state]))

                V_x[state] = v
            # Stop evaluating once our value function change is below a threshold
            if delta < 0.02:
                break
        return V_x

    def policy_iteration(self):
        #Initialize a random policy
        PI_policy=dict()
        V_0=dict.fromkeys(self.state_space, 0)
        episodeV=[]
        episodeV.append(V_0)
        for state in self.state_space:
            PI_policy[state]=random.choice(self.action_space)


        while True:
            V_x=self.policy_evaluation(PI_policy)
            episodeV.append(V_x)
            a=episodeV[-2:]
            delta=0
            for state in self.state_space:
                action=PI_policy[state]
                A=self.one_step_look_ahead(state,V_x)
                best_action=min(A, key=lambda k: A[k])
                PI_policy[state]=best_action
            preV=episodeV[-2]
            currV=episodeV[-1]
            for state in self.state_space:
                delta = max(delta, np.abs(preV[state] - currV[state]))
            print('Policy Iteration Error', delta)
            if delta<0.02:
                return V_x, PI_policy,episodeV



    def policy_interpolation(self,policy):
        x1=np.asarray(self.state_x1)
        x2=np.asarray(self.state_x2)

        fxy=np.zeros((len(self.state_x2),len(self.state_x1)))
        for i in range(len(self.state_x1)):
            for j in range(len(self.state_x2)):
                fxy[j,i]=policy[self.state_x1[i],self.state_x2[j]]
        con_policy=scipy.interpolate.interp2d(x1, x2, fxy, kind='cubic')

        return con_policy


    def generate_trajectory(self, init_state, policy):
        x1 = np.asarray(self.state_x1)
        x2 = np.asarray(self.state_x2)

        fxy = np.zeros((len(self.state_x2), len(self.state_x1)))
        for i in range(len(self.state_x1)):
            for j in range(len(self.state_x2)):
                fxy[j, i] = policy[self.state_x1[i], self.state_x2[j]]
        con_policy = scipy.interpolate.interp2d(x1, x2, fxy, kind='cubic')

        theta=[]
        u=[]
        state=init_state
        covariance = self.sigma.dot(self.sigma.T) * self.dt
        for dt in self.t:
            action=con_policy(state[0],state[1])

            mean = np.array([state[0] + state[1] * self.dt,
                             state[1] + self.dt * (self.a * np.sin(state[0]) - self.b * state[1] + action[0])])

            noise=np.random.multivariate_normal(np.array([0,0]), covariance, 1)
            print(noise)
            state[0]=mean[0]+noise[0][0]
            state[1]=mean[1]+noise[0][1]
            theta.append(np.pi-state[0])
            u.append(action[0])

        return theta, u

    def plot_value_policy(self,epVI_x, epPI_x, VI_policy, PI_policy):

        #iterVI=np.arange(0,len(epVI_x),1)
        #iterPI=np.arange(0,len(epPI_x),1)
        num_state=len(self.state_space)
        states_to_show=[self.state_space[0],self.state_space[round(num_state/3)],self.state_space[round(2*num_state/3)],self.state_space[-1]]

        #Value VI
        plt.figure(1)
        for state in states_to_show:
            value_VI=[]
            for VI in epVI_x:
                value_VI.append(VI[state])
            s="state: (%.1f, %.1f)" %(state[0],state[1])
            plt.plot(value_VI,label=s)
        plt.xlabel('Iteration')
        plt.ylabel('Value')
        plt.legend()

        #Policy VI
        data = np.zeros((4, 4))
        for i in range(4):
            for j in range(4):
                state=self.state_space[i*3+j]
                data[i,j]=VI_policy[state]
        plt.matshow(data)

        for i in range(4):
            for j in range(4):
                state = self.state_space[i * 3 + j]
                s = "(%0.1f,%0.1f): %0.1f" % (state[0], state[1], VI_policy[state])
                plt.text(i,j, s, va="center", ha="center")
        plt.show()

        #Value PI
        plt.figure(3)
        for state in states_to_show:
            value_PI = []
            for PI in epPI_x:
                value_PI.append(PI[state])
            s = "state: (%.1f, %.1f)" % (state[0], state[1])
            plt.plot(value_PI, label=s)
        plt.xlabel('Iteration')
        plt.ylabel('Value')
        plt.legend()

        #Policy PI
        data = np.zeros((4, 4))
        for i in range(4):
            for j in range(4):
                state=self.state_space[i*3+j]
                data[i,j]=PI_policy[state]
        plt.matshow(data)

        for i in range(4):
            for j in range(4):
                state = self.state_space[i * 3 + j]
                s = "(%0.1f,%0.1f): %0.1f" % (state[0], state[1], PI_policy[state])
                plt.text(i,j, s, va="center", ha="center")

        #Value Comparison
        for i, state in enumerate(states_to_show):
            plt.figure(5+i)
            value_PI = []
            value_VI=[]
            for PI in epPI_x:
                value_PI.append(PI[state])
            for VI in epVI_x:
                value_VI.append(VI[state])
            s1 = "Policy Iteration with state: (%.1f, %.1f)" % (state[0], state[1])
            s2 = "Value Iteration with state: (%.1f, %.1f)" % (state[0], state[1])
            plt.plot(value_PI, label=s1)
            plt.plot(value_VI,label=s2)
            plt.xlabel('Iteration')
            plt.ylabel('Value')
            plt.legend()

if __name__ == '__main__':
    # Usually, you will need to load parameters including
    # constants: dt, vmax, umax, n1, n2, nu
    # parameters: a, b, σ, k, r, γ
    sigma=np.array([[.1,0],[0,.1]])
    dt=0.1
    vmax=3
    length=3
    mass=1
    umax=7
    damping=0.09
    g=9.8
    n1=50
    n2=60
    nu=20
    a=-g/length
    b=0.4
    control_cost=.001
    k=1
    gamma=0.6
    inv_pendulum = InvertedPendulum(dt,vmax,umax,n1,n2,nu,a,b,sigma,k,control_cost,gamma)
    Q_value, q_policy=inv_pendulum.Q_learning()
    #vi1, VI_policy, epV1 = inv_pendulum.value_iteration()
    #con_VI_policy = inv_pendulum.policy_interpolation(q_policy)
    #Set the initial state
    initial_state = [0, 0]

    #Start simulation and generate trajectory
    theta, u = inv_pendulum.generate_trajectory(initial_state,q_policy)
    inv_pendulum.load_trajectory(theta, u)
    inv_pendulum.start()
    '''
    #Value Iteration
    vi1, VI_policy,epV1=inv_pendulum.value_iteration()

    #Policy Iteration
    vi2, PI_policy,epV2 = inv_pendulum.policy_iteration()

    #Plot results
    inv_pendulum.plot_value_policy(epV1,epV2,VI_policy,PI_policy)

    #Interpolation
    con_VI_policy=inv_pendulum.policy_interpolation(VI_policy)


    ######## TODO: Implement functions for visualization ########
    #############################################################
    #Set the initial state
    initial_state = [0, 0]

    #Start simulation and generate trajectory
    theta, u = inv_pendulum.generate_trajectory(initial_state,con_VI_policy)
    inv_pendulum.load_trajectory(theta, u)
    inv_pendulum.start()
    '''
