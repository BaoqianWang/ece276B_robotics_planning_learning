"""
==================================
Inverted pendulum animation class
==================================

Adapted from the double pendulum problem animation.
https://matplotlib.org/examples/animation/double_pendulum_animated.html
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.stats import multivariate_normal
import random


class EnvAnimate:
    '''
    Initialize Inverted Pendulum Animation Settings
    '''

    def __init__(self):
        pass

    def load_random_test_trajectory(self, ):
        # Random trajectory for example
        self.theta = np.linspace(-np.pi, np.pi, self.t.shape[0])
        self.u = np.zeros(self.t.shape[0])

        self.x1 = np.sin(self.theta)
        self.y1 = np.cos(self.theta)

        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(111, autoscale_on=False, xlim=(-2, 2), ylim=(-2, 2))
        self.ax.grid()
        self.ax.axis('equal')
        plt.axis([-2, 2, -2, 2])

        self.line, = self.ax.plot([], [], 'o-', lw=2)
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
        self.ax = self.fig.add_subplot(111, autoscale_on=False, xlim=(-2, 2), ylim=(-2, 2))
        self.ax.grid()
        self.ax.axis('equal')
        plt.axis([-2, 2, -2, 2])
        self.line, = self.ax.plot([], [], 'o-', lw=2)
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
        self.ani = FuncAnimation(self.fig, self._update, frames=range(len(self.x1)), interval=25, blit=True,
                                 init_func=self.init, repeat=False)
        plt.show()
    # endregion: Animation


class InvertedPendulum(EnvAnimate):
    def __init__(self, dt, vmax, umax, n1, n2, nu, a, b, sigma, k, r, gamma):
        EnvAnimate.__init__(self, )
        # Change this to match your discretization
        # Usually, you will need to load parameters including
        # constants: dt, vmax, umax, n1, n2, nu
        # parameters: a, b, σ, k, r, γ
        self.dt = dt
        self.vmax = vmax
        self.umax = umax
        self.n1 = n1  # The discretization should be chosen carefully
        self.n2 = n2
        self.nu = nu
        self.a = a
        self.b = b
        self.sigma = sigma
        self.k = k
        self.r = r
        self.gamma = gamma
        self.t = np.arange(0.0, 2.0, self.dt)
        # self.load_random_test_trajectory()
        # Initialize states and control inputs
        self.x1 = 0
        self.x2 = 0
        self.u = 0

        self.state_space= a = ["foo", "melon"]
b = [True, False]
c = list(itertools.product(a, b))
>> [("foo", True), ("foo", False), ("melon", True), ("melon", False)]
        pass

    # TODO: Implement your own VI and PI algorithms
    # Feel free to edit/delete these functions
    def l_xu(self, x1, x2, u):
        # Stage cost
        cost = (1 - np.exp(self.k * np.cos(x1) - self.k) + (self.r / 2) * u ** 2) * self.dt
        return cost


    def policy_iteration(self,):
        # Note the discounted problem or first exit problem
        state_x1 = [j for j in np.linspace(-np.pi, np.pi, self.n1, endpoint=True)]
        state_x2 = [i for i in np.linspace(-self.vmax, self.vmax, self.n2, endpoint=True)]
        control = [k for k in np.linspace(-self.umax, self.umax, self.nu, endpoint=True)]
        V_x,preV_x = dict(),dict()
        PI_policy = dict()

        # Initialization
        for x1 in state_x1:
            for x2 in state_x2:
                V_x[x1, x2] = 0
                preV_x[x1, x2] = 0
                PI_policy[x1,x2]=random.choice(control)

        while(1):
            diff=0
            # Policy evaluation
            for x1 in state_x1:
                for x2 in state_x2:
                    mu=PI_policy[x1,x2]
                    V_x[x1,x2]=self.l_xu(x1,x2,mu)+self.gamma*self.get_expectation(x1,x2,mu,V_x)
                    diff += np.abs(V_x[x1, x2]-preV_x[x1,x2])
                    preV_x[x1,x2] = V_x[x1,x2]
            print('Policy Iteration Error',diff)
            if(diff<0.01):
                break

            # Policy improvement
            for x1 in state_x1:
                for x2 in state_x2:
                # Policy improvement
                    for mu2 in control:
                        minV_x = 99999
                        cost=self.l_xu(x1,x2,mu2)+self.gamma*self.get_expectation(x1,x2,mu2,V_x)
                        if(cost<=minV_x):
                            minV_x = cost
                            PI_policy[x1,x2]=mu2

        return V_x, PI_policy,


    def one_step_lookahead(self,state, V):
        """
        Helper function to calculate the value for all action in a given state.

        Args:
            state: The state to consider (int)
            V: The value to use as an estimator, Vector of length env.nS

        Returns:
            A vector of length env.nA containing the expected value of each action.
        """
        A = np.zeros(self.nu)
        for a in range(self.nu):
            for prob, next_state, reward, done in env.P[state][a]:
                A[a] += prob * (reward + discount_factor * V[next_state])
        return A



    def value_iteration(self, theta=0.0001, discount_factor=1.0):


        """
        Value Iteration Algorithm.

        Args:
            env: OpenAI env. env.P represents the transition probabilities of the environment.
                env.P[s][a] is a list of transition tuples (prob, next_state, reward, done).
                env.nS is a number of states in the environment.
                env.nA is a number of actions in the environment.
            theta: We stop evaluation once our value function change is less than theta for all states.
            discount_factor: Gamma discount factor.

        Returns:
            A tuple (policy, V) of the optimal policy and the optimal value function.
        """

        numS=self.n1*self.n2
        V = np.zeros(numS)
        while True:
            # Stopping condition
            delta = 0
            # Update each state...
            for s in range(numS):
                # Do a one-step lookahead to find the best action
                A = one_step_lookahead(s, V)
                best_action_value = np.max(A)
                # Calculate delta across all states seen so far
                delta = max(delta, np.abs(best_action_value - V[s]))
                # Update the value function. Ref: Sutton book eq. 4.10.
                V[s] = best_action_value
                # Check if we can stop
            if delta < theta:
                break

        # Create a deterministic policy using the optimal value function
        policy = np.zeros(numS)
        for s in range(numS):
            # One step lookahead to find the best action for this state
            A = one_step_lookahead(s, V)
            best_action = np.argmax(A)
            # Always take the best action
            policy[s] = best_action

        return policy, V


    # def f_xu(self, x1, x2, u):
    #     # Motion model
    #     mean=np.array([x1+x2*self.dt,x2+self.dt*(self.a*np.sin(x1)-self.b*np.sin(x2)+u)])
    #     covariance=self.sigma.dot(self.sigma.T)*self.dt
    #     #Need pay attention on how to choose proper number
    #     #Address the key error
    #     #nextStatesX1=[i for i in np.linspace(mean[0]-mean[0]/2, mean[0]+mean[0]/2, self.n1) if np.abs(i) <= np.pi ]
    #     #nextStatesX2=[j for j in np.linspace(mean[1]-mean[1]/2, mean[1]+mean[1]/2, self.n2) if np.abs(j) <= self.vmax]
    #     pd =multivariate_normal(mean=[mean[0], mean[1]], cov=[[covariance[0][0],covariance[0][1]],[covariance[1][0],covariance[1][1]]])
    #     #probabilityStates=dict()
    #     #total_probability=0
    #     #for nextX1 in nextStatesX1:
    #         #for nextX2 in nextStatesX2:
    #     probability = pd.pdf([nextX1, nextX2])
    #
    #     #Normalize the probability
    #     #for nextX1 in nextStatesX1:
    #         #for nextX2 in nextStatesX2:
    #             #probabilityStates[nextX1, nextX2]=probabilityStates[nextX1,nextX2]/total_probability
    #
    #     return probability

    # def get_expectation(self, x1, x2, mu, V_x):
    #     mean = np.array([x1 + x2 * self.dt, x2 + self.dt * (self.a * np.sin(x1) - self.b * np.sin(x2) + mu)])
    #     covariance = self.sigma.dot(self.sigma.T) * self.dt
    #     state_x1 = [j for j in np.linspace(-np.pi, np.pi, self.n1, endpoint=True)]
    #     state_x2 = [i for i in np.linspace(-self.vmax, self.vmax, self.n2, endpoint=True)]
    #
    #     nextX1 = [min(state_x1, key=lambda x: abs(x - j)) for j in
    #               np.linspace(-mean[0] / 2, mean[0] / 2, 5, endpoint=True)]
    #     nextX2 = [min(state_x2, key=lambda x: abs(x - j)) for j in
    #               np.linspace(-mean[1] / 2, mean[1] / 2, 5, endpoint=True)]
    #     nextX1 = set(nextX1)
    #     nextX2 = set(nextX2)
    #
    #     pd = multivariate_normal(mean=[mean[0], mean[1]],
    #                              cov=[[covariance[0][0], covariance[0][1]], [covariance[1][0], covariance[1][1]]])
    #
    #     expectation = 0
    #     for nx1 in nextX1:
    #         for nx2 in nextX2:
    #             expectation += pd.pdf([nx1, nx2]) * V_x[nx1, nx2]
    #     return expectation

    # def value_iteration(self):
    #     # Note the discounted problem or first exit problem
    #     state_x1 = [j for j in np.linspace(-np.pi, np.pi, self.n1, endpoint=True)]
    #     state_x2 = [i for i in np.linspace(-self.vmax, self.vmax, self.n2, endpoint=True)]
    #     control = [k for k in np.linspace(-self.umax, self.umax, self.nu, endpoint=True)]
    #     V_x = dict()
    #     VI_policy = dict()
    #
    #     # Initialization
    #     for x1 in state_x1:
    #         for x2 in state_x2:
    #             V_x[x1, x2] = 0
    #
    #     # Start value iteration
    #     while (1):
    #         diff = 0
    #         for x1 in state_x1:
    #             for x2 in state_x2:
    #                 minV_x = 99999
    #                 for mu in control:
    #                     cost = self.l_xu(x1, x2, mu) + self.gamma * self.get_expectation(x1, x2, mu, V_x)
    #                     if (cost <= minV_x):
    #                         minV_x = cost
    #                         VI_policy[x1, x2] = mu
    #                 diff += np.abs(minV_x - V_x[x1, x2])
    #                 V_x[x1, x2] = minV_x
    #         print('Value Iteration Error', diff)
    #         if (diff < .01):
    #             break
    #     return V_x, VI_policy,

    def policy_iteration(self, ):
        # Note the discounted problem or first exit problem
        state_x1 = [j for j in np.linspace(-np.pi, np.pi, self.n1, endpoint=True)]
        state_x2 = [i for i in np.linspace(-self.vmax, self.vmax, self.n2, endpoint=True)]
        control = [k for k in np.linspace(-self.umax, self.umax, self.nu, endpoint=True)]
        V_x, preV_x = dict(), dict()
        PI_policy = dict()

        # Initialization
        for x1 in state_x1:
            for x2 in state_x2:
                V_x[x1, x2] = 0
                preV_x[x1, x2] = 0
                PI_policy[x1, x2] = random.choice(control)

        while (1):
            diff = 0
            # Policy evaluation
            for x1 in state_x1:
                for x2 in state_x2:
                    mu = PI_policy[x1, x2]
                    V_x[x1, x2] = self.l_xu(x1, x2, mu) + self.gamma * self.get_expectation(x1, x2, mu, V_x)
                    diff += np.abs(V_x[x1, x2] - preV_x[x1, x2])
                    preV_x[x1, x2] = V_x[x1, x2]
            print('Policy Iteration Error', diff)
            if (diff < 0.01):
                break

            # Policy improvement
            for x1 in state_x1:
                for x2 in state_x2:
                    # Policy improvement
                    for mu2 in control:
                        minV_x = 99999
                        cost = self.l_xu(x1, x2, mu2) + self.gamma * self.get_expectation(x1, x2, mu2, V_x)
                        if (cost <= minV_x):
                            minV_x = cost
                            PI_policy[x1, x2] = mu2

        return V_x, PI_policy,

    def generate_trajectory(self, init_state, policy, t):
        theta, u = None, None
        return theta, u


if __name__ == '__main__':
    # Usually, you will need to load parameters including
    # constants: dt, vmax, umax, n1, n2, nu
    # parameters: a, b, σ, k, r, γ
    sigma = np.array([[1, 0.4], [0.3, 1]])
    inv_pendulum = InvertedPendulum(0.1, 10, 10, 50, 50, 50, 1, 1, sigma, 1, 1, 0.4)
    # inv_pendulum.start()
    # states=inv_pendulum.f_xu(2,2,1)
    vi1, VI_policy = inv_pendulum.value_iteration()
    vi2, PI_policy = inv_pendulum.policy_iteration()


    ######## TODO: Implement functions for visualization ########
    #############################################################




