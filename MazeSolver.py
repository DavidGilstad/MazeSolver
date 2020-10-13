#
# Maze Solver
#
# Machine Learning project on reinforcement learning by teaching
# an agent how to go through a maze in optimal steps without running
# into objects along the way
#
#   authors: David Gilstad, Ian Schwind
#   created: October 9, 2020
#
import numpy as np
import math
import matplotlib.pyplot as plt
import cv2

# TODO: Find out what are good values and explain in paper
alpha, epsilon, gamma = 0.9, .2, 0.9
height = width = 6

# locations of the following objects on the grid
start = [5, 0]
end = [5, 5]
walls = [[2, 2], [3, 2], [3, 4], [5, 3], [4, 4]]

min_steps = 10  # minimum number of steps from start to finish
Rend, Rwall = 10, -0.01  # reward values
l, r, u, d = [0, 0, -1], [0, width-1, 1], [1, 0, -1], [1,height-1, 1]  # coordinate, boundary check, shift

class TD_agent:
    def __init__(self, lamda=0):
        self.prev_loc, self.curr_loc = None, list.copy(
            start)  # keep track of agent's location
        self.v = np.zeros([width, height])  # values of each state in the grid
        self.e = np.zeros([width, height]) # eligibility traces
        self.actions = [l, r, u, d]
        self.episode, self.start, self.end = 0, list.copy(start), list.copy(end)
        self.count = 0
        self.lamda = lamda

    # get values at each state the agent can move to
    def getValues(self):
        values = [-1, -1, -1, -1]
        if self.curr_loc[0] != 0:
            values[0] = self.v[self.curr_loc[0]-1, self.curr_loc[1]]
            # values[0] = self.v(self.curr_loc - [1,0])
        if self.curr_loc[0] < width-1:
            values[1] = self.v[self.curr_loc[0]+1, self.curr_loc[1]]
        if self.curr_loc[1] != 0:
            values[2] = self.v[self.curr_loc[0], self.curr_loc[1]-1]
        if self.curr_loc[1] < height-1:
            values[3] = self.v[self.curr_loc[0], self.curr_loc[1]+1]
        return values

    def vval(self, loc): return self.v[loc[0]][loc[1]]

    def TD(self, p, r, c, lamda):
        delta = r + gamma*self.vval(c) - self.vval(p) # get delta
        self.e[p[0]][p[1]] += 1 # update eligibility trace
        for i in range(0,height):
            for j in range(0,width):
                self.v[i][j] += alpha*delta*self.e[i][j]
                self.e[i][j] = self.e[i][j]*gamma*lamda

    def greedy(self):
        values = self.getValues()  # get all the surrounding state values

        # if values are all the same pick a random one, otherwise do the first max
        if (sum(values) == values[0]/len(values)):
            return self.explore()
        else:
            return values.index(max(values))

    # try a random option
    def explore(self):
        return np.random.choice([0, 1, 2, 3])

    # choose an action greedily, i.e. the max value action with probability
    # 1-epsilon, and a random action with probability epsilon
    def action(self):
        if self.prev_loc == end:
            self.count = 1
        else: self.count += 1

        i = np.random.choice([self.greedy(), self.explore()], p=[1-epsilon, epsilon])  # pick action

        # check for boundary violation, try different action if so
        if self.curr_loc[self.actions[i][0]] == self.actions[i][1]: self.action()
        else:  # do action
            self.prev_loc = list.copy(self.curr_loc)
            self.curr_loc[self.actions[i][0]] += self.actions[i][2]
            # learn from the action taken
            self.TD(self.prev_loc, 0, self.curr_loc, self.lamda)
            if self.curr_loc in walls:  # check if we entered into a wall
                self.prev_loc, self.curr_loc = list.copy(
                    self.curr_loc), list.copy(self.prev_loc)
                self.TD(self.prev_loc, Rwall, self.curr_loc, 0.8)
            elif self.curr_loc == self.end:  # check for end of maze
                self.episode += 1  # end of episode
                self.curr_loc, self.prev_loc = list.copy(
                    self.start), list.copy(end)
                self.TD(self.end, Rend, self.curr_loc, 0.8)

class SARSA_agent:
    def __init__(self):
        self.prev_loc = self.curr_loc = list.copy(
            start)  # keep track of agent's location
        # all actions have same value to start, so start anywhere
        self.prev_a = self.curr_a = 0
        # values of each state in the grid
        self.q = np.zeros([width, height, 4])
        self.actions = [l, r, u, d]
        self.episode, self.start, self.end = 0, list.copy(
            start), list.copy(end)
        self.reward = 0
        self.count = 0

    def qvals(self, loc): return self.q[loc[0]][loc[1]]

    def getReward(self): pass

    def SARSA(self, p, a1, r, c, a2):
        self.q[p[0]][p[1]][a1] += alpha * \
            (r + gamma*self.q[c[0]][c[1]][a2] - self.q[p[0]][p[1]][a1])

    # pick the action with maximum value
    def greedy(self):
        i = self.qvals(self.curr_loc)  # get available actions
        values = [i[0], i[1], i[2], i[3]]
        if (sum(values) == values[0]/len(values)):
            return self.explore()
        else:
            return values.index(max(values))

    def explore(self):
        return np.random.choice([0, 1, 2, 3])

    def action(self):
        if sarsa.prev_loc == [4, 5] and sarsa.curr_loc == start:
            self.count = 1
        else: self.count += 1

        # select an action
        i = self.curr_a = np.random.choice(
            [self.greedy(), self.explore()], p=[1-epsilon, epsilon])

        # check for boundary violation, try different action if so
        if self.curr_loc[self.actions[i][0]] == self.actions[i][1]:
            self.q[self.curr_loc[0]][self.curr_loc[1]][i] = - \
                1  # avoid boundary checks
            self.action()
        else:  # do action
            # update previous state now that we have the action
            self.SARSA(self.prev_loc, self.prev_a,
                       self.reward, self.curr_loc, self.curr_a)
            self.prev_a = self.curr_a

            # update locations
            self.prev_loc = list.copy(self.curr_loc)
            self.curr_loc[self.actions[i][0]] += self.actions[i][2]

            if self.curr_loc in walls:  # check if we entered into a wall
                self.curr_loc = list.copy(self.prev_loc)  # reset location
                self.reward = Rwall  # update reward
            elif self.curr_loc == self.end:  # check for end of maze
                self.episode += 1  # end of episode
                self.curr_loc = list.copy(self.start)  # reset to start
                self.reward = Rend
            else:
                # 1/(abs(end[0]-self.curr_loc[0]) + abs(end[1]-self.curr_loc[1]))
                self.reward = 0


if __name__ == '__main__':
    iters = 300
    y1 = np.linspace(0, 0, iters)  # track steps taken at each episode
    y2 = np.linspace(0, 0, iters)

    for experiments in range(0, 50):
        td, sarsa = TD_agent(0.8), SARSA_agent()
        while(td.episode < iters):
            td.action()
            if td.prev_loc == end:
                y1[td.episode-1] += td.count/50
                print(td.count, td.prev_loc, "->", td.curr_loc)

        while(sarsa.episode > iters):
            # sarsa.action()
            if sarsa.prev_loc == [4, 5] and sarsa.curr_loc == start:
                y2[sarsa.episode-1] += sarsa.count/50
                print(experiments,sarsa.count)

    x = np.linspace(1, iters, iters)
    fig, ax = plt.subplots()
    ax.plot(x, y1, 'g', label='TD-learning')
    #ax.plot(x, y2, 'b', label='SARSA')

    fig.suptitle("TD(lambda) average after 50 experiments")
    legend = ax.legend(loc='upper right', fontsize='x-large')
    plt.show()

    for i in range(0, height):
        for j in range(0, width):
            if [i, j] in walls: state = 'w'
            elif [i, j] == end: state = 'e'
            elif [i, j] == start: state = 's'
            else: state = '[]'
            print(state, round(td.v[i][j], 2), end='\t')
        print()

    def direc(i,j):
        if sarsa.q[i][j][0] == np.max(sarsa.q[i][j]): return 'U'
        elif sarsa.q[i][j][1] == np.max(sarsa.q[i][j]): return 'D'
        elif sarsa.q[i][j][2] == np.max(sarsa.q[i][j]): return 'L'
        elif sarsa.q[i][j][3] == np.max(sarsa.q[i][j]): return 'R'
        else: return 'X'

    for i in range(0, height):
        for j in range(0, width):
            if [i, j] in walls: state = 'w'
            elif [i, j] == end: state = 'e'
            elif [i, j] == start: state = 's'
            else: state = '[]'
            #print(state,round(a.q[i][j],2),end='\t')
            m = direc(i,j)
            print(state, m, end='\t')
        print()