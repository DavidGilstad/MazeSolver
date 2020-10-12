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

alpha, epsilon, gamma = 0.1, .1, 0.9 # TODO: Find out what are good values and explain in paper
height = width = 6

# locations of the following objects on the grid
start = [5,0]
end = [5,5]
walls = [[2,2],[3,2],[5,3],[3,4],[4,4]]

min_steps = 10 # minimum number of steps from start to finish
Rend, Rwall = 10, -1 # reward values
l, r, u, d = [0,0,-1], [0,width-1,1], [1,0,-1], [1,height-1,1] # coordinate, boundary check, shift

class TD_agent:
    def __init__(self):
        self.prev_loc, self.curr_loc = None, list.copy(start) # keep track of agent's location
        self.v = np.zeros([width,height]) # values of each state in the grid
        self.actions = [l, r, u, d]
        self.episode, self.start, self.end = 0, list.copy(start), list.copy(end)
    
    # get values at each state the agent can move to
    def getValues(self):
        values = [-1,-1,-1,-1]
        if self.curr_loc[0] != 0:
            values[0] = self.v[self.curr_loc[0]-1, self.curr_loc[1]]
            #values[0] = self.v(self.curr_loc - [1,0])
        if self.curr_loc[0] < width-1:
            values[1] = self.v[self.curr_loc[0]+1, self.curr_loc[1]]
        if self.curr_loc[1] != 0:
            values[2] = self.v[self.curr_loc[0], self.curr_loc[1]-1]
        if self.curr_loc[1] < height-1:
            values[3] = self.v[self.curr_loc[0], self.curr_loc[1]+1]
        return values

    def vval(self,loc): return self.v[loc[0]][loc[1]]

    def TD(self,p,r,c):
        self.v[p[0]][p[1]] += alpha*(r + gamma*self.v[c[0]][c[1]] - self.v[p[0]][p[1]])


    # pick the action 
    def greedy(self):
        values = self.getValues() # get all the surrounding state values

        # if values are all the same pick a random one, otherwise do the first max
        if (sum(values) == values[0]/len(values)): return self.explore()
        else: return values.index(max(values))

    # try a random option
    def explore(self):
        return np.random.choice([0,1,2,3])

    # choose an action greedily, i.e. the max value action with probability
    # 1-epsilon, and a random action with probability epsilon
    def action(self):
        i = np.random.choice([self.greedy(),self.explore()], p=[1-epsilon,epsilon]) # pick action

        # check for boundary violation, try different action if so
        if self.curr_loc[self.actions[i][0]] == self.actions[i][1]: self.action() 
        else: # do action
            self.prev_loc = list.copy(self.curr_loc) 
            self.curr_loc[self.actions[i][0]] += self.actions[i][2]
            self.TD(self.prev_loc,0,self.curr_loc) # learn from the action taken
            if self.curr_loc in walls: # check if we entered into a wall
                self.prev_loc, self.curr_loc = list.copy(self.curr_loc), list.copy(self.prev_loc)
                self.TD(self.prev_loc,Rwall,self.curr_loc)
            elif self.curr_loc == self.end: # check for end of maze
                self.episode += 1 # end of episode
                self.curr_loc, self.prev_loc = list.copy(self.start), list.copy(end)
                self.TD(self.end,Rend,self.curr_loc)

class SARSA_agent:
    def __init__(self):
        self.prev_loc = self.curr_loc = list.copy(start) # keep track of agent's location
        self.prev_a = self.curr_a = 0 # all actions have same value to start, so start anywhere
        self.q = np.zeros([width,height,4]) # values of each state in the grid
        self.actions = [l, r, u, d]
        self.episode, self.start, self.end = 0, list.copy(start), list.copy(end)
        self.reward = 0

    def qvals(self,loc): return self.q[loc[0]][loc[1]]

    def getReward(self): pass

    def SARSA(self,p,a1,r,c,a2):
        self.q[p[0]][p[1]][a1] += alpha*(r + gamma*self.q[c[0]][c[1]][a2] - self.q[p[0]][p[1]][a1])

    # pick the action with maximum value
    def greedy(self):
        i = self.qvals(self.curr_loc) # get available actions
        values = [i[0],i[1],i[2],i[3]]
        if (sum(values) == values[0]/len(values)): return self.explore()
        else: return values.index(max(values))

    def explore(self):
        return np.random.choice([0,1,2,3])

    def action(self):
        # select an action
        i = self.curr_a = np.random.choice([self.greedy(),self.explore()], p=[1-epsilon,epsilon])

        # check for boundary violation, try different action if so
        if self.curr_loc[self.actions[i][0]] == self.actions[i][1]: 
            self.q[self.curr_loc[0]][self.curr_loc[1]][i] = -1 # avoid boundary checks
            self.action() 
        else: # do action
            # update previous state now that we have the action
            self.SARSA(self.prev_loc,self.prev_a,self.reward,self.curr_loc,self.curr_a)
            self.prev_a = self.curr_a

            # update locations
            self.prev_loc = list.copy(self.curr_loc) 
            self.curr_loc[self.actions[i][0]] += self.actions[i][2]

            if self.curr_loc in walls: # check if we entered into a wall
                self.curr_loc = list.copy(self.prev_loc) # reset location
                self.reward = Rwall # update reward
            elif self.curr_loc == self.end: # check for end of maze
                self.episode += 1 # end of episode
                self.curr_loc = list.copy(self.start) # reset to start
                self.reward = Rend
            else:
                self.reward = 0 #1/(abs(end[0]-self.curr_loc[0]) + abs(end[1]-self.curr_loc[1]))
    
if __name__ == '__main__':
    a, count, iters = SARSA_agent(), 0, 300
    steps = np.linspace(0,0,iters) # track steps taken at each episode
        
    while(a.episode < iters):
        a.action()
        count += 1
        #print(count,a.prev_loc,"->",a.curr_loc)
        if a.prev_loc == [4,5] and a.curr_loc == start: 
            print(count,a.prev_loc,"->",a.curr_loc)
            steps[a.episode-1] = count
            count = 0

    for k in range(0,4):
        for i in range(0,height):
            for j in range(0,width):
                if [i,j] in walls: state = 'w'
                elif [i,j] == end: state = 'e'
                elif [i,j] == start: state = 's'
                else: state = '[]'
                #print(state,round(a.q[i][j],2),end='\t')
                print(state,round(a.q[i][j][k],1),end='\t')
            print()
        print()

    plt.plot(np.linspace(1,iters,iters),steps)
    plt.show()