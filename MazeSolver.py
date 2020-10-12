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

alpha, epsilon, gamma = 0.05, .2, 0.9 # TODO: Find out what are good values and explain in paper
height = width = 6

# locations of the following objects on the grid
start = [0,5]
end = [5,4]
walls = [[2,2],[2,3],[3,5],[4,3],[4,4]]

min_steps = 10 # minimum number of steps from start to finish
Rend, Rwall, Rdistance = 5, -1, 0.5 # reward values
l, r, u, d = [0,0,-1], [0,width-1,1], [1,0,-1], [1,height-1,1] # coordinate, boundary check, and shift

class TD_agent:
    def __init__(self):
        self.prev_loc, self.curr_loc = None, start # keep track of agent's location
        self.v = np.zeros([width,height]) # values of each state in the grid
        self.actions = [l, r, u, d]
    
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

    def Rdist(self):
        #prev = (end[0]-self.prev_loc[0])^2 + (end[1]-self.prev_loc[1])^2
        curr = (end[0]-self.curr_loc[0])^2 + (end[1]-self.curr_loc[1])^2
        return -0.5*math.sqrt(curr)


    def TD(self):
        pi, pj, ni, nj = self.prev_loc[0], self.prev_loc[1], self.curr_loc[0], self.curr_loc[1]
        self.v[pi][pj] += alpha*(self.Rdist() + gamma*self.v[ni][nj] - self.v[pi][pj])

    # pick the action 
    def greedy(self):
        values = self.getValues() # get all the surrounding state values

        # if values are the same pick a random one, otherwise do the first max
        if (sum(values) == values[0]/len(values)): return self.explore()
        else: return values.index(max(values))

    # try a random option
    def explore(self):
        return np.random.choice([0,1,2,3])

    # choose an action greedily, i.e. the max value action with probability
    # 1-epsilon, and a random action with probability epsilon
    def action(self):
        i = np.random.choice([self.greedy(),self.explore()], p=[1-epsilon,epsilon]) # pick an action

        # check for boundary violation, try different action if so
        if self.curr_loc[self.actions[i][0]] == self.actions[i][1]: self.action() 
        else: # do action
            self.prev_loc = list.copy(self.curr_loc) 
            self.curr_loc[self.actions[i][0]] += self.actions[i][2]
            self.TD() # learn from the action taken
    
if __name__ == '__main__':
    a, count, episodes = TD_agent(), 11, 0
    
    while(episodes < 100):
        count, episodes = 0, episodes + 1
        a.curr_loc = [1,5]
        while(a.curr_loc != end):
            a.action()
            count += 1
        print(count,a.prev_loc,"->",a.curr_loc)
        
    print("ended at:",a.curr_loc)
    print("number of episodes:",episodes)
    print(a.v)
    

 # TODO: idea to get rid of different actions	 
 # 
 # generic move has [coordinate,movement,check] where coordinate is	 # generic move has [coordinate,movement,check] where coordinate is
 # which axis it affects (x=0 or y=1), movement is direction (+/- 1),	 # which axis it affects (x=0 or y=1), movement is direction (+/- 1),
 # and check is value to do boundary check on (0, width, height) 	 # and check is value to do boundary check on (0, width, height)
