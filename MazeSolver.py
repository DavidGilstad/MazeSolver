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

alpha, epsilon, gamma = 0.1, .1, 0.9 # TODO: Find out what are good values and explain in paper
height = width = 6

# locations of the following objects on the grid
start = [5,0]
end = [4,5]
walls = [[2,2],[3,2],[5,3],[3,4],[4,4]]

min_steps = 10 # minimum number of steps from start to finish
Rend, Rwall, Rdistance = 10, -0.01, 0 # reward values
l, r, u, d = [0,0,-1], [0,width-1,1], [1,0,-1], [1,height-1,1] # coordinate, boundary check, and shift

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

    def Rdist(self):
        #prev = (end[0]-self.prev_loc[0])^2 + (end[1]-self.prev_loc[1])^2
        curr = abs(end[0]-self.curr_loc[0])^2 + abs(end[1]-self.curr_loc[1])^2
        return Rdistance*curr

    #def TD(self):
    #    pi, pj, ni, nj = self.prev_loc[0], self.prev_loc[1], self.curr_loc[0], self.curr_loc[1]
    #    self.v[pi][pj] += alpha*(self.Rdist() + gamma*self.v[ni][nj] - self.v[pi][pj])

    def TD(self,p,r,c):
        self.v[p[0]][p[1]] += alpha*(r + gamma*self.v[c[0]][c[1]] - self.v[p[0]][p[1]])


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
            self.TD(self.prev_loc,self.Rdist(),self.curr_loc) # learn from the action taken
            if self.curr_loc in walls: # check if we entered into a wall
                self.prev_loc, self.curr_loc = list.copy(self.curr_loc), list.copy(self.prev_loc)
                self.TD(self.prev_loc,Rwall,self.curr_loc)
            elif self.curr_loc == self.end: # check for end of maze
                self.episode += 1 # end of episode
                self.curr_loc, self.prev_loc = list.copy(self.start), list.copy(end)
                self.TD(self.end,Rend,self.curr_loc)
        
    
if __name__ == '__main__':
    a, count, iters = TD_agent(), 0, 300
    
    steps = np.linspace(0,0,iters)
        
    while(a.episode < iters):
        a.action()
        count += 1
        if a.prev_loc == end or a.episode > iters*.8:
            steps[a.episode-1] = count
            print(count,a.prev_loc,"->",a.curr_loc)
        if a.prev_loc == end: count = 0

    for i in range(0,height):
        for j in range(0,width):
            if [i,j] in walls:
                print('w',round(a.v[i][j],2),end='\t')
            elif [i,j] == end:
                print('e',round(a.v[i][j],2),end='\t')
            elif [i,j] == start:
                print('s',round(a.v[i][j],2),end='\t')
            else: print(0,round(a.v[i][j],2),end='\t')
        print()

    plt.plot(np.linspace(1,iters,iters),steps)
    plt.show()
    

 # TODO: idea to get rid of different actions	 
 # 
 # generic move has [coordinate,movement,check] where coordinate is	 # generic move has [coordinate,movement,check] where coordinate is
 # which axis it affects (x=0 or y=1), movement is direction (+/- 1),	 # which axis it affects (x=0 or y=1), movement is direction (+/- 1),
 # and check is value to do boundary check on (0, width, height) 	 # and check is value to do boundary check on (0, width, height)
