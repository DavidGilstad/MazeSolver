import numpy as np

#
# Maze Solver
#
# Machine Learning project on reinforcement learning by teaching
# an agent how to go through a maze in optimal steps without running
# into objects along the way
#
#	authors: David Gilstad, Ian Schwind
#	created: October 9, 2020
#
alpha, epsilon, gamma = 0.05, 0.1, 0.9 # TODO: Find out what are good values and explain in paper
height = width = 6

# locations of the following objects on the grid
start = [0,5]
end = [5,4]
walls = [[2,2],[2,3],[3,5],[4,3],[4,4]]

min_steps = 10 # minimum number of steps from start to finish
Rend, Rwall, Rdistance = 5, -1, 0.5 # reward values

class TD_agent:
    def __init__(self):
        self.curr_loc = start # keep track of agent's location
        self.v_values = np.zeros(width,height) # values of each state in the grid
        self.actions = [self.left, self.right, self.up, self.down]
    
    # check what value is at state of next move
    def getleft(self):
        if self.curr_loc[0] == 0: return Rwall
        else: return self.v_values[self.curr_loc[0]-1, self.curr_loc[1]]

    def getright(self):
        if self.curr_loc[0] == width: return Rwall
        else: return self.v_values[self.curr_loc[0]+1, self.curr_loc[1]]

    def getup(self):
        if self.curr_loc[1] == height: return Rwall
        else: return self.v_values[self.curr_loc[0], self.curr_loc[1]+1]

    def getdown(self):
        if self.curr_loc[1] == 0: return Rwall
        else: return self.v_values[self.curr_loc[0], self.curr_loc[1]-1]

    # movements: move to the selected state, reward the start state
    # based on 
    def left(self):
        if self.curr_loc[0] == 0: return 

    def right(self):
        pass

    def up(self):
        pass

    def down(self):
        pass

    # determines the probability of going into the next state based on the value
    # at that state already. gives higher probability to higher value states.
    def action_probs(self): # TODO: explain epsilon value choice
        probs = [self.getleft(),self.getright(),self.getup(),self.getdown()]
        probs = probs - min(probs) + epsilon
        return probs / probs.sum()

    # pick the action 
    def greedy(self):
        values = [self.getleft(),self.getright(),self.getup(),self.getdown()]
        return self.actions[values.index(max(values))]

    # try a random option
    def explore(self):
        return self.actions[np.random.choice([0,1,2,3])]

    # choose an action greedily, i.e. the max value action with probability
    # 1-epsilon, and a random action with probability epsilon
    def action(self):
        np.random.choice([self.greedy(),self.explore()], p=[1-epsilon,epsilon])()
    

# Really cool method!
# np.random.choice picks name of method from actions using the given
# probabilities, then the extra () at the end uses the name to call
# that method!
# reward, action_idx = np.random.choice(self.actions, p=action_probs)()
#
#values = self.getValues()
        #return self.actions[values.index(max(values))]