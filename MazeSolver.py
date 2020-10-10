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
alpha, gamma = 0.05, 0.9 # TODO: Find out what are good values and explain in paper
height = width = 6

# locations of the following objects on the grid
start = [0,5]
end = [5,4]
walls = [[2,2],[2,3],[3,5],[4,3],[4,4]]

min_steps = 10 # minimum number of steps from start to finish
Rend, Rwall, Rdistance = 5, -1, 0.5 # reward values

