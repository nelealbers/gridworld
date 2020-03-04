'''
5x5 grid domain with goal state in the center of the grid.
'''

import gym
from gym import error, spaces, utils
from gym.utils import seeding
import numpy as np
from random import choice

class GridWorldSingle(gym.Env):

  def __init__(self):
    self.reward_range = (-1, 1)
    self.action_space = spaces.Discrete(4) # up, right, down, left
    self.observation_space = spaces.Discrete(25)
    
    # grid boundary states
    self.BOUNDARY_STATES = [0, 1, 2, 3, 4, 5, 9, 10, 14, 15, 19, 20, 21, 22, 23, 24]
    
    self.GOAL_STATE = 12

    gridworld = np.arange(
            self.observation_space.n
            ).reshape((5, 5))
    
    # state transition matrix
    self.P = np.zeros((self.action_space.n,
                          self.observation_space.n,
                          self.observation_space.n))
    
    # any action taken in terminal state has no effect
    self.P[:, self.GOAL_STATE, self.GOAL_STATE] = 1
    
    # start state
    self.state = choice(self.BOUNDARY_STATES)

    for s in gridworld.flat:
        row, col = np.argwhere(gridworld == s)[0]
        for a, d in zip(
                range(self.action_space.n),
                [(-1, 0), (0, 1), (1, 0), (0, -1)]
                ):
            next_row = max(0, min(row + d[0], 4))
            next_col = max(0, min(col + d[1], 4))
            s_prime = gridworld[next_row, next_col]
            self.P[a, s, s_prime] = 1

    # rewards
    self.R = np.full((self.action_space.n,
                         self.observation_space.n), -1)
    self.R[:, self.GOAL_STATE] = 1
    
    self.trans = [[np.where(self.P[i][j] == 1)[0][0] for i in range(4)] for j in range(25)]
        
  def step(self, action):
    self.state = self.trans[self.state][action]
    return self.state, self.R[action, self.state], self.state == self.GOAL_STATE, ""
      
  def reset(self):
    self.state = choice(self.BOUNDARY_STATES)
    return self.state

  def render(self, mode='human', close = False):
    '''
    A "O" marks the current agent position.
    '''
    for i in range(self.state):
        if i % 5 == 4:
            print("x")
        else:
            print("x", end = "")
    if self.state % 5 == 4:
        print("O")
    else:
        print("O", end = "")
    for i in range(self.state + 1, self.observation_space.n + 1):
        if (i) % 5 == 4:
            print("x")
        else:
            print("x", end = "")
    