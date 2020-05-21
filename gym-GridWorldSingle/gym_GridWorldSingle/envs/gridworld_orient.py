'''
    January 2020
    Implementation of the gridworld domains used by Norm Ferns.
    
    Actions: forward, rotate
    Orientations: up, right, down, left
'''
import gym
from gym import spaces
import numpy as np
from random import choice

class GridWorldOrient(gym.Env):
  metadata = {'render.modes': ['human']}

  def __init__(self, grid_width = 3, augmented = False, q_values = [], 
               modified_rewards = False):
      
    '''
    augmented: if true, then a copy of each non-terminal state is made.
               Each copy has the same Q*-values as the original state but 
               different transition probabilities.
    q-values: needed if augmented = True to set the rewards for the state copies
              (dim.: num_orig_states x num_actions)
    modified_rewards: modified rewards
    '''
    self.reward_range = (0, 1)
    self.action_space = spaces.Discrete(2) # forward, rotate
    
    self.grid_width = grid_width
    self.augmented = augmented
    self.q_values = q_values
    self.modified_rewards = modified_rewards
    
    # number of non-copied states
    self.num_true_states = grid_width * grid_width * 4
    
    if not augmented:
        self.observation_space = spaces.Discrete(self.num_true_states)
    else:
        # duplicate each state but the goal states and add another terminal state
        self.observation_space = spaces.Discrete(self.num_true_states * 2 - 3)
    
    self.MAX_STEPS = 100 # max. number of steps per episodes
    self.num_steps = 0 # steps taken so far

    gridworld = np.arange(
            self.grid_width * self.grid_width
            ).reshape((self.grid_width, self.grid_width))
    
    # types of states
    self.TERMINAL_STATES = [int(np.floor((self.grid_width * self.grid_width) / 2)) * 4 + i for i in range(4)]
    if augmented:
        self.TERMINAL_STATES.append(self.observation_space.n - 1)
    self.NON_TERMINAL_STATES = np.arange(0, self.observation_space.n)
    self.NON_TERMINAL_STATES = [i for i in self.NON_TERMINAL_STATES if i not in self.TERMINAL_STATES]
    
    self.ORIENT_TO_MOVE = [(-1, 0), (0, 1), (1, 0), (0, -1)]
    
    # state transition matrix
    self.P = np.zeros((self.action_space.n,
                          self.observation_space.n,
                          self.observation_space.n))
    
    # any action taken in terminal state has no effect
    for i in self.TERMINAL_STATES:
        self.P[:, i, i] = 1
    
    # start state
    self.state = choice(self.NON_TERMINAL_STATES)
    
    # transition probabilities for rotating
    if not self.augmented:
        for s in self.NON_TERMINAL_STATES:
            s_prime = s + 1
            if np.floor(s / 4) != np.floor(s_prime/4):
                s_prime = np.floor(s/4) * 4
            self.P[1, int(s), int(s_prime)] = 1
        
    else:
        for s in self.NON_TERMINAL_STATES:
            if not s > self.num_true_states - 1:
                s_prime = s + 1
                if np.floor(s / 4) != np.floor(s_prime/4):
                    s_prime = np.floor(s/4) * 4
                self.P[1, int(s), int(s_prime)] = 1
        
    # transition probabilities for moving for original states
    for s in gridworld.flat:
        row, col = np.argwhere(gridworld == s)[0]
        for orient_index, orient in enumerate(self.ORIENT_TO_MOVE):
            next_row = max(0, min(row + orient[0], self.grid_width - 1))
            next_col = max(0, min(col + orient[1], self.grid_width - 1))
            s_prime = gridworld[next_row, next_col]
            new_s = s * 4 + orient_index
            if not new_s in self.TERMINAL_STATES:
                self.P[0, new_s, s_prime * 4 + orient_index] = 1
    
    # transition probabilities for state copies
    if self.augmented:
        for s in range(self.num_true_states, self.observation_space.n - 1):
            index = self._get_index_augmented_state(s)
            opt_act = np.argmax(q_values[index])
            next_state_opt_act = int(np.where(self.P[opt_act, index] == 1)[0][0])
            self.P[opt_act, s, next_state_opt_act] = 1 # maintain trans. prob. for optimal action
            self.P[(opt_act + 1) % 2, s, self.observation_space.n - 1] = 1 # go to new terminal states for non-optimal action
    
    # rewards for arriving in a certain state
    self.R = np.full((self.action_space.n,
                         self.observation_space.n), 0)
    
    # rewards for arriving in goal states
    for i in self.TERMINAL_STATES:
        self.R[:, i] = 1
        
    # change some immediate rewards
    if self.modified_rewards:
        for i in [1, int(self.grid_width * 4 - 2), 
                  int(self.grid_width * self.grid_width - 1), 
                  int(self.grid_width * 4 * (self.grid_width - 1))]:
            self.R[0, i] = -1
    
    # reward for arriving in new terminal state is 0
    if augmented:
        self.R[:, self.observation_space.n - 1] = 0
        
    self.trans = [[np.where(self.P[i][j] == 1)[0][0] for i in range(self.action_space.n)] for j in range(self.observation_space.n)]    
    
  def step(self, action):
    orig_state = self.state
    self.state = self.trans[self.state][action]
    self.num_steps += 1
    done = (self.state in self.TERMINAL_STATES) or (self.num_steps == self.MAX_STEPS - 1)
    
    # current state is an original state
    if not (self.augmented and orig_state >= self.num_true_states):
        return self.state, self.R[action, self.state], done , ""
    
    # state is the new terminal state
    elif orig_state == self.observation_space.n - 1:
        return self.state, self.R[action, self.state], done, ""
    
    # current state is a state copy
    else:
        index = self._get_index_augmented_state(orig_state)
        opt_act = np.argmax(self.q_values[index])
        if action == opt_act:
            return self.state, self.R[action, self.state], done , ""
        else:
            # non-optimal action in augmented state terminates episdoe
            return self.state, self.q_values[index, action], True, ""
      
  def reset(self):
    self.state = choice(self.NON_TERMINAL_STATES)
    self.num_steps = 0
    return self.state

  def render(self, mode ='human', close = False):
    '''
    A digit marks the current agent grid position. The digit
    denotes the agent orientation.
    If the current state is augmented, the corresponding original state is marked.
    '''
    if self.augmented:
        state = self._get_index_augmented_state(self.state)
    else:
        state = self.state
        
    for i in range(int((np.floor(state/4)))):
        if i % self.grid_width == self.grid_width - 1:
            print("x")
        else:
            print("x", end = "")
    if int((np.floor(state/4))) % self.grid_width == self.grid_width - 1:
        print(state % 4)
    else:
        print(state % 4, end = "")
    for i in range(int((np.floor(state/4))) + 1, self.grid_width * self.grid_width):
        if i % self.grid_width == self.grid_width - 1:
            print("x")
        else:
            print("x", end = "")
    
  def _get_index_augmented_state(self, s):
    '''
    Returns the index of the original state an augmented state
    corresponds to.
    '''
    if s < self.num_true_states: # not an augmented state
        return s
    
    if self.augmented and s == self.observation_space.n - 1: # new terminal state
        return s
    
    index = s - self.num_true_states
    if index >= self.TERMINAL_STATES[0]: # goal states are not duplicated
        index += 4
    return int(index)
    