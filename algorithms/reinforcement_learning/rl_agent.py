"""
RL agent implementation for TSP solving.
"""

import numpy as np

class TSPRLAgent:
    """Reinforcement learning agent for TSP route construction."""
    
    def __init__(self, data, config):
        # extract the test scenerio data
        self.test_data = data
        self.distance_matrix = data['distance_matrix']
        self.name = data['name']
        self.dimension = data['dimension'] 
        self.coordinates = data['coordinates']
        self.optimal_value = data['optimal_value']
        
        # pull out the RL hyper parameters
        self.config = config
        self.alpha = config.get('alpha', 0.01)  # Learning rate
        self.gamma = config.get('gamma', 0.15)  # Discount factor
        self.episodes = config.get('episodes', 1000)
        self.reward_type = config.get('reward_type', 1)
        self.epsilon_greedy_type = config.get('epsilon_greedy_type',1)
        self.method = config.get('method', 'double_q_learning')
        #print(f"Initialized TSPRLAgent for instance {self.name} with method {self.method}")

        # initialize our Q tables
        self.QA = np.zeros([self.dimension, self.dimension]) # Q-value table
        self.QB = np.zeros([self.dimension, self.dimension]) # Q-value table
        self.done = False
        
        # # set random seed
        # random_seed = 42
        # np.random.seed(random_seed)
    
    def action_space(self, visited):
        """Return the action space (list of cities that have not been visited)."""
        return [coord for coord in range(self.dimension) if coord not in visited]
    def reward(self, current_city, action):
        """Calculate reward for moving from current_city to next_city."""
        distance = self.distance_matrix[current_city, action]
        
        # return the reward type specified in the config (either inverse of distance or negative squared distance)
        if self.reward_type == 1:
            return 1 / distance
        if self.reward_type == 2:
            return -distance**2
            
    def step(self, current_city, action, visited):
        """Take a step in the environment."""
        
        # compute the reward at our current state and action
        reward = self.reward(current_city, action)
        
        # define the next state
        next_city = action
        
        # compute our list of visited states at from the action
        next_visited = visited | {action}
        
        # if we are at the last city, return to the initial city
        if len(next_visited) == self.dimension:
            # reward += self.reward(next_city, 0)
            self.done= True
        return next_city, reward, next_visited
    def epsilon_greedy_policy(self, epsilon, current_city, visited):
        """Select next city using epsilon-greedy policy."""
        
        # compute the action space of remaining cities from the ones we've visited
        action_space = self.action_space(visited)
        
        if np.random.rand() < epsilon:
            # Explore - choose a random city
            next_city = np.random.choice(action_space)
        else:
            # Exploit - choose the best known city
            next_city = max(action_space, key = lambda a: 0.5* (self.QA[current_city, a] + self.QB[current_city, a]))
        return next_city
    def compute_epsilon(self, episode):
        """Compute our epsilon for epsilon greedy double Q-learning"""
        
        if self.epsilon_greedy_type == 1:
            return 1 - episode / self.episodes
        if self.epsilon_greedy_type == 2:
            return 1 - (episode / self.episodes)**6
        else:
            return 1 - 0.1 * (np.floor(episode/self.episodes))
        
    def train_episode(self, epsiode_num):
        """Generate RL epsiode the RL agent."""
        
        # initialize environment
        current_city = 0
        visited = frozenset({current_city})
        self.done = False
        epsilon = self.compute_epsilon(epsiode_num)
        
        # initialize Q value for all final states
        for state in range(1,self.dimension):
            self.QA[state, 0] = self.reward(state, 0)
            self.QB[state, 0] = self.reward(state, 0)

        # loop until episode is done
        while not self.done:
            
            # pick action using epsilon-greedy policy
            action = self.epsilon_greedy_policy(epsilon, current_city, visited)
            
            # proceed to next state
            next_city, reward, next_visited = self.step(current_city, action, visited)
            
            # update Q-values
            if len(next_visited) == self.dimension:
                
                if self.method == 'q_learning':
                    self.QA[(current_city, action)] = self.QA[current_city, action] + self.alpha * (reward + self.gamma * (self.QA[next_city, 0]) - self.QA[current_city, action])
                    self.QB = self.QA
                elif self.method == 'double_q_learning':
                    self.QA[(current_city, action)] = self.QA[current_city, action] + self.alpha * (reward + self.gamma * (self.QB[next_city, 0]) - self.QA[current_city, action])
                    # else:
                    self.QB[(current_city, action)] = self.QB[current_city, action] + self.alpha * (reward + self.gamma * (self.QA[next_city, 0]) - self.QB[current_city, action])
                    self.done = True
                elif self.method == 'sarsa':
                    self.QA[(current_city, action)] = self.QA[current_city, action] + self.alpha * (reward + self.gamma * (self.QA[next_city, 0]) - self.QA[current_city, action])
                    self.QB = self.QA
            else:
                if self.method == 'q_learning':
                    self.QA[(current_city, action)] = self.QA[current_city, action] + self.alpha * (reward + self.gamma * max(self.QA[next_city, a] for a in self.action_space(next_visited)) - self.QA[current_city, action])
                    self.QB = self.QA
                elif self.method == 'double_q_learning':
                    self.QA[(current_city, action)] = self.QA[current_city, action] + self.alpha * (reward + self.gamma * max(self.QB[next_city, a] for a in self.action_space(next_visited)) - self.QA[current_city, action])
                    self.QB[(current_city, action)] = self.QB[current_city, action] + self.alpha * (reward + self.gamma * max(self.QA[next_city, a] for a in self.action_space(next_visited)) - self.QB[current_city, action])
                elif self.method == 'sarsa':
                    next_action = self.epsilon_greedy_policy(epsilon, next_city, next_visited)
                    self.QA[(current_city, action)] = self.QA[current_city, action] + self.alpha * (reward + self.gamma * (self.QA[next_city, next_action]) - self.QA[current_city, action])
                    self.QB = self.QA
            # increment state
            current_city = next_city
            visited = next_visited

            
    def train(self):
        """Train the RL agent."""
        
        # perform Q-learning for the number of episodes specified
        for episode in range(self.episodes):
            self.train_episode(episode)
    
    
    def optimal_route(self):
        """Construct optimal route using trained policy."""
        
        # initialize enviorment
        current_city = 0
        visited = frozenset({current_city})
        route = [current_city]
        self.done = False
        cost = 0
        
        # Loop through the states selecting the action that maximizes our learned Q functions
        while not self.done:
            action_space = self.action_space(visited)
            next_city = max(action_space, key = lambda a: 0.5* (self.QA[current_city, a] + self.QB[current_city, a]))
            route.append(next_city)
            visited = visited | {next_city}
            cost += self.distance_matrix[current_city, next_city]
            current_city = next_city
            if len(visited) == self.dimension:
                route.append(0)  # Return to starting city
                cost += self.distance_matrix[next_city, 0]
                self.done = True
                
        return cost,route
        

