import numpy as np
import sys
sys.path.append('../data')
sys.path.append('../algorithms/reinforcement_learning')
from rl_agent import TSPRLAgent
from tsplib_loader import TSPLIBLoader
if __name__ == "__main__":
    loader = TSPLIBLoader(data_dir="../data/problem_instances")
    instances = ['eil51', 'berlin52', 'st70', 'eil76', 'kroA100', 'pr107']
    for instance in instances:
        print(f"\n📋 Testing {instance}:")

        data = loader.load_instance(instance)
        
        # Calculate tour length
        expected = data['optimal_value']
        agent_config = {
            'alpha': 0.01,
            'gamma': 0.01,
            'episodes': 10000,
            'instance' : instance,
            'reward_type' : 1,
            'epsilon_greedy_type' : 1
        }
        agent = TSPRLAgent(data, agent_config)
        agent.train()
        cost, constructed_route = agent.optimal_route()
        print("Cost:", cost)
        print(f"Optimal Cost:", expected)
            
