"""
Configuration management for experiments.
"""

class Config:
    """Configuration settings for TSP experiments."""
    
    # RL Configuration
    RL_ALGORITHM = "PPO"
    RL_LEARNING_RATE = 0.001
    RL_TRAINING_ITERATIONS = 1000
    
    # GA Configuration
    GA_POPULATION_SIZES = [20, 40, 600]
    GA_MUTATION_RATES = [0.01, 0.05, 0.1]
    GA_SELECTION_METHODS = ["roulette_wheel", "elitist"]
    GA_GENERATIONS = 500
    
    # Hybrid Configuration
    INTEGRATION_STRATEGIES = ["sequential", "parallel", "adaptive"]
    
    # Experiment Configuration
    NUM_RUNS = 10
    RANDOM_SEED = 42
    
    # TSPLIB Instances
    TEST_INSTANCES = [
        "small_instance",  # <60 cities
        "medium_instance",  # Ruan et al. instances
        "large_instance"   # >99 cities
    ]
