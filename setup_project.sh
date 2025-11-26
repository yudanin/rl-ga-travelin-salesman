#!/bin/bash

# TSP RL-GA Project Directory Setup Script
# Run this script to create the complete project structure

echo "Creating TSP_RL_GA_Solver project structure..."


# Create main directories
mkdir -p data/problem_instances
mkdir -p algorithms/reinforcement_learning
mkdir -p algorithms/genetic_algorithm
mkdir -p algorithms/hybrid_solver
mkdir -p environments
mkdir -p evaluation
mkdir -p experiments
mkdir -p utils
mkdir -p config
mkdir -p results
mkdir -p logs

# Create Python files with basic structure
echo "Creating Python files..."

# Data module files
cat > data/__init__.py << 'EOF'
"""
Data management module for TSP instances and benchmarks.
"""
EOF

cat > data/tsplib_loader.py << 'EOF'
"""
TSPLIB instance loader and parser for TSP problems.
"""

class TSPLIBLoader:
    """Load and parse TSPLIB format files."""
    
    def __init__(self):
        pass
    
    def load_instance(self, filepath):
        """Load a TSP instance from file."""
        pass
    
    def parse_coordinates(self, content):
        """Parse city coordinates from TSPLIB format."""
        pass
    
    def get_optimal_solution(self, instance_name):
        """Retrieve optimal solution if available."""
        pass
EOF

# Algorithm module files
cat > algorithms/__init__.py << 'EOF'
"""
Algorithm implementations for TSP solving.
"""
EOF

cat > algorithms/reinforcement_learning/__init__.py << 'EOF'
"""
Reinforcement learning algorithms for TSP.
"""
EOF

cat > algorithms/reinforcement_learning/rl_agent.py << 'EOF'
"""
RL agent implementation using Ray RLlib for TSP solving.
"""

class TSPRLAgent:
    """Reinforcement learning agent for TSP route construction."""
    
    def __init__(self, config):
        self.config = config
    
    def train(self, environment, num_iterations):
        """Train the RL agent."""
        pass
    
    def get_policy(self):
        """Return trained policy."""
        pass
    
    def construct_route(self, tsp_instance):
        """Construct route using trained policy."""
        pass
EOF

cat > algorithms/genetic_algorithm/__init__.py << 'EOF'
"""
Genetic algorithm implementation for TSP.
"""
EOF

cat > algorithms/genetic_algorithm/ga_solver.py << 'EOF'
"""
Genetic algorithm implementation for TSP optimization.
"""

class TSPGeneticAlgorithm:
    """Genetic algorithm solver for TSP."""
    
    def __init__(self, population_size, mutation_rate, selection_method):
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.selection_method = selection_method
    
    def initialize_population(self, tsp_instance, initial_routes=None):
        """Initialize population with random or provided routes."""
        pass
    
    def roulette_wheel_selection(self, population, fitness_scores):
        """Roulette wheel selection method."""
        pass
    
    def elitist_selection(self, population, fitness_scores):
        """Elitist selection method."""
        pass
    
    def crossover(self, parent1, parent2):
        """Crossover operation for TSP routes."""
        pass
    
    def mutate(self, route):
        """Mutation operation for TSP routes."""
        pass
    
    def evolve(self, tsp_instance, generations):
        """Main evolution loop."""
        pass
EOF

cat > algorithms/hybrid_solver/__init__.py << 'EOF'
"""
Hybrid RL-GA solver combining both approaches.
"""
EOF

cat > algorithms/hybrid_solver/hybrid_solver.py << 'EOF'
"""
Hybrid solver combining RL and GA approaches.
"""

class HybridRLGASolver:
    """Hybrid solver combining reinforcement learning and genetic algorithms."""
    
    def __init__(self, rl_agent, ga_solver, integration_strategy):
        self.rl_agent = rl_agent
        self.ga_solver = ga_solver
        self.integration_strategy = integration_strategy
    
    def sequential_solve(self, tsp_instance):
        """Sequential RL then GA approach."""
        pass
    
    def parallel_solve(self, tsp_instance):
        """Parallel RL and GA with solution sharing."""
        pass
    
    def adaptive_solve(self, tsp_instance):
        """Adaptive switching between RL and GA."""
        pass
EOF

# Environment files
cat > environments/__init__.py << 'EOF'
"""
Environment definitions for RL training.
"""
EOF

cat > environments/tsp_environment.py << 'EOF'
"""
TSP environment for reinforcement learning training.
"""
import gym
from gym import spaces

class TSPEnvironment(gym.Env):
    """Custom TSP environment for RL agent training."""
    
    def __init__(self, tsp_instance):
        super(TSPEnvironment, self).__init__()
        self.tsp_instance = tsp_instance
        self.num_cities = len(tsp_instance)
        
        # Define action and observation spaces
        self.action_space = spaces.Discrete(self.num_cities)
        self.observation_space = spaces.Box(
            low=0, high=1, shape=(self.num_cities * 3,), dtype=float
        )
    
    def reset(self):
        """Reset environment to initial state."""
        pass
    
    def step(self, action):
        """Execute action and return new state, reward, done, info."""
        pass
    
    def render(self, mode='human'):
        """Render the environment."""
        pass
    
    def calculate_reward(self, action):
        """Calculate reward for taking an action."""
        pass
EOF

# Evaluation files
cat > evaluation/__init__.py << 'EOF'
"""
Evaluation and metrics calculation module.
"""
EOF

cat > evaluation/metrics_evaluator.py << 'EOF'
"""
Performance metrics and evaluation tools.
"""

class MetricsEvaluator:
    """Evaluate algorithm performance using various metrics."""
    
    def __init__(self):
        pass
    
    def calculate_route_length(self, route, distance_matrix):
        """Calculate total route length."""
        pass
    
    def gap_from_optimal(self, found_solution, optimal_solution):
        """Calculate percentage gap from optimal solution."""
        pass
    
    def convergence_analysis(self, solution_history):
        """Analyze convergence behavior."""
        pass
    
    def statistical_comparison(self, results1, results2):
        """Statistical comparison between algorithm results."""
        pass
    
    def generate_performance_report(self, experiment_results):
        """Generate comprehensive performance report."""
        pass
EOF

# Experiments files
cat > experiments/__init__.py << 'EOF'
"""
Experiment runner and configuration management.
"""
EOF

cat > experiments/experiment_runner.py << 'EOF'
"""
Experiment orchestration and execution.
"""

class ExperimentRunner:
    """Run and manage TSP algorithm experiments."""
    
    def __init__(self, config):
        self.config = config
    
    def run_baseline_experiments(self):
        """Run baseline experiments to replicate Ruan et al. results."""
        pass
    
    def run_enhanced_experiments(self):
        """Run experiments with proposed enhancements."""
        pass
    
    def parameter_sweep(self, parameter_ranges):
        """Run parameter sweep experiments."""
        pass
    
    def collect_results(self):
        """Collect and organize experimental results."""
        pass
EOF

# Utils files
cat > utils/__init__.py << 'EOF'
"""
Utility functions and helper modules.
"""
EOF

cat > utils/visualization.py << 'EOF'
"""
Visualization and plotting utilities for TSP results.
"""
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

class TSPVisualizer:
    """Visualization tools for TSP experiments and results."""
    
    def __init__(self, style='seaborn'):
        plt.style.use(style)
        sns.set_palette("husl")
    
    def plot_route(self, cities, route, title="TSP Route"):
        """Plot TSP route on 2D city layout."""
        pass
    
    def plot_convergence(self, solution_history, title="Convergence Analysis"):
        """Plot algorithm convergence over iterations."""
        pass
    
    def plot_performance_comparison(self, results_dict, metric="route_length"):
        """Compare performance across different algorithms."""
        pass
    
    def plot_parameter_sensitivity(self, parameter_results):
        """Visualize parameter sensitivity analysis."""
        pass
    
    def plot_computation_time_analysis(self, timing_results):
        """Analyze and plot computation time comparisons."""
        pass
    
    def create_performance_heatmap(self, results_matrix, x_labels, y_labels):
        """Create heatmap for parameter combinations."""
        pass
    
    def animate_route_construction(self, construction_steps):
        """Create animation of route construction process."""
        pass
    
    def generate_publication_figures(self, experiment_results):
        """Generate publication-ready figures."""
        pass
EOF

cat > utils/config.py << 'EOF'
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
EOF

# Create additional utility files
cat > requirements.txt << 'EOF'
# Core dependencies
numpy>=1.21.0
scipy>=1.7.0
matplotlib>=3.4.0
seaborn>=0.11.0
pandas>=1.3.0

# RL dependencies
ray[rllib]>=2.0.0
gym>=0.21.0
torch>=1.9.0

# Optimization dependencies
networkx>=2.6.0
scikit-learn>=1.0.0

# Development dependencies
pytest>=6.2.0
jupyter>=1.0.0
black>=21.0.0
flake8>=3.9.0
EOF

cat > README.md << 'EOF'
# TSP RL-GA Solver

Hybrid reinforcement learning and genetic algorithm approach for solving the Asymmetric Traveling Salesman Problem (ATSP).

## Project Structure

- `data/`: TSPLIB instance loading and management
- `algorithms/`: RL, GA, and hybrid algorithm implementations  
- `environments/`: RL environment definitions
- `evaluation/`: Performance metrics and analysis
- `experiments/`: Experiment orchestration
- `utils/`: Visualization and configuration utilities

## Installation

```bash
pip install -r requirements.txt
```

## Usage

```python
from experiments.experiment_runner import ExperimentRunner
from utils.config import Config

config = Config()
runner = ExperimentRunner(config)
runner.run_baseline_experiments()
```

## Experiments

This implementation extends the work of Ruan et al. (2024) with:
- Elitist selection method for GA
- Variable population sizes (20, 40, 600)
- Higher mutation probabilities (>0.01)
- Extended TSPLIB benchmark instances
EOF

cat > .gitignore << 'EOF'
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg

# Results and logs
results/
logs/
*.log

# Data files
data/problem_instances/*.tsp
data/problem_instances/*.opt
data/problem_instances/*.tour

# Jupyter
.ipynb_checkpoints

# IDE
.vscode/
.idea/
*.swp
*.swo

# OS
.DS_Store
Thumbs.db
EOF

echo "✅ Project structure created successfully!"
echo ""
echo "📁 Directory structure:"
find TSP_RL_GA_Solver -type d | sort
echo ""
echo "📄 Python files created:"
find TSP_RL_GA_Solver -name "*.py" | sort
echo ""
echo "🚀 To get started:"
echo "   cd TSP_RL_GA_Solver"
echo "   pip install -r requirements.txt"
echo "   # Add your TSPLIB instances to data/problem_instances/"
echo ""
echo "Project setup complete! 🎉"
