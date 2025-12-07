# Implementing hybrid algorithms that combine reinforcement learning (RL) with genetic algorithms (GA) to solve the Asymmetric Traveling Salesman Problem (ATSP)

by Connor Russell Plaks (cplaks), Ari Vauhkonen (arivauhkonen29-cmd), Michael Yudanin (yudanin)

Implementing the research in
Ruan, Y., Cai, W., Wang, J.: Combining reinforcement learning algorithm and  genetic algorithm to solve the traveling salesman  problem. J. Eng. 2024, e12393 (2024).
https://doi.org/10.1049/tje2.12393

with the following enhancements:

* In the genetic algorithm, in addition to the roulette wheel selection method used
by Ruan et al., we will also use the elitist selection method where the
best-performing individuals are selected.
* Experiment with the lower and higher numbers of optimal routes used as the
initial population for the genetic algorithm, specifically 20 and 600, and compare
the results with Ruan et al.’s 40 routes.
* Experimenting with the mutation probability higher than 0.01 in Ruan et al.

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
