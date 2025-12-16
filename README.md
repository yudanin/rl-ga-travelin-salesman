# Implementing hybrid algorithms that combine reinforcement learning (RL) with genetic algorithms (GA) to solve the Asymmetric Traveling Salesman Problem (ATSP)

by Connor Russell Plaks (cplaks), Ari Vauhkonen (arivauhkonen29-cmd), Michael Yudanin (yudanin)

Implementing the research in
Ruan, Y., Cai, W., Wang, J.: Combining reinforcement learning algorithm and  genetic algorithm to solve the traveling salesman  problem. J. Eng. 2024, e12393 (2024).
https://doi.org/10.1049/tje2.12393

with the following enhancements:

* In the Rl algorithm, using SARSA is addition to Q-learning and Double Q-learning used by Ruan et al.
* In the genetic algorithm, using Elitist selection method, Tournament variety, in addition to the Roulette Wheel selection method used by Ruan et al.
* Reporting all the combinations of parameters

## Project Structure

- `algorithms/`: RL, GA, and hybrid algorithm implementations  
- `data/`: TSPLIB instance loading and management
- `environments/`: RL environment definitions
- `experiments/`: Experiment orchestration and results
- `outputs/`: temporary outputs
- `results/`: summary of results, graphs
- `utils/`: Visualization and configuration utilities

## Installation

```bash
pip install -r requirements.txt
```

## Usage

To run all experiments (GA-only, RL-only, and RL+GA hybrid) across all TSP instances:
```bash
cd experiments
python rl_ga_experiment_runner.py --data-dir ../data/problem_instances
```


