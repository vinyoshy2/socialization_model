import itertools
import random
import subprocess
import sys
from pathlib import Path

PARAM_GRID = {
    "topics": [100],
    "iterations": [2000],
    "warmup_steps": [500],
    "alpha_sum_topics": [0.01, 0.1, 1.0],
    "alpha_sum_vocab": [0.01, 0.1, 1.0],
    "alpha_sum_edges": [0.01, 0.1, 1.0],
}

N_RANDOM_SAMPLES = int(sys.argv[1])
START_MONTH = 2
END_MONTH = 2

def main():
    values = [PARAM_GRID[k] for k in PARAM_GRID.keys()]
    all_combinations = list(itertools.product(*values))

    n_samples = min(N_RANDOM_SAMPLES, len(all_combinations))
    sampled_combinations = random.sample(all_combinations, n_samples)

    for idx, params in enumerate(sampled_combinations):
        param_dict = dict(zip(PARAM_GRID.keys(), params))
        print(", ".join([f'{param}={value}' for param, value in param_dict.items()]))

        for month in range(START_MONTH, END_MONTH + 1):
            input_dir = f"data/{month}"
            result_dir = f"data/results/priors/k{param_dict['topics']}_i{param_dict['iterations']}_w{param_dict['warmup_steps']}_aT{param_dict['alpha_sum_topics']}_aV{param_dict['alpha_sum_vocab']}_aE{param_dict['alpha_sum_edges']}/{month}"
            Path(result_dir).mkdir(parents=True, exist_ok=True)

            cmd = [
                "./socialization",
                input_dir,
                result_dir,
                "--topics", str(param_dict['topics']),
                "--iters", str(param_dict['iterations']),
                "--warmup", str(param_dict['warmup_steps']),
                "--alpha-topics", str(param_dict['alpha_sum_topics']),
                "--alpha-vocab", str(param_dict['alpha_sum_vocab']),
                "--alpha-edges", str(param_dict['alpha_sum_edges']),
            ]
            
            print(f"\tRunning month {month}...")
            subprocess.run(cmd, check=True)

if __name__ == "__main__":
    main()
