from pathlib import Path
import itertools
import random
import subprocess
import sys


def prior_search(model, param_grid, in_path, out_path, verbose=False):
    for idx, param_dict in enumerate(param_grid):
        if verbose:
            print(", ".join([f'{param}={value}' for param, value in param_dict.items()]))

        out_path = f"{out_path}/atopics_{param_dict['alpha_sum_topics']}_avocab_{param_dict['alpha_sum_vocab']}_aedges_{param_dict['alpha_edges']}"
        Path(out_path).mkdir(parents=True, exist_ok=True)

        cmd = [
            f"./{model}",
            in_path,
            out_path,
            "--topics", str(param_dict['topics']),
            "--iters", str(param_dict['iterations']),
            "--warmup", str(param_dict['warmup_steps']),
            "--alpha-topics", str(param_dict['alpha_sum_topics']),
            "--alpha-vocab", str(param_dict['alpha_sum_vocab']),
            "--alpha-edges", str(param_dict['alpha_edges']),
        ]
        
        subprocess.run(cmd, check=True)

if __name__ == "__main__":
    MODEL = sys.argv[1]
    IN_PATH = sys.argv[2]
    OUT_PATH = sys.argv[3]
    MODE = int(sys.argv[4])

    if MODE == "random":
        SAMPLES = sys.argv[5]

        PARAM_GRID = {
            "topics": [25, 50, 100, 250, 500, 1000],
            "iterations": [2000],
            "warmup_steps": [500],
            "alpha_sum_topics": [0.01, 0.1, 1.0],
            "alpha_sum_vocab": [0.01, 0.1, 1.0],
            "alpha_edges": [0.01, 0.1, 1.0],
        }

        keys = list(PARAM_GRID.keys())
        values = [PARAM_GRID[k] for k in keys]
        all_combinations = list(itertools.product(*values))
        PARAM_GRID = [dict(zip(keys, combo)) for combo in all_combinations]
        PARAM_GRID = random.sample(PARAM_GRID, k=min(int(SAMPLES), len(PARAM_GRID)))

        prior_search(MODEL, PARAM_GRID, IN_PATH, OUT_PATH, verbose=True)


    if MODE == "grid":
        PRIORS = [
            [0.1, 0.01, 0.1],
            [0.1, 1.0, 1.0],
            [1.0, 0.1, 0.01],
            [1.0, 1.0, 0.01],
            [0.01, 0.1, 0.01],
            [0.01, 1.0, 0.01],
            [0.01, 1.0, 0.1],
            [1.0, 0.1, 0.1],
            [1.0, 1.0, 0.1],
        ]

        PARAM_GRID = []
        for priors in PRIORS:
            PARAM_GRID.append({
                "topics": 100,
                "iterations": 2000,
                "warmup_steps": 500,
                "alpha_sum_topics": priors[0],
                "alpha_sum_vocab": priors[1],
                "alpha_edges": priors[2],
            })
        
        prior_search(MODEL, PARAM_GRID, IN_PATH, OUT_PATH, verbose=True)
