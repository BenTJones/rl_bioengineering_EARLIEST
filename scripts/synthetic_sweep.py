
from pathlib import Path
import csv
import re
import subprocess
import sys
import numpy as np
import torch


root = Path(__file__).resolve().parents[1]
results = root / "results"
checkpoints = root / "saved_models"
results.mkdir(parents=True, exist_ok=True)
checkpoints.mkdir(parents=True, exist_ok=True)
lambdas = np.round(np.linspace(0.0, 0.15, 16), 3)
seeds = [42, 123, 456]

output_file = results / "synthetic_baseline.csv"
def run_command(args):
    result = subprocess.run(
        [sys.executable, *args],
        cwd=root,
        capture_output=True,
        text=True
    )
    if result.returncode != 0:
        print(result.stdout)
        print(result.stderr)
        raise RuntimeError("Experiment failed")
    return result.stdout


def read_metrics(text, name):
    pattern = rf"{re.escape(name)}:\s*([0-9.]+)"
    match = re.search(pattern, text)
    if not match:
        raise ValueError(f"Could not find {name} in output")
    return float(match.group(1))


#Run experiments and write the results to csv
with output_file.open("w", newline="") as file:
    writer = csv.DictWriter(
        file,
        fieldnames=[
            "lambda",
            "seed",
            "test_accuracy",
            "mean_proportion_used_percent"
        ]
    )
    writer.writeheader()

    for l in lambdas:
        for seed in seeds:
            name = f"lam_{l:g}_seed_{seed}"
            model_dir = checkpoints / name
            model_dir.mkdir(parents=True, exist_ok=True)
            common_args = [
                "--dataset", "synthetic",
                "--lam", str(l),
                "--random_seed", str(seed),
                "--model_save_path", str(model_dir) + "/"
            ]
            print(f"Training: lambda={l:.3f}, seed={seed}", flush=True)
            train_output = run_command(["train.py", *common_args])
            test_output = run_command(["test.py", *common_args])

            #Extract metrics
            accuracy = read_metrics(
                test_output, "Testing Accuracy"
            )

            proportion = read_metrics(
                test_output, "Mean proportion used"
            )

            #Saves result
            writer.writerow({
                "lambda": float(l),
                "seed": seed,
                "test_accuracy": accuracy,
                "mean_proportion_used_percent": proportion
            })
            file.flush()
            (results / f"{name}_train.txt").write_text(
                train_output, encoding="utf-8")
            (results / f"{name}_test.txt").write_text(
                test_output, encoding="utf-8")
            
            print(
                f"Completed: lambda={l:.3f}, seed={seed} | "
                f"Accuracy={accuracy:.3f} | "
                f"Proportion used={proportion:.2f}%\n",
                flush=True
            )
