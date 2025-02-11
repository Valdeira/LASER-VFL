import argparse
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import wandb

MAP_METHOD_TO_LABEL = {
    "laser": "LASER-VFL",
    "combinatorial": "Combinatorial",
    "local": "Local",
    "svfl": "Standard VFL",
    "ensemble": "Ensemble",
    "plug": "PlugVFL",
}


def get_result(project_name, run_names, metric_name, api):
    """Fetch metric/runtimes from multiple runs and return aggregated stats."""
    metric_vals, runtimes, num_clients = [], [], []
    for run_name in run_names:
        runs = api.runs(f"{project_name}", filters={"display_name": run_name})
        if not runs:
            print(f"No run found with name {run_name}. Skipping...")
            continue
        run = runs[0]
        val = run.summary.get(metric_name)
        if val is not None:
            metric_vals.append(val)
        else:
            print(f"Metric '{metric_name}' not found in run '{run_name}'.")
        runtimes.append(run.summary.get("_runtime", 0) / 60)  # seconds -> minutes
        num_clients.append(run.config.get("num_clients"))

    if not metric_vals:
        print("No valid metric values found.")
        return None
    if len(set(num_clients)) != 1:
        print("Inconsistent 'num_clients' among runs.")
        return None

    if "f1" in metric_name:
        metric_vals = [v * 100 for v in metric_vals]

    return (metric_name,
            np.mean(metric_vals), np.std(metric_vals),
            np.mean(runtimes), np.std(runtimes),
            num_clients[0])


def plot(plot_name: str, trajectories: dict, 
         x_label: str, y_label: str, 
         ylim: Optional[Tuple[int, int]] = None):
    """Plot single or mean+std trajectories."""
    fig, ax = plt.subplots(figsize=(6, 5))
    for method, (x, y) in trajectories.items():
        if isinstance(y[0], tuple):
            means = np.array([val[0] for val in y])
            stds = np.array([val[1] for val in y])
            ax.plot(x, means, marker='o', linewidth=2, label=MAP_METHOD_TO_LABEL[method])
            ax.fill_between(x, means - stds, means + stds, alpha=0.2)
        else:
            ax.plot(x, y, marker='o', label=MAP_METHOD_TO_LABEL[method])

    ax.set_xlabel(x_label, fontsize=16)
    ax.set_ylabel(y_label, fontsize=16)
    ax.tick_params(labelsize=14)
    ax.legend(loc='best', fontsize=14)
    ax.grid(True)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    if ylim:
        ax.set_ylim(ylim)

    fig.tight_layout()
    out_dir = Path("./results/plots")
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / f"{plot_name}.pdf", format="pdf", bbox_inches="tight")
    plt.close(fig)


def print_runtime_ratios(runtime_trajectories):
    """Print the ratio of powerset to laser runtime for the first 6 values."""
    K = 6
    laser_clients, laser_runtimes = runtime_trajectories["laser"]
    powerset_clients, powerset_runtimes = runtime_trajectories["powerset"]

    laser_clients, laser_runtimes = laser_clients[:K], laser_runtimes[:K]
    powerset_clients, powerset_runtimes = powerset_clients[:K], powerset_runtimes[:K]

    assert laser_clients == powerset_clients, "Mismatch in client list"
    ratios = [p[0] / l[0] for l, p in zip(laser_runtimes, powerset_runtimes)]
    for nc, ratio in zip(laser_clients, ratios):
        print(f"runtime ratio (powerset/laser) for K={nc} clients: {ratio}")


def main(project_name, experiments, metric_name, methods):
    api = wandb.Api()
    runtime_traj, metric_traj = {}, {}

    for exp in experiments:
        
        results = [get_result(project_name, run_names, metric_name, api) for run_names in exp]
        results = [r for r in results if r is not None]
        if not results:
            continue
        
        method = exp[0][0].split('_')[1]
        clients, rtimes, metrics = [], [], []
        for (_, avg_m, std_m, avg_r, std_r, nc) in results:
            clients.append(nc)
            rtimes.append((avg_r, std_r))
            metrics.append((avg_m, std_m))

        runtime_traj[method] = (clients, rtimes)
        metric_traj[method] = (clients, metrics)

    plot("Runtime scalability", runtime_traj, "Number of clients", "Runtime (min)")
    plot("Performance scalability", metric_traj, "Number of clients", "Test accuracy", (0, 100))
    if "laser" in methods and "powerset" in methods:
        print_runtime_ratios(runtime_traj)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--project_name', default='pvaldeira-team/laser-vfl')
    parser.add_argument('--methods', nargs='+', default=['laser', 'combinatorial', 'local', 'svfl', 'ensemble', 'plug'])
    args = parser.parse_args()

    task_name = "cifar10"
    p_miss_train = "p_miss_train0.1"
    metric = "final_test_acc_0.1"
    num_seeds = 5
    max_num_clients = 8
    
    exps = [
        [[f"{task_name}_{m}_K{k}_{p_miss_train}_s{i}" for i in range(num_seeds)]
            for k in (range(2, max_num_clients) if m == "combinatorial" else range(2, max_num_clients + 1))]
        for m in args.methods
    ]

    main(args.project_name, exps, metric, args.methods)
