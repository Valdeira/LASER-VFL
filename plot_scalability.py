import wandb
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from typing import Optional, Tuple
import argparse


MAP_METHOD_TO_LABEL = {
    "rvfl": "LASER-VFL",
    "powerset": "Combinatorial",
    "local": "Local",
    "svfl": "Standard VFL",
    "ensemble": "Ensemble",
    "plug": "PlugVFL",
    }


def get_result(project_name, run_names, metric_name, api):
    metric_values, runtimes, num_clients_l = [], [], []
    for run_name in run_names:

        runs = api.runs(f"{project_name}", filters={"display_name": run_name})

        if len(runs) > 0:
            run = runs[0]
            # print(f"Run Name: {run.name}, Run ID: {run.id}")
        else:
            print(f"No run found with name {run_name}")

        metric_value = run.summary.get(metric_name, None)
        if metric_value is not None:
            metric_values.append(metric_value)
        else:
            print(f"Metric {metric_name} not found in run {run_name}")

        runtimes.append(run.summary['_runtime'] / 60) # from s to min
        
        num_clients_l.append(run.config.get("num_clients", None))

    assert all(x == num_clients_l[0] for x in num_clients_l), "Not all elements in num_clients_l are equal!"
    num_clients = num_clients_l[0]

    if metric_values:
        if "f1" in metric_name:
            metric_values = [metric_value * 100 for metric_value in metric_values]
        metric_average, metric_std_dev = np.mean(metric_values), np.std(metric_values)
        runtime_average, runtime_std_dev = np.mean(runtimes), np.std(runtimes)
        return metric_name, metric_average, metric_std_dev, runtime_average, runtime_std_dev, num_clients
    else:
        print("No valid metric values found to calculate average and standard deviation.")


def plot(plot_name: str, trajectories: dict, x_label: str, y_label: str, ylim: Optional[Tuple[int, int]] = None):
    
    plt.figure(figsize=(6, 5))

    for method, (x, y) in trajectories.items():
        
        if isinstance(y[0], tuple): # (mean, std_dev)
            means, std_devs = np.array([_y[0] for _y in y]), np.array([_y[1] for _y in y])
            plt.plot(x, means, marker='o', label=f"{MAP_METHOD_TO_LABEL[method]}", linewidth=2)
            plt.fill_between(x, means - std_devs, means + std_devs, alpha=0.2)
            if ylim is not None:
                plt.ylim(ylim)
        else: # single value
            plt.plot(x, y, marker='o')

    # plt.title(plot_name, fontsize=16)
    plt.xlabel(x_label, fontsize=16)
    plt.ylabel(y_label, fontsize=16)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)

    plt.legend(loc='best', fontsize=14)

    plt.grid(True)
    plt.tight_layout()
    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.savefig(f"./{plot_name}.pdf", format='pdf', bbox_inches='tight')
    plt.close()


def print_runtime_ratios(runtime_trajectories):
    
    K = 6 # only including the first 6 in runtime ratio print (powerset only runs for 2,...,7)
    
    rvfl_num_clients_l, rvfl_runtimes = runtime_trajectories['rvfl']
    powerset_num_clients_l, powerset_runtimes = runtime_trajectories['powerset']

    rvfl_num_clients_l, rvfl_runtimes = rvfl_num_clients_l[:K], rvfl_runtimes[:K]
    powerset_num_clients_l, powerset_num_clients_l = powerset_num_clients_l[:K], powerset_num_clients_l[:K]

    assert rvfl_num_clients_l == powerset_num_clients_l
    num_clients_l = rvfl_num_clients_l

    rvfl_runtime_means = np.array([x[0] for x in rvfl_runtimes])
    powerset_runtime_means = np.array([x[0] for x in powerset_runtimes])

    ratios = [b / a for a, b in zip(rvfl_runtime_means, powerset_runtime_means)]
    for num_clients, ratio in zip(num_clients_l, ratios):
        print(f"runtime ratio (powerset/rvfl) for K={num_clients} clients: {ratio}")


def main(project_name, experiments, metric_name):
    
    api = wandb.Api()

    runtime_trajectories, metric_trajectories = {}, {}
    for experiment in experiments:
        runtime_l, num_clients_l, metric_l, metric_name_l = [], [], [], []
        for run_names in experiment:
            metric_name, average, std_dev, runtime_average, runtime_std_dev, num_clients = get_result(project_name, run_names, metric_name, api)
            # print(f"{metric_name}: {average:.1f} ± {std_dev:.1f}")
            metric_name_l.append(metric_name)
            runtime_l.append((runtime_average, runtime_std_dev))
            num_clients_l.append(num_clients)
            metric_l.append((average, std_dev))

        assert all(x == metric_name_l[0] for x in metric_name_l), "Not all elements in metric_name_l are equal!"
        metric_name = metric_name_l[0]
        
        method = run_names[0].split('_')[0]
        runtime_trajectories[method] = (num_clients_l, runtime_l)
        metric_trajectories[method] = (num_clients_l, metric_l)
        # print(f"{method}, runtime_l: {[_r[0] for _r in runtime_l]}")

    plot("Runtime scalability", runtime_trajectories, x_label="Number of clients", y_label="Runtime (min)")
    plot("Performance scalability", metric_trajectories, x_label="Number of clients", y_label="Test accuracy", ylim=(0, 100))
    print_runtime_ratios(runtime_trajectories)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--project_name', default='my-team/vfl-sandbox')
    args = parser.parse_args()

    metric_name = "final_test_acc_0.1"
    experiments = [
        [["rvfl_cifar10_" + str(k) + "K_s" + str(i) for i in range(5)] for k in range(2, 9)],
        [["powerset_cifar10_" + str(k) + "K_s" + str(i) for i in range(5)] for k in range(2, 8)],
        [["local_cifar10_" + str(k) + "K_s" + str(i) for i in range(5)] for k in range(2, 9)],
        [["svfl_cifar10_" + str(k) + "K_s" + str(i) for i in range(5)] for k in range(2, 9)],
        [["ensemble_cifar10_" + str(k) + "K_s" + str(i) for i in range(5)] for k in range(2, 9)],
        [["plug_cifar10_" + str(k) + "K_s" + str(i) for i in range(5)] for k in range(2, 9)],
    ]
    main(args.project_name, experiments, metric_name)
