import wandb
import numpy as np
import argparse


def main(project_name, run_names, metric_name):
    
    api = wandb.Api()

    metric_values = []
    for run_name in run_names:

        runs = api.runs(f"{project_name}", filters={"display_name": run_name})

        if len(runs) > 0:
            run = runs[0]
            print(f"Run Name: {run.name}, Run ID: {run.id}")
        else:
            print(f"No run found with name {run_name}")

        metric_value = run.summary.get(metric_name, None)
        if metric_value is not None:
            metric_values.append(metric_value)
        else:
            print(f"Metric {metric_name} not found in run {run_name}")

    if metric_values:
        if "f1" in metric_name:
            metric_values = [metric_value * 100 for metric_value in metric_values]
        average = np.mean(metric_values)
        std_dev = np.std(metric_values)
        print(f"{metric_name}: {average:.1f} ± {std_dev:.1f}")
    else:
        print("No valid metric values found to calculate average and standard deviation.")


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--project_name', default='my-team/vfl-sandbox')
    args = parser.parse_args()

    metric_name = "final_test_acc_None" # 0.0 | 0.1 | 0.5 | None
    run_names = ["powerset_cifar10_beta_s" + str(i) for i in range(5)]

    main(args.project_name, run_names, metric_name)
