import wandb
import numpy as np
import argparse


def main(project_name, run_names, metric_name):

    api = wandb.Api()

    metric_values = []
    
    for run_name in run_names:
        # Grab the first run matching the filter, or None if none exist
        run = next(api.runs(project_name, filters={"display_name": run_name}), None)
        
        if run is None:
            print(f"No run found with name '{run_name}'")
            continue
        
        print(f"Run Name: {run.name}, Run ID: {run.id}")
        
        # Get the metric value from the summary
        metric_value = run.summary.get(metric_name)
        if metric_value is not None:
            metric_values.append(metric_value)
        else:
            print(f"Metric '{metric_name}' not found in run '{run_name}'")

    if metric_values:
        # Multiply by 100 for F1-like metrics
        if "f1" in metric_name.lower():
            metric_values = [v * 100 for v in metric_values]
        
        mean_val, std_val = np.mean(metric_values), np.std(metric_values)
        print(f"{metric_name}: {mean_val:.1f} ± {std_val:.1f}")
    else:
        print("No valid metric values found to calculate average and standard deviation.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--project_name', default='pvaldeira-team/vfl-sandbox')
    args = parser.parse_args()

    # Example usage (None means p_{miss} is sampled):
    metric_name = "final_test_acc_None"
    run_names = [f"powerset_cifar10_beta_s{i}" for i in range(5)]

    main(args.project_name, run_names, metric_name)
