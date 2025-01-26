import random

import torch

from utils import time_decorator
from methods.method_utils import get_f1
from methods.decoupled import train_decoupled as train_ensemble

@time_decorator
def test_ensemble(dataloader, models, criterion, args, is_final=False, compute_f1=False, is_train_data=False):
    
    for model in models:
        model.eval()
    num_samples = len(dataloader.dataset)
    num_batches = len(dataloader)
    loss_l = [0.0 for _ in range(len(models))]
    correct_l = [0.0 for _ in range(len(models))]
    if compute_f1:
        true_positive_l = [0.0 for _ in range(len(models))]
        false_positive_l = [0.0 for _ in range(len(models))]
        false_negative_l = [0.0 for _ in range(len(models))]
    
    final_correct = 0.0
    if compute_f1 and is_final:
        final_true_positive = 0.0
        final_false_positive = 0.0
        final_false_negative = 0.0
    
    with torch.no_grad():
        for batch in dataloader:
            *inputs, targets, mask = batch
            inputs, targets = [tensor.to(args.device) for tensor in inputs], targets.to(args.device)

            if torch.sum(mask).item() == 0:
                num_batches -= 1 # in practice, this batch is not used
                num_samples -= len(inputs[0]) # in practice, these samples are not used
                continue
            
            cur_batch_size = inputs[0].size(0)
            ensemble_votes = torch.zeros(cur_batch_size, len(dataloader.dataset.classes)).to(args.device)
            for i, model in enumerate(models):
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                loss_l[i] += loss.item()
                predicted = outputs.argmax(1)
                correct_l[i] += (predicted == targets).float().sum().item()
                if compute_f1:
                    true_positive_l[i] += ((predicted == 1) & (targets == 1)).sum().item()
                    false_positive_l[i] += ((predicted == 1) & (targets == 0)).sum().item()
                    false_negative_l[i] += ((predicted == 0) & (targets == 1)).sum().item()
                
                if is_final:
                    if len(model.clients_in_model) == 1: # local model
                        client_in_model = next(iter(model.clients_in_model))
                        observed_blocks_l = torch.nonzero(mask).view(-1).tolist()
                        if client_in_model in observed_blocks_l:
                            argmax_indices = outputs.argmax(1)
                            one_hot = torch.zeros_like(outputs).scatter_(1, argmax_indices.unsqueeze(1), 1)
                            ensemble_votes += one_hot
                    else:
                        raise NotImplementedError("The final accuracy with p_miss@test>=0 is only implemented for local models and standard VFL models.")
            
            if is_final:
                predicted = random_argmax_along_axis(ensemble_votes).to(args.device)
                final_correct += (predicted == targets).float().sum().item()
                # assuming that when training local model we train one per each client
                if compute_f1:
                    final_true_positive += ((predicted == 1) & (targets == 1)).sum().item()
                    final_false_positive += ((predicted == 1) & (targets == 0)).sum().item()
                    final_false_negative += ((predicted == 0) & (targets == 1)).sum().item()

    data_split_type = "train" if is_train_data else "test"
    metrics = {}
    if is_final:
        metrics[f"final_{data_split_type}_acc"] = 100 * final_correct / num_samples
        if compute_f1:
            metrics[f"final_{data_split_type}_f1"] = get_f1(final_true_positive, final_false_positive, final_false_negative)
    else:
        avg_loss = [loss / num_batches for loss in loss_l]
        accuracy = [100 * correct / num_samples for correct in correct_l]
        metrics[f"{data_split_type}_loss"] = avg_loss
        metrics[f"{data_split_type}_acc"] = accuracy
        if compute_f1:
            metrics[f"{data_split_type}_f1"] = [get_f1(tp, fp, fn) for tp, fp, fn in zip(true_positive_l, false_positive_l, false_negative_l)]

    return metrics

def random_argmax_along_axis(tensor, axis=1):
    
    if axis != 1:
        raise NotImplementedError("This function only supports axis=1")

    random_argmax_indices = []
    for row in tensor:
        max_value = torch.max(row)
        max_indices = torch.nonzero(row == max_value, as_tuple=False).squeeze()
        if len(max_indices.shape) > 0 and max_indices.shape[0] > 1:
            random_index = max_indices[random.choice(range(max_indices.shape[0]))]
        else:
            random_index = max_indices.item()
        random_argmax_indices.append(random_index)

    return torch.tensor(random_argmax_indices)
