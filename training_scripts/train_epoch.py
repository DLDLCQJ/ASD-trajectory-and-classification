import time
import torch
import torch.nn as nn
import torch.nn.functional as F

from training_scripts.evaluate import *

def train_epoch(data_tr_list, adj_tr_list, data_trval_list, adj_trval_list, labels,
                tr_idx, trval_idx, model_dict, optim_dict, LMF=True):
    start_time = time.time()
    criterion = nn.CrossEntropyLoss(reduction='none')
    tr_loss_dict = {}
    num_views = len(data_tr_list)

    # Pretraining phase
    for i in range(num_views):
        model_key = f"C{i+1}"
        gcn_key = f"G{i+1}"

        optim_dict[model_key].zero_grad()
        output = model_dict[model_key](model_dict[gcn_key](data_tr_list[i], adj_tr_list[i]))
        pre_loss = torch.mean(criterion(output, torch.LongTensor(labels[tr_idx])))
        pre_loss.backward()
        optim_dict[model_key].step()
        tr_loss_dict[model_key] = pre_loss.item()
    
    # Training phase
    if LMF and num_views >= 2:
        optim_dict["F"].zero_grad()
        outputs = [model_dict[f"G{i+1}"](data_tr_list[i], adj_tr_list[i]) for i in range(num_views)]
        combined_output = model_dict["F"](outputs)
        tr_loss = torch.mean(criterion(combined_output, torch.LongTensor(labels[tr_idx])))
        tr_loss.backward()
        optim_dict["F"].step()
        tr_loss_dict["F"] = tr_loss.item()

    # Validating phase
    val_loss, val_acc = evaluate(data_trval_list, adj_trval_list, trval_idx, labels, model_dict, label=True)

    print(f"Training epoch completed in {time.time() - start_time:.2f} seconds")
    return tr_loss_dict, val_loss, val_acc


