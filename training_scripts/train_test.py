from models import *
from training_scripts import *


def train_test(trval_idx, tr_idx, te_idx, labels, data_list, view_list, num_class, num_layers, model_type, lr_e, lr_e_pretrain, lr_c, num_epoch, patience,checkpoints, num_epoch_pretrain, parameter, latent_dim_list, gcn_dropout, k_order, rank):
    num_view = len(view_list)
    dim_out = num_class

    # Data processing
    data_all_list, data_trval_list, data_tr_list, data_te_list, adj_te_list, adj_trval_list, adj_tr_list = process_data(trval_idx, tr_idx, te_idx, data_list, parameter)
    dim_list = [x.shape[1] for x in data_all_list]

    # Initialize model
    model_dict = init_model(num_view, num_class, dim_list, latent_dim_list, num_layers, gcn_dropout, k_order, rank, dim_out, model_type)
    for m in model_dict:
        model_dict[m].to(device)

    # Pretraining
    optim_dict = init_optim(num_view, model_dict, lr_e_pretrain, lr_c)
    pretrain(num_epoch_pretrain, data_tr_list, adj_tr_list, data_trval_list, adj_trval_list, labels, tr_idx, trval_idx, model_dict, optim_dict)

    # Training
    optim_dict = init_optim(num_view, model_dict, lr_e, lr_c)
    val_acc = train(num_epoch, data_tr_list, adj_tr_list, data_trval_list, adj_trval_list, labels, tr_idx, trval_idx, model_dict, optim_dict, patience, checkpoints)

    # Testing
    te_acc, precision, recall, fscore = test(data_all_list, adj_te_list, te_idx, labels, model_dict)

    return val_acc, te_acc, precision, recall, fscore

def pretrain(num_epoch_pretrain, data_tr_list, adj_tr_list, data_trval_list, adj_trval_list, labels, tr_idx, trval_idx, model_dict, optim_dict):
    """
    Pretraining phase.
    """
    for epoch in range(num_epoch_pretrain):
        train_epoch(data_tr_list, adj_tr_list, data_trval_list, adj_trval_list, labels, tr_idx, trval_idx, model_dict, optim_dict, LMF=False)

def train(num_epoch, data_tr_list, adj_tr_list, data_trval_list, adj_trval_list, labels, tr_idx, trval_idx, model_dict, optim_dict, patience, checkpoints):
    """
    Training phase with early stopping.
    """
    early_stopping = EarlyStopping(patience, checkpoints, verbose=True)
    for epoch in range(num_epoch + 1):
        tr_loss_epoch,  val_loss_epoch, val_acc_epoch = train_epoch(data_tr_list, adj_tr_list, data_trval_list, adj_trval_list, labels, tr_idx, trval_idx, model_dict, optim_dict)
        early_stopping(val_loss_epoch, model_dict)
        if early_stopping.early_stop:
            print("Early stopping")
            break
    return val_acc_epoch

def test(data_list, adj_list, indices, labels, model_dict):
    """
    Testing phase.
    """
    te_prob,_ = evaluate(data_list, adj_list, indices, labels, model_dict)
    te_acc = accuracy_score(labels[indices], te_prob.argmax(1))
    precision, recall, fscore, _ = precision_recall_fscore_support(labels[indices], te_prob.argmax(1), average='macro')
    print("Test set results:", "accuracy= {:.4f}".format(te_acc))
    return te_acc, precision, recall, fscore
