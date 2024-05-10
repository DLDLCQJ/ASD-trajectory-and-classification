from utils import *

def process_data(trval_idx, tr_idx, te_idx, data_list, parameter):
    # Convert and split data into different sets
    data_all_list = [torch.FloatTensor(data).to(device) for data in data_list]
    data_tr_list = [data[tr_idx] for data in data_all_list]
    data_trval_list = [data[trval_idx] for data in data_all_list]
    data_te_list = [data[te_idx] for data in data_all_list]

    # Prepare adjacency matrices
    adj_tr_list, adj_trval_list, adj_te_list = [], [], []
    for i in range(len(data_all_list)):
        adj_parameter_tr = cal_adj_parameter(parameter, data_tr_list[i])
        adj_tr_list.append(cal_adj(data_tr_list[i], adj_parameter_tr))
        adj_trval_list.append(cal_adj(data_trval_list[i], adj_parameter_tr))
        adj_te_list.append(cal_te_adj(data_all_list[i], data_trval_list[i], data_te_list[i], trval_idx, te_idx, adj_parameter_tr))
    return data_all_list, data_trval_list, data_tr_list, data_te_list, adj_te_list, adj_trval_list, adj_tr_list
