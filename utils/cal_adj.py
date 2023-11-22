def cal_adj(data, parameter):
    """Calculate the adjacency matrix"""
    dist = cosine_dis(data, data)
    graph = create_graph(dist, parameter, self_dist=True)
    adj = (1 - dist) * graph
    adj += adj.T - adj.multiply(adj.T)
    adj = F.normalize(adj + torch.eye(adj.shape[0]), p=1)
    return to_sparse(adj)


def cal_te_adj(data_all, data_train_val, data_test, train_val_indices, test_indices, threshold):
    num_train_val = len(train_val_indices)
    num_test = len(test_indices)

    # Initialize adjacency matrix
    adjacency = torch.zeros((data_all.shape[0], data_all.shape[0]), device=device)

    # Compute distances and create graph for training/validation to test
    dist_train_val_to_test = cosine_dis(data_train_val, data_test)
    graph_train_val_to_test = create_graph(dist_train_val_to_test, threshold, self_dist=False)
    adjacency[:num_train_val, :num_test] = (1 - dist_train_val_to_test) * graph_train_val_to_test

    # Compute distances and create graph for test to training/validation
    dist_test_to_train_val = cosine_dis(data_test, data_train_val)
    graph_test_to_train_val = create_graph(dist_test_to_train_val, threshold, self_dist=False)
    adjacency[num_train_val:, num_test:] = (1 - dist_test_to_train_val) * graph_test_to_train_val

    # Symmetrize and normalize the adjacency matrix
    adjacency = sym_norm(adjacency)

    # Convert to sparse format
    return to_sparse(adjacency)

def sym_norm(adjacency):
    """
    Symmetrize and normalize the adjacency matrix.
    This function uses a globally defined device variable.
    """
    adj_T = adjacency.transpose(0, 1)
    identity = torch.eye(adjacency.shape[0], device=device)
    adjacency = adjacency + adj_T * (adj_T > adjacency).float() - adjacency * (adj_T > adjacency).float()
    return F.normalize(adjacency + identity, p=1)

def to_sparse(x):
    # Check if the tensor is already sparse
    if x.is_sparse:
        return x
    # Find non-zero indices
    indices = torch.nonzero(x, as_tuple=False)
    # Handle the case where all elements are zeros
    if indices.numel() == 0:
        return torch.sparse_coo_tensor([], [], x.size()).to(x.dtype)

    # Transpose the indices to match sparse tensor format
    indices = indices.t()
    # Extract values at the non-zero indices
    values = x[tuple(indices[i] for i in range(indices.size(0)))]
    # Create a sparse tensor with the same type as the input tensor
    return torch.sparse_coo_tensor(indices, values, x.size()).to(x.dtype)

