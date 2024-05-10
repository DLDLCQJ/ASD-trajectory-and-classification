import numpy as np

def create_graph(dist, parameter, self_dist=True):
    if self_dist:
        assert dist.shape[0] == dist.shape[1], "Input must be a square matrix for self distances."
    graph = (dist <= parameter).float()
    if self_dist:
        # Avoid self loops in the graph by setting diagonal elements to 0
        np.fill_diagonal(graph.numpy(), 0)
    return graph
