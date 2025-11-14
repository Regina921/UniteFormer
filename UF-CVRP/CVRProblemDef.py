
import torch
import numpy as np
 
def get_random_problems(batch_size, problem_size):
    depot_xy = torch.rand(size=(batch_size, 1, 2))  
    # shape: (batch, 1, 2)
    node_xy = torch.rand(size=(batch_size, problem_size, 2))
    # shape: (batch, problem, 2)
    if problem_size == 20:
        demand_scaler = 30  # capacity
    elif problem_size == 50:
        demand_scaler = 40  # capacity
    elif problem_size == 100:
        demand_scaler = 50  # capacity
    else:
        raise NotImplementedError
    node_demand = torch.randint(1, 10, size=(batch_size, problem_size)) / float(demand_scaler)   
    # shape: (batch, problem)
    return depot_xy, node_xy, node_demand

 
def get_edge_node_problems(depot_node_xy, num_neighbors):
    batch_size = depot_node_xy.shape[0]
    problem_size = depot_node_xy.shape[1]
    # -------------------------

    W_val_batch = (depot_node_xy[:, :, None, :] - depot_node_xy[:, None, :, :]).norm(p=2, dim=-1)  # distance_matrixs

    # instances_temp = depot_node_xy.unsqueeze(1)
    # distance_matrixs_temp = instances_temp - instances_temp.transpose(1, 2)
    # distance_matrixs = torch.norm(distance_matrixs_temp, dim=-1)
    # W_val_batch = distance_matrixs   
 
    W_val_batch_clone = W_val_batch.clone().detach()
    if num_neighbors == -1:     
        W_batch = torch.ones((batch_size, problem_size, problem_size))  # Graph is fully connected 
    else:   
        # Initialize the adjacency matrix with zeros
        W_batch = torch.zeros((batch_size, problem_size, problem_size))   
        knn_edge, knn_indices = torch.topk(W_val_batch_clone, k=num_neighbors+1, dim=-1, largest=False)  
        knn_edge_indices = knn_indices[:, :, 1:num_neighbors+1]   

        # Set connections for k-nearest neighbors
        batch_idx = torch.arange(batch_size)[:, None, None]
        problem_idx = torch.arange(problem_size)[None, :, None]
        W_batch[batch_idx, problem_idx, knn_edge_indices] = 1

    # Set the diagonal elements to 2 for self-connections
    torch.diagonal(W_batch, dim1=-2, dim2=-1).fill_(2)

    batch_edges = torch.tensor(W_batch, dtype=torch.int)   
    batch_edges_values = W_val_batch        

    return batch_edges, batch_edges_values


def augment_xy_data_by_8_fold(xy_data):
    # xy_data.shape: (batch, N, 2)
    x = xy_data[:, :, [0]]
    y = xy_data[:, :, [1]]
    # x,y shape: (batch, N, 1)

    dat1 = torch.cat((x, y), dim=2)
    dat2 = torch.cat((1 - x, y), dim=2)
    dat3 = torch.cat((x, 1 - y), dim=2)
    dat4 = torch.cat((1 - x, 1 - y), dim=2)
    dat5 = torch.cat((y, x), dim=2)
    dat6 = torch.cat((1 - y, x), dim=2)
    dat7 = torch.cat((y, 1 - x), dim=2)
    dat8 = torch.cat((1 - y, 1 - x), dim=2)

    aug_xy_data = torch.cat((dat1, dat2, dat3, dat4, dat5, dat6, dat7, dat8), dim=0)
    # shape: (8*batch, N, 2)

    return aug_xy_data