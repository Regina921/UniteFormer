
from dataclasses import dataclass
import torch
from CVRProblemDef import get_random_problems, augment_xy_data_by_8_fold, get_edge_node_problems
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import os


@dataclass
class Reset_State:
    depot_xy: torch.Tensor = None  
    # shape: (batch, 1, 2)
    node_xy: torch.Tensor = None   
    # shape: (batch, problem, 2)
    node_demand: torch.Tensor = None   
    # shape: (batch, problem)
    depot_node_demand: torch.Tensor = None   
    # shape: (batch, problem+1)


@dataclass
class Step_State:
    BATCH_IDX: torch.Tensor = None
    POMO_IDX: torch.Tensor = None
    # shape: (batch, pomo)
    selected_count: int = None
    load: torch.Tensor = None     
    # shape: (batch, pomo)
    current_node: torch.Tensor = None
    # shape: (batch, pomo)
    ninf_mask: torch.Tensor = None
    # shape: (batch, pomo, problem+1)
    finished: torch.Tensor = None  
    # shape: (batch, pomo)


class CVRPEnv:
    def __init__(self, **env_params):
        # Const @INIT
        ####################################
        self.env_params = env_params
        self.problem_size = env_params['problem_size']
        self.pomo_size = env_params['pomo_size']
   
        self.num_neighbors = env_params['num_neighbors']   
        self.data_path = env_params['data_path']          
        self.mode = env_params['mode']
        self.optimal_label = env_params['optimal_label']
        self.raw_pkl_data = None

        self.FLAG__use_saved_problems = False
        self.saved_depot_xy = None
        self.saved_node_xy = None
        self.saved_node_demand = None
        self.saved_index = None

        # Const @Load_Problem
        ####################################
        self.batch_size = None
        self.BATCH_IDX = None
        self.POMO_IDX = None
        # IDX.shape: (batch, pomo)
        self.depot_node_xy = None
        # shape: (batch, problem+1, 2)
        self.depot_node_demand = None
        # shape: (batch, problem+1)

        # Dynamic-1
        ####################################
        self.selected_count = None
        self.current_node = None
        # shape: (batch, pomo)
        self.selected_node_list = None
        # shape: (batch, pomo, 0~)

        # Dynamic-2
        ####################################
        self.at_the_depot = None
        # shape: (batch, pomo)
        self.load = None
        # shape: (batch, pomo)
        self.visited_ninf_flag = None
        # shape: (batch, pomo, problem+1)
        self.ninf_mask = None
        # shape: (batch, pomo, problem+1)
        self.finished = None
        # shape: (batch, pomo)
        # states to return
        ####################################
        self.reset_state = Reset_State()
        self.step_state = Step_State()
 
        self.x_edges = None          
        self.x_edges_values = None    
        self.noaug_problems = None
        self.dist = None        
 
    def load_problems(self, episode, batch_size, aug_factor=1):
        self.batch_size = batch_size

        if self.mode == 'train':
            depot_xy, node_xy, node_demand = get_random_problems(batch_size, self.problem_size)

        else:
            if os.path.splitext(self.data_path)[1] == '.txt':  
                self.problems_nodes = self.raw_data_nodes[episode: episode + batch_size]
                # shape (B, V+1, 2) 
                self.Batch_demand = self.raw_data_demand[episode: episode + batch_size]
                # shape (B, V+1)  
                self.Batch_capacity = self.raw_data_capacity[episode: episode + batch_size]  # [B,1] 
                self.solution = self.raw_data_node_flag[episode: episode + batch_size]
                # shape (B,V,2)
                self.Batch_capacity = self.Batch_capacity[:, None].repeat(1, self.solution.shape[1] + 1)  # [B,N+1]
                # shape (B,V+1)
                self.problems = torch.cat((self.problems_nodes, self.Batch_demand[:, :, None], self.Batch_capacity[:, :, None]), dim=2)

                depot_xy = self.problems_nodes[:, 0, :]
                depot_xy = depot_xy[:, None, :]
                node_xy = self.problems_nodes[:, 1:, :]
                node_demand = self.Batch_demand[:, 1:]

            # test: cluster/expansion/grid/implosion/mixed/uniform
            if os.path.splitext(self.data_path)[1] == '.pkl':
                if episode is not None:
                    data = self.raw_pkl_data[episode: episode + batch_size]
                for i in range(len(data)):
                    depot_xy = torch.FloatTensor(data[i][0]).unsqueeze(0) if i == 0 else torch.cat( (depot_xy, torch.FloatTensor(data[i][0]).unsqueeze(0)), dim=0)
                    node_xy = torch.FloatTensor(data[i][1]).unsqueeze(0).cuda() if i == 0 else torch.cat( (node_xy, torch.FloatTensor(data[i][1]).unsqueeze(0).cuda()), dim=0)
                    node_demand = torch.FloatTensor(data[i][2]).unsqueeze(0) if i == 0 else torch.cat( (node_demand, torch.FloatTensor(data[i][2]).unsqueeze(0)), dim=0)
                depot_xy = depot_xy.unsqueeze(1).cuda()
                node_demand = node_demand.cuda() / float(data[0][3])
                self.problems = None
                self.solution = None

        self.depot_node_xy = torch.cat((depot_xy, node_xy), dim=1)
        self.x_edges, self.x_edges_values = get_edge_node_problems(self.depot_node_xy, self.num_neighbors)
        # [B, N+1, N+1]; [B, N+1, K]

        if aug_factor > 1:
            self.batch_size = self.batch_size * aug_factor
            if aug_factor == 8:  
                depot_xy = augment_xy_data_by_8_fold(depot_xy)
                node_xy = augment_xy_data_by_8_fold(node_xy)
                node_demand = node_demand.repeat(aug_factor, 1)
                self.x_edges = self.x_edges.repeat(aug_factor, 1, 1)
                self.x_edges_values = self.x_edges_values.repeat(aug_factor, 1, 1)

            else:   
                depot_xy = depot_xy.repeat(aug_factor, 1, 1)
                node_xy = node_xy.repeat(aug_factor, 1, 1)
                node_demand = node_demand.repeat(aug_factor, 1)

                self.x_edges = self.x_edges.repeat(aug_factor, 1, 1)
                self.x_edges_values = self.x_edges_values.repeat(aug_factor, 1, 1)

        # ===========================================================
        self.depot_node_xy = torch.cat((depot_xy, node_xy), dim=1)   # shape: (batch, problem+1, 2)
        depot_demand = torch.zeros(size=(self.batch_size, 1))          # shape: (batch, 1)
        self.depot_node_demand = torch.cat((depot_demand, node_demand), dim=1)   # shape: (batch, problem+1)
        # ==============================================================
        self.dist = (self.depot_node_xy[:, :, None, :] - self.depot_node_xy[:, None, :, :]).norm(p=2, dim=-1)
        # shape: (batch, problem+1, problem+1)

        self.BATCH_IDX = torch.arange(self.batch_size)[:, None].expand(self.batch_size, self.pomo_size)
        self.POMO_IDX = torch.arange(self.pomo_size)[None, :].expand(self.batch_size, self.pomo_size)

        self.reset_state.depot_xy = depot_xy     
        self.reset_state.node_xy = node_xy         
        self.reset_state.node_demand = node_demand  
        self.reset_state.depot_node_demand = self.depot_node_demand  
        self.step_state.BATCH_IDX = self.BATCH_IDX
        self.step_state.POMO_IDX = self.POMO_IDX

    def reset(self):
        self.selected_count = 0
        self.current_node = None
        # shape: (batch, pomo)
        self.selected_node_list = torch.zeros((self.batch_size, self.pomo_size, 0), dtype=torch.long)
        # shape: (batch, pomo, 0~)
        self.at_the_depot = torch.ones(size=(self.batch_size, self.pomo_size), dtype=torch.bool)
        # shape: (batch, pomo)
        self.load = torch.ones(size=(self.batch_size, self.pomo_size))  
        # shape: (batch, pomo)
        self.visited_ninf_flag = torch.zeros(size=(self.batch_size, self.pomo_size, self.problem_size+1))
        # shape: (batch, pomo, problem+1)
        self.ninf_mask = torch.zeros(size=(self.batch_size, self.pomo_size, self.problem_size+1))
        # shape: (batch, pomo, problem+1)
        self.finished = torch.zeros(size=(self.batch_size, self.pomo_size), dtype=torch.bool)
        # shape: (batch, pomo)
        reward = None
        done = False
        return self.reset_state, reward, done

    def pre_step(self):
        self.step_state.selected_count = self.selected_count
        self.step_state.load = self.load
        self.step_state.current_node = self.current_node
        self.step_state.ninf_mask = self.ninf_mask
        self.step_state.finished = self.finished
        reward = None
        done = False
        return self.step_state, reward, done

    def step(self, selected):
        # selected.shape: (batch, pomo)
        # Dynamic-1
        ####################################
        self.selected_count += 1
        self.current_node = selected
        # shape: (batch, pomo)
        self.selected_node_list = torch.cat((self.selected_node_list, self.current_node[:, :, None]), dim=2)
        # shape: (batch, pomo, 0~)

        # Dynamic-2
        ####################################
        self.at_the_depot = (selected == 0)
        demand_list = self.depot_node_demand[:, None, :].expand(self.batch_size, self.pomo_size, -1)
        # shape: (batch, pomo, problem+1)
        gathering_index = selected[:, :, None]
        # shape: (batch, pomo, 1)
        selected_demand = demand_list.gather(dim=2, index=gathering_index).squeeze(dim=2)
        # shape: (batch, pomo)
        self.load -= selected_demand

        self.load[self.at_the_depot] = 1    # refill loaded at the depot

        self.visited_ninf_flag[self.BATCH_IDX, self.POMO_IDX, selected] = float('-inf')
        # shape: (batch, pomo, problem+1)
        self.visited_ninf_flag[:, :, 0][~self.at_the_depot] = 0  # depot is considered unvisited, unless you are AT the depot

        self.ninf_mask = self.visited_ninf_flag.clone()
        round_error_epsilon = 0.00001
        demand_too_large = self.load[:, :, None] + round_error_epsilon < demand_list
        # shape: (batch, pomo, problem+1)
        self.ninf_mask[demand_too_large] = float('-inf')
        # shape: (batch, pomo, problem+1)
        newly_finished = (self.visited_ninf_flag == float('-inf')).all(dim=2)
        # shape: (batch, pomo)
        self.finished = self.finished + newly_finished
        # shape: (batch, pomo)
        # do not mask depot for finished episode.
        self.ninf_mask[:, :, 0][self.finished] = 0

        self.step_state.selected_count = self.selected_count
        self.step_state.load = self.load
        self.step_state.current_node = self.current_node
        self.step_state.ninf_mask = self.ninf_mask
        self.step_state.finished = self.finished
        # returning values
        done = self.finished.all()
        if done:
            reward = -self._get_travel_distance()  # note the minus sign!
        else:
            reward = None
        return self.step_state, reward, done


    def _get_travel_distance(self):
        gathering_index = self.selected_node_list[:, :, :, None].expand(-1, -1, -1, 2)
        # shape: (batch, pomo, selected_list_length, 2)
        all_xy = self.depot_node_xy[:, None, :, :].expand(-1, self.pomo_size, -1, -1)
        # shape: (batch, pomo, problem+1, 2)
        ordered_seq = all_xy.gather(dim=2, index=gathering_index)
        # shape: (batch, pomo, selected_list_length, 2)
        rolled_seq = ordered_seq.roll(dims=2, shifts=-1)
        segment_lengths = ((ordered_seq-rolled_seq)**2).sum(3).sqrt()
        # shape: (batch, pomo, selected_list_length)
        travel_distances = segment_lengths.sum(2)
        # shape: (batch, pomo)
        return travel_distances


    def get_cur_feature(self):
        if self.current_node is None:
            return None
        current_node = self.current_node[:, :, None, None].expand(self.batch_size, self.pomo_size, 1, self.problem_size + 1)  # [B,M,1,N+1]
        cur_dist = torch.take_along_dim(self.dist[:, None, :, :].expand(self.batch_size, self.pomo_size, self.problem_size + 1, self.problem_size + 1), current_node, dim=2).squeeze(2)
        # shape: (batch, multi, problem)  
        return cur_dist


    def _get_best_distance(self, problems_, solution_,):  
        if self.solution != None:  
            problems = problems_[:, :, [0, 1]].clone()
            order_node = solution_[:, :, 0].clone()
            order_flag = solution_[:, :, 1].clone()
            travel_distances = self.cal_length(problems, order_node, order_flag).mean()
        else: 
            travel_distances = self.optimal_label
        return travel_distances

    def cal_length(self, problems, order_node, order_flag):
        # problems:   [B,V+1,2]
        # order_node: [B,V]
        # order_flag: [B,V]
        order_node_ = order_node.clone()
        order_flag_ = order_flag.clone()
        index_small = torch.le(order_flag_, 0.5)
        index_bigger = torch.gt(order_flag_, 0.5)
        order_flag_[index_small] = order_node_[index_small]
        order_flag_[index_bigger] = 0
        roll_node = order_node_.roll(dims=1, shifts=1)
        problem_size = problems.shape[1] - 1
        order_gathering_index = order_node_.unsqueeze(2).expand(-1, problem_size, 2)
        order_loc = problems.gather(dim=1, index=order_gathering_index)
        roll_gathering_index = roll_node.unsqueeze(2).expand(-1, problem_size, 2)
        roll_loc = problems.gather(dim=1, index=roll_gathering_index)
        flag_gathering_index = order_flag_.unsqueeze(2).expand(-1, problem_size, 2)
        flag_loc = problems.gather(dim=1, index=flag_gathering_index)
        order_lengths = ((order_loc - flag_loc) ** 2)
        order_flag_[:,0]=0
        flag_gathering_index = order_flag_.unsqueeze(2).expand(-1, problem_size, 2)
        flag_loc = problems.gather(dim=1, index=flag_gathering_index)
        roll_lengths = ((roll_loc - flag_loc) ** 2)
        length = (order_lengths.sum(2).sqrt() + roll_lengths.sum(2).sqrt()).sum(1)
        return length

# ==================================================================
    def load_raw_data(self, episode=100000):
        def tow_col_nodeflag(node_flag):
            tow_col_node_flag = []
            V = int(len(node_flag) / 2)
            for i in range(V):
                tow_col_node_flag.append([node_flag[i], node_flag[V + i]])
            return tow_col_node_flag

        # Because the dataset is too large, I split it into two reads
        if self.env_params['mode'] == 'test':
            self.raw_data_nodes = []  
            self.raw_data_capacity = []
            self.raw_data_demand = []
            self.raw_data_cost = []
            self.raw_data_node_flag = []
            for line in tqdm(open(self.data_path, "r").readlines()[0:episode], ascii=True):
                line = line.split(",")
                depot_index = int(line.index('depot'))
                customer_index = int(line.index('customer'))
                capacity_index = int(line.index('capacity'))
                demand_index = int(line.index('demand'))
                cost_index = int(line.index('cost'))
                node_flag_index = int(line.index('node_flag'))

                depot = [[float(line[depot_index + 1]), float(line[depot_index + 2])]]
                customer = [[float(line[idx]), float(line[idx + 1])] for idx in range(customer_index + 1, capacity_index, 2)]

                loc = depot + customer
                capacity = int(float(line[capacity_index + 1]))
                if int(line[demand_index + 1]) ==0:
                    demand = [int(line[idx]) for idx in range(demand_index + 1, cost_index)]
                else:
                    demand = [0] + [int(line[idx]) for idx in range(demand_index + 1, cost_index)]

                cost = float(line[cost_index + 1])
                node_flag = [int(line[idx]) for idx in range(node_flag_index + 1, len(line))]

                node_flag = tow_col_nodeflag(node_flag)

                self.raw_data_nodes.append(loc)   
                self.raw_data_capacity.append(capacity/capacity) 
                self.raw_data_demand.append(demand) 
                self.raw_data_cost.append(cost)  
                self.raw_data_node_flag.append(node_flag)  

            self.raw_data_nodes = torch.tensor(self.raw_data_nodes, requires_grad=False)
            # shape (B,V+1,2)  customer num + depot
            self.raw_data_capacity = torch.tensor(self.raw_data_capacity, requires_grad=False)  # 
            # shape (B )
            self.raw_data_demand = torch.tensor(self.raw_data_demand, requires_grad=False)/float(capacity)
            # shape (B,V+1) customer num + depot
            self.raw_data_cost = torch.tensor(self.raw_data_cost, requires_grad=False)
            # shape (B )
            self.raw_data_node_flag = torch.tensor(self.raw_data_node_flag, requires_grad=False)
            # shape (B,V,2)
        print(f'load raw dataset done!', )


    def load_pkl_distribution_problems(self, load_path=None, episode=None):
        if self.data_path is not None:  # test
            import os
            import pickle
            filename = self.data_path
            assert os.path.splitext(filename)[1] == '.pkl'
            with open(filename, 'rb') as f:
                data = pickle.load(f)
            self.raw_pkl_data = data

