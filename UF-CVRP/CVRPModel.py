import torch
import torch.nn as nn
import torch.nn.functional as F
from gcn_layers import ResidualGatedGCNLayer, MLP


class CVRPModel(nn.Module):
    def __init__(self, **model_params):
        super().__init__()
        self.model_params = model_params
        self.node_encoder = CVRP_NodeEncoder(**model_params)
        self.graph_encoder = CVRP_GraphEncoder(**model_params)

        self.selfattention = SelfAttention(**model_params)

        self.decoder = CVRP_Decoder(**model_params)
        self.encoded_nodes = None
        self.encoded_graph = None
        # shape: (batch, problem+1, EMBEDDING_DIM)

    def pre_forward(self, reset_state, x_edges, x_edges_values, xe_choice):
        depot_xy = reset_state.depot_xy   
        # shape: (batch, 1, 2)
        node_xy = reset_state.node_xy
        # shape: (batch, problem, 2)
        node_demand = reset_state.node_demand
        # shape: (batch, problem)
        node_xy_demand = torch.cat((node_xy, node_demand[:, :, None]), dim=2)  
        # shape: (batch, problem, 3)
        depot_node_demand = reset_state.depot_node_demand   

 
        self.encoded_graph = self.graph_encoder(depot_xy, node_xy_demand, depot_node_demand[:, :, None], x_edges, x_edges_values, xe_choice)  # GCN
        self.encoded_nodes = self.node_encoder(depot_xy, node_xy_demand, xe_choice)
        # shape: (batch, problem+1, EMBEDDING_DIM) 

        self.encoded_graph = self.selfattention(self.encoded_graph)   

        self.decoder.set_kv(self.encoded_graph, self.encoded_nodes)  # (batch, problem+1, embedding)

    def forward(self, state):
        batch_size = state.BATCH_IDX.size(0)
        pomo_size = state.BATCH_IDX.size(1)
        if state.selected_count == 0:  # First Move, depot
            selected = torch.zeros(size=(batch_size, pomo_size), dtype=torch.long) 
            prob = torch.ones(size=(batch_size, pomo_size))
 
            # # Use Averaged encoded nodes for decoder input_1  
            encoded_nodes_mean = self.encoded_nodes.mean(dim=1, keepdim=True)  # shape: (batch, 1, embedding)
            encoded_graph_mean = self.encoded_graph.mean(dim=1, keepdim=True)  # shape: (batch, 1, embedding)
            self.decoder.set_q1(encoded_nodes_mean, encoded_graph_mean)

            # # Use encoded_depot for decoder input_2   
            encoded_first_node = self.encoded_nodes[:, [0], :]  # shape: (batch, 1, embedding)
            encoded_first_graph = self.encoded_graph[:, [0], :]  # shape: (batch, 1, embedding)
            self.decoder.set_q2(encoded_first_node, encoded_first_graph)

        elif state.selected_count == 1:  # Second Move, POMO   
            selected = torch.arange(start=1, end=pomo_size + 1)[None, :].expand(batch_size, pomo_size)
            prob = torch.ones(size=(batch_size, pomo_size))

        else:
            encoded_last_node = _get_encoding(self.encoded_nodes, state.current_node)
            encoded_last_graph = _get_encoding(self.encoded_graph, state.current_node)
            # shape: (batch, pomo, embedding)
            probs = self.decoder(encoded_last_node, encoded_last_graph, state.load, ninf_mask=state.ninf_mask)
            # shape: (batch, pomo, problem+1)

            if self.training or self.model_params['eval_type'] == 'softmax':
                while True:  # to fix pytorch.multinomial bug on selecting 0 probability elements
                    with torch.no_grad():
                        selected = probs.reshape(batch_size * pomo_size, -1).multinomial(1) \
                            .squeeze(dim=1).reshape(batch_size, pomo_size)
                    # shape: (batch, pomo)
                    prob = probs[state.BATCH_IDX, state.POMO_IDX, selected].reshape(batch_size, pomo_size)
                    # shape: (batch, pomo)
                    if (prob != 0).all():
                        break
            else:
                selected = probs.argmax(dim=2)
                # shape: (batch, pomo)
                prob = None  # value not needed. Can be anything.
        return selected, prob


def _get_encoding(encoded_nodes, node_index_to_pick):
    # encoded_nodes.shape: (batch, problem, embedding)
    # node_index_to_pick.shape: (batch, pomo)
    batch_size = node_index_to_pick.size(0)
    pomo_size = node_index_to_pick.size(1)
    embedding_dim = encoded_nodes.size(2)
    gathering_index = node_index_to_pick[:, :, None].expand(batch_size, pomo_size, embedding_dim)
    # shape: (batch, pomo, embedding)
    picked_nodes = encoded_nodes.gather(dim=1, index=gathering_index)
    # shape: (batch, pomo, embedding)
    return picked_nodes


########################################
# DOUBLE ENCODER
########################################
 
class SelfAttention(nn.Module):
    def __init__(self, **model_params):
        super().__init__()
        self.model_params = model_params
        self.embedding_dim = self.model_params['embedding_dim']
 
        self.Wq = nn.Linear(self.embedding_dim, self.embedding_dim, bias=False)
        self.Wk = nn.Linear(self.embedding_dim, self.embedding_dim, bias=False)
        self.Wv = nn.Linear(self.embedding_dim, self.embedding_dim, bias=False)
        self.add_n_normalization_1 = AddAndBatchNormalization(**model_params)
        self.add_n_normalization_2 = AddAndBatchNormalization(**model_params)
        self.feed_forward = FeedForward(**model_params)

    def forward(self, input1): 
 
        Q = self.Wq(input1)  # (batch_size, seq_len, embedding_dim)
        K = self.Wk(input1)  # (batch_size, seq_len, embedding_dim)
        V = self.Wv(input1)  # (batch_size, seq_len, embedding_dim)
    
        energy = torch.bmm(Q, K.transpose(1, 2))  # (batch_size, seq_len, seq_len)
        attention_scores = F.softmax(energy / (self.embedding_dim ** 0.5), dim=-1)   
   
        attention_output = torch.bmm(attention_scores, V)  # (batch_size, seq_len, embedding_dim)
        out1 = self.add_n_normalization_1(input1, attention_output)
        out2 = self.feed_forward(out1)
        out3 = self.add_n_normalization_2(out1, out2)
        return out3

 
class Node_EncodingBlock(nn.Module):
    def __init__(self, **model_params):
        super().__init__()
        self.model_params = model_params
        self.embedding_dim = self.model_params['embedding_dim']  
        self.row_encoding_block = EncodingBlock(**model_params)   
        self.embedding_demand = nn.Linear(1, self.embedding_dim)   
        self.embedding_node = nn.Linear(self.embedding_dim * 2, self.embedding_dim)  

    def forward(self, x_edges_values, depot_node_demand):  
 
        batch_size = x_edges_values.size(0)
        problem_size = x_edges_values.size(1)
        # [A]row_emb=0
        row_emb = torch.zeros(size=(batch_size, problem_size, self.embedding_dim))   
        # emb.shape: (batch, node+1, embedding)

        # [B]col_emb=one-hot
        col_emb = torch.zeros(size=(batch_size, problem_size, self.embedding_dim))   
        # shape: (batch, node+1, embedding)
 
        if problem_size <= self.embedding_dim:   
            seed_cnt = problem_size  
            rand = torch.rand(batch_size, seed_cnt)
            batch_rand_perm = rand.argsort(dim=1)   
        else:  
            seed_cnt = self.embedding_dim
            rand = torch.rand(batch_size, seed_cnt)
            batch_rand_perm = rand.argsort(dim=1)
            if self.embedding_dim <= problem_size < self.embedding_dim * 2:   
                batch_rand_perm = torch.cat((batch_rand_perm, batch_rand_perm), dim=-1)
            elif self.embedding_dim * 2 <= problem_size < self.embedding_dim * 3:   
                batch_rand_perm = torch.cat((batch_rand_perm, batch_rand_perm, batch_rand_perm), dim=-1)
            elif self.embedding_dim * 3 <= problem_size < self.embedding_dim * 4:   
                batch_rand_perm = torch.cat((batch_rand_perm, batch_rand_perm, batch_rand_perm, batch_rand_perm),
                                            dim=-1)
            else:
                raise NotImplementedError

        rand_idx = batch_rand_perm[:, :problem_size]
        b_idx = torch.arange(batch_size)[:, None].expand(batch_size, problem_size)
        n_idx = torch.arange(problem_size)[None, :].expand(batch_size, problem_size)
        col_emb[b_idx, n_idx, rand_idx] = 1
 
        row_emb_out = self.row_encoding_block(row_emb, col_emb, x_edges_values)   
  
        demand_feature = self.embedding_demand(depot_node_demand)   
        node_feature_demand = torch.cat((row_emb_out, demand_feature), dim=2) 
        node_feature = self.embedding_node(node_feature_demand)   

        return node_feature  # (batch, node+1, embedding)

 
class EncodingBlock(nn.Module):
    def __init__(self, **model_params):
        super().__init__()
        self.model_params = model_params
        embedding_dim = self.model_params['embedding_dim']
        head_num = self.model_params['head_num']
        qkv_dim = self.model_params['qkv_dim']
        self.Wq = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.Wk = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.Wv = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.mixed_score_MHA = MixedScore_MultiHeadAttention(**model_params)   
        self.multi_head_combine = nn.Linear(head_num * qkv_dim, embedding_dim)
        self.feed_forward = FeedForward(**model_params)
 
        self.add_n_normalization_1 = AddAndBatchNormalization(**model_params)
        self.add_n_normalization_2 = AddAndBatchNormalization(**model_params)

    def forward(self, row_emb, col_emb, cost_mat):  
        # [NOTE]: row and col can be exchanged, if cost_mat.transpose(1,2) is used
        # input1.shape: (batch, row_cnt, embedding)
        # input2.shape: (batch, col_cnt, embedding)
        # cost_mat.shape: (batch, row_cnt, col_cnt)
        head_num = self.model_params['head_num']
        q = reshape_by_heads(self.Wq(row_emb), head_num=head_num)
        # q shape: (batch, head_num, row_cnt, qkv_dim)
        k = reshape_by_heads(self.Wk(col_emb), head_num=head_num)
        v = reshape_by_heads(self.Wv(col_emb), head_num=head_num)
        # kv shape: (batch, head_num, col_cnt, qkv_dim)
        out_concat = self.mixed_score_MHA(q, k, v, cost_mat)
        # shape: (batch, row_cnt, head_num*qkv_dim)
        multi_head_out = self.multi_head_combine(out_concat)
        # shape: (batch, row_cnt, embedding)
        out1 = self.add_n_normalization_1(row_emb, multi_head_out)
        out2 = self.feed_forward(out1)
        out3 = self.add_n_normalization_2(out1, out2)
        return out3
        # shape: (batch, row_cnt, embedding)


########################################
#  Graph ENCODER
########################################
 
class CVRP_GraphEncoder(nn.Module):
    def __init__(self, **model_params):
        super().__init__()
        self.model_params = model_params
 
        self.embedding_dim = self.model_params['embedding_dim'] 
        self.num_layers = self.model_params['GCN_dim']  
        self.mlp_layers = self.model_params['mlp_layers']   
        self.aggregation = self.model_params['aggregation']  
        self.embedding_depot = nn.Linear(2, self.embedding_dim)
        self.embedding_node = nn.Linear(3, self.embedding_dim)  

        # Node and edge embedding layers/lookups
        self.edges_values_embedding = nn.Linear(1, self.embedding_dim // 2)   
        self.edges_embedding = nn.Embedding(3, self.embedding_dim // 2)  
        self.node_encodingblock = Node_EncodingBlock(**model_params)   

        # GCN Layers
        gcn_layers = []
        for layer in range(self.num_layers):  # 3
            gcn_layers.append(ResidualGatedGCNLayer(self.embedding_dim, self.aggregation))
        self.gcn_layers = nn.ModuleList(gcn_layers)
        # MLP classifiers
        self.mlp_nodes = MLP(self.embedding_dim, self.embedding_dim, self.mlp_layers) 

    def forward(self, depot_xy, node_xy_demand, depot_node_demand, x_edges, x_edges_values, xe_choice):
        # depot_node_demand: [B,N+1,1]
        batch_size = node_xy_demand.shape[0]   
        problem_size = node_xy_demand.shape[1]   

        if xe_choice == 0:  # 0 edge: X=precoder
            x = self.node_encodingblock(x_edges_values, depot_node_demand)   

            e_vals = self.edges_values_embedding(x_edges_values.unsqueeze(3))  # B x V x V x H
            e_tags = self.edges_embedding(x_edges)  # B x V x V x H
            e = torch.cat((e_vals, e_tags), dim=3)  # [B N+1 N+1 H]

        elif xe_choice == 1:  # 1node: e=0
            embedded_depot = self.embedding_depot(depot_xy)  # shape: (batch, 1, embedding)
            embedded_node = self.embedding_node(node_xy_demand)   # shape: (batch, problem, embedding)
            x = torch.cat((embedded_depot, embedded_node), dim=1)  # [B,N+1,emb]
            e = torch.zeros(batch_size, problem_size + 1, problem_size + 1, self.embedding_dim)  # e=0

        elif xe_choice == 2:  # edge+node
            embedded_depot = self.embedding_depot(depot_xy)    # shape: (batch, 1, embedding)
            embedded_node = self.embedding_node(node_xy_demand)    # shape: (batch, problem, embedding)
            x = torch.cat((embedded_depot, embedded_node), dim=1)  # [B,N+1,emb]
            # [2]edge-embedding
            e_vals = self.edges_values_embedding(x_edges_values.unsqueeze(3))  # B x V x V x H
            e_tags = self.edges_embedding(x_edges)  # B x V x V x H
            e = torch.cat((e_vals, e_tags), dim=3)  # [B N N H]
        else:
            raise NotImplementedError("Unknown search method")
        # GCN layers
        for layer in range(self.num_layers):
            x, e = self.gcn_layers[layer](x, e)  # B x V x H, B x V x V x H

        # MLP classifier
        graph_encoded = self.mlp_nodes(x)  # B x V x voc_nodes_out  [B,N,256]
 
        return graph_encoded


#################################################
# Node ENCODER
########################################
class CVRP_NodeEncoder(nn.Module):
    def __init__(self, **model_params):
        super().__init__()
        self.model_params = model_params
        self.embedding_dim = self.model_params['embedding_dim']
        encoder_layer_num = self.model_params['encoder_layer_num']
        self.embedding_depot = nn.Linear(2, self.embedding_dim)
        self.embedding_node = nn.Linear(3, self.embedding_dim)
        self.layers = nn.ModuleList([EncoderLayer(**model_params) for _ in range(encoder_layer_num)])

    def forward(self, depot_xy, node_xy_demand, xe_choice):  # x_nodes_false, node_feature
        # depot_xy.shape: (batch, 1, 2)
        # node_xy_demand.shape: (batch, problem, 3)
        # data.shape: (batch, problem, 2)
        batch_size = depot_xy.shape[0]
        problem_size = node_xy_demand.shape[1]
        if xe_choice == 0:  # edge
            out = torch.zeros(size=(batch_size, problem_size + 1, self.embedding_dim))   

        elif xe_choice == 1 or xe_choice == 2:  # node / edge+node, e=0
            embedded_depot = self.embedding_depot(depot_xy)  # shape: (batch, 1, embedding)
            embedded_node = self.embedding_node(node_xy_demand)  # shape: (batch, problem, embedding)
            embedded_input = torch.cat((embedded_depot, embedded_node), dim=1)  
            # shape: (batch, problem+1, embedding)
            out = embedded_input
            for layer in self.layers:
                out = layer(out)
        else:
            raise NotImplementedError("Unknown search method")
        return out  # shape: (batch, problem+1, embedding)


class EncoderLayer(nn.Module):
    def __init__(self, **model_params):
        super().__init__()
        self.model_params = model_params
        embedding_dim = self.model_params['embedding_dim']
        head_num = self.model_params['head_num']
        qkv_dim = self.model_params['qkv_dim']

        self.Wq = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.Wk = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.Wv = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.multi_head_combine = nn.Linear(head_num * qkv_dim, embedding_dim)
        self.feed_forward = FeedForward(**model_params)
 
        self.add_n_normalization_1 = AddAndBatchNormalization(**model_params)
        self.add_n_normalization_2 = AddAndBatchNormalization(**model_params)

    def forward(self, input1):
        # input1.shape: (batch, problem+1, embedding)
        head_num = self.model_params['head_num']
        q = reshape_by_heads(self.Wq(input1), head_num=head_num)
        k = reshape_by_heads(self.Wk(input1), head_num=head_num)
        v = reshape_by_heads(self.Wv(input1), head_num=head_num)
        # qkv shape: (batch, head_num, problem, qkv_dim)
        out_concat = multi_head_attention(q, k, v)
        # shape: (batch, problem, head_num*qkv_dim)
        multi_head_out = self.multi_head_combine(out_concat)
        # shape: (batch, problem, embedding)
        out1 = self.add_n_normalization_1(input1, multi_head_out)
        out2 = self.feed_forward(out1)
        out3 = self.add_n_normalization_2(out1, out2)
        return out3
        # shape: (batch, problem, embedding)


########################################
# DECODER
########################################
 
class CVRP_Decoder(nn.Module):
    def __init__(self, **model_params):
        super().__init__()
        self.model_params = model_params
        embedding_dim = self.model_params['embedding_dim']
        head_num = self.model_params['head_num']
        qkv_dim = self.model_params['qkv_dim']

        # [1] 
        self.Wq1 = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.Wq2 = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.Wq_last = nn.Linear(embedding_dim + 1, head_num * qkv_dim, bias=False)
        self.Wk = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.Wv = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.multi_head_combine = nn.Linear(head_num * qkv_dim, embedding_dim)
        self.k = None  # saved key, for multi-head attention
        self.v = None  # saved value, for multi-head_attention
        self.single_head_key = None  # saved, for single-head attention
        self.q1 = None  # saved q1, for multi-head attention
        self.q2 = None  # saved q2, for multi-head attention

        # [2] 
        self.Wq1_e = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.Wq2_e = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.Wq_last_e = nn.Linear(embedding_dim + 1, head_num * qkv_dim, bias=False)
        self.Wk_e = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.Wv_e = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.multi_head_combine2 = nn.Linear(head_num * qkv_dim, embedding_dim)
        self.k_e = None
        self.v_e = None
        self.single_head_key_e = None
        self.q1_e = None
        self.q2_e = None
 
        self.feed_forward = FeedForward(**model_params)
 
    def set_kv(self, encoded_graph, encoded_nodes):
        # encoded_nodes.shape: (batch, problem+1, embedding)
        head_num = self.model_params['head_num']
 
        self.k = reshape_by_heads(self.Wk(encoded_nodes), head_num=head_num)
        self.v = reshape_by_heads(self.Wv(encoded_nodes), head_num=head_num)
        # shape: (batch, head_num, problem+1, qkv_dim)
        self.single_head_key = encoded_nodes.transpose(1, 2)
        # shape: (batch, embedding, problem+1)
 
        self.k_e = reshape_by_heads(self.Wk_e(encoded_graph), head_num=head_num)
        self.v_e = reshape_by_heads(self.Wv_e(encoded_graph), head_num=head_num)
        # shape: (batch, head_num, pomo, qkv_dim)
        self.single_head_key_e = encoded_graph.transpose(1, 2)
   
    def set_q1(self, encoded_q1, encoded_graph_q1):  
        # encoded_q.shape: (batch, n, embedding)  # n can be 1 or pomo
        head_num = self.model_params['head_num']
        self.q1 = reshape_by_heads(self.Wq1(encoded_q1), head_num=head_num)
        self.q1_e = reshape_by_heads(self.Wq1_e(encoded_graph_q1), head_num=head_num)
        # shape: (batch, head_num, n, qkv_dim)

    def set_q2(self, encoded_q2, encoded_graph_q2):   
        # encoded_q.shape: (batch, n, embedding)  # n can be 1 or pomo
        head_num = self.model_params['head_num']
        self.q2 = reshape_by_heads(self.Wq2(encoded_q2), head_num=head_num)
        self.q2_e = reshape_by_heads(self.Wq2_e(encoded_graph_q2), head_num=head_num)
        # shape: (batch, head_num, n, qkv_dim)

    def forward(self, encoded_last_node, encoded_last_graph, load, ninf_mask):
        # encoded_last_node.shape: (batch, pomo, embedding)
        # load.shape: (batch, pomo)
        # ninf_mask.shape: (batch, pomo, problem)
        head_num = self.model_params['head_num']

        #  Multi-Head Attention
        #######################################################
        input_cat = torch.cat((encoded_last_node, load[:, :, None]), dim=2)  # shape = (batch, group, EMBEDDING_DIM+1
        q_last = reshape_by_heads(self.Wq_last(input_cat), head_num=head_num)
        # shape: (batch, head_num, pomo, qkv_dim)
        input_cat_e = torch.cat((encoded_last_graph, load[:, :, None]),
                                dim=2)  # shape = (batch, group, EMBEDDING_DIM+1)
        q_last_e = reshape_by_heads(self.Wq_last_e(input_cat_e), head_num=head_num)
        # shape: (batch, head_num, pomo, qkv_dim)
 
        q_x = self.q1 + self.q2 + q_last
        q_e = self.q1_e + self.q2_e + q_last_e
        q = q_x + q_e
        # =====================================
        out_concat1 = multi_head_attention(q, self.k, self.v, rank3_ninf_mask=ninf_mask)
        out_concat2 = multi_head_attention(q, self.k_e, self.v_e, rank3_ninf_mask=ninf_mask)
        # shape: (batch, pomo, head_num*qkv_dim)
        mh_atten_out1 = self.multi_head_combine(out_concat1)
        mh_atten_out2 = self.multi_head_combine2(out_concat2)
        # shape: (batch, pomo, embedding)

        ############ ============
        batch = q.size(0)
        head_num = q.size(1)
        n = q.size(2)
        key_dim = q.size(3)
        q0 = q.transpose(1, 2)  # shape: (batch, n, head_num, key_dim)
        q0 = q0.reshape(batch, n, head_num * key_dim)  # shape: (batch, n, head_num*key_dim)

        ########## ##########
        mh_atten_out = mh_atten_out1 + mh_atten_out2   

        mh_atten_out = mh_atten_out + q0  # +
        mh_atten_out = mh_atten_out + self.feed_forward(mh_atten_out)  
 
        single_head_key = self.single_head_key + self.single_head_key_e   
        #  Single-Head Attention, for probability calculation
        #######################################################
        score = torch.matmul(mh_atten_out, single_head_key)
        # shape: (batch, pomo, problem)
        sqrt_embedding_dim = self.model_params['sqrt_embedding_dim']
        logit_clipping = self.model_params['logit_clipping']
        score_scaled = score / sqrt_embedding_dim
        # shape: (batch, pomo, problem)
        score_clipped = logit_clipping * torch.tanh(score_scaled)
        score_masked = score_clipped + ninf_mask
        probs = F.softmax(score_masked, dim=2)
        # shape: (batch, pomo, problem)
        return probs


########################################
# NN SUB CLASS / FUNCTIONS
########################################
def reshape_by_heads(qkv, head_num):
    # q.shape: (batch, n, head_num*key_dim)   : n can be either 1 or PROBLEM_SIZE
    batch_s = qkv.size(0)
    n = qkv.size(1)
    q_reshaped = qkv.reshape(batch_s, n, head_num, -1)
    # shape: (batch, n, head_num, key_dim)
    q_transposed = q_reshaped.transpose(1, 2)
    # shape: (batch, head_num, n, key_dim)
    return q_transposed


def multi_head_attention(q, k, v, rank2_ninf_mask=None, rank3_ninf_mask=None):
    # q shape: (batch, head_num, n, key_dim)   : n can be either 1 or PROBLEM_SIZE
    # k,v shape: (batch, head_num, problem, key_dim)
    # rank2_ninf_mask.shape: (batch, problem)
    # rank3_ninf_mask.shape: (batch, group, problem)
    batch_s = q.size(0)
    head_num = q.size(1)
    n = q.size(2)
    key_dim = q.size(3)
    input_s = k.size(2)
    score = torch.matmul(q, k.transpose(2, 3))
    # shape: (batch, head_num, n, problem)

    score_scaled = score / torch.sqrt(torch.tensor(key_dim, dtype=torch.float))
    if rank2_ninf_mask is not None:
        score_scaled = score_scaled + rank2_ninf_mask[:, None, None, :].expand(batch_s, head_num, n, input_s)
    if rank3_ninf_mask is not None:
        score_scaled = score_scaled + rank3_ninf_mask[:, None, :, :].expand(batch_s, head_num, n, input_s)
    weights = nn.Softmax(dim=3)(score_scaled)
    # shape: (batch, head_num, n, problem)
    out = torch.matmul(weights, v)
    # shape: (batch, head_num, n, key_dim)
    out_transposed = out.transpose(1, 2)
    # shape: (batch, n, head_num, key_dim)
    out_concat = out_transposed.reshape(batch_s, n, head_num * key_dim)
    # shape: (batch, n, head_num*key_dim)
    return out_concat


# BN
class AddAndBatchNormalization(nn.Module):
    def __init__(self, **model_params):
        super().__init__()
        embedding_dim = model_params['embedding_dim']
        self.norm_by_EMB = nn.BatchNorm1d(embedding_dim, affine=True)
        # 'Funny' Batch_Norm, as it will normalized by EMB dim

    def forward(self, input1, input2):
        # input.shape: (batch, problem, embedding)
        batch_s = input1.size(0)
        problem_s = input1.size(1)
        embedding_dim = input1.size(2)
        added = input1 + input2
        normalized = self.norm_by_EMB(added.reshape(batch_s * problem_s, embedding_dim))
        back_trans = normalized.reshape(batch_s, problem_s, embedding_dim)

        return back_trans


class FeedForward(nn.Module):
    def __init__(self, **model_params):
        super().__init__()
        embedding_dim = model_params['embedding_dim']
        ff_hidden_dim = model_params['ff_hidden_dim']
        self.W1 = nn.Linear(embedding_dim, ff_hidden_dim)
        self.W2 = nn.Linear(ff_hidden_dim, embedding_dim)

    def forward(self, input1):
        # input.shape: (batch, problem, embedding)
        return self.W2(F.relu(self.W1(input1)))

 
class MixedScore_MultiHeadAttention(nn.Module):
    def __init__(self, **model_params):
        super().__init__()
        self.model_params = model_params
        head_num = model_params['head_num']
        ms_hidden_dim = model_params['ms_hidden_dim']   
        mix1_init = model_params['ms_layer1_init']  
        mix2_init = model_params['ms_layer2_init']   

        mix1_weight = torch.torch.distributions.Uniform(low=-mix1_init, high=mix1_init).sample(
            (head_num, 2, ms_hidden_dim))
        mix1_bias = torch.torch.distributions.Uniform(low=-mix1_init, high=mix1_init).sample((head_num, ms_hidden_dim))
        self.mix1_weight = nn.Parameter(mix1_weight)
        # shape: (head, 2, ms_hidden)
        self.mix1_bias = nn.Parameter(mix1_bias)
        # shape: (head, ms_hidden)

        mix2_weight = torch.torch.distributions.Uniform(low=-mix2_init, high=mix2_init).sample(
            (head_num, ms_hidden_dim, 1))
        mix2_bias = torch.torch.distributions.Uniform(low=-mix2_init, high=mix2_init).sample((head_num, 1))
        self.mix2_weight = nn.Parameter(mix2_weight)
        # shape: (head, ms_hidden, 1)
        self.mix2_bias = nn.Parameter(mix2_bias)
        # shape: (head, 1)

    def forward(self, q, k, v, cost_mat):  # (q, k, v, cost_mat=problem)
        # q shape: (batch, head_num, row_cnt, qkv_dim)
        # k,v shape: (batch, head_num, col_cnt, qkv_dim)
        # cost_mat.shape: (batch, row_cnt, col_cnt)
        batch_size = q.size(0)
        row_cnt = q.size(2)
        col_cnt = k.size(2)
        head_num = self.model_params['head_num']
        qkv_dim = self.model_params['qkv_dim']
        sqrt_qkv_dim = self.model_params['sqrt_qkv_dim']

        dot_product = torch.matmul(q, k.transpose(2, 3))
        # shape: (batch, head_num, row_cnt, col_cnt)
        dot_product_score = dot_product / sqrt_qkv_dim
        # shape: (batch, head_num, row_cnt, col_cnt)
        cost_mat_score = cost_mat[:, None, :, :].expand(batch_size, head_num, row_cnt, col_cnt)
        # shape: (batch, head_num, row_cnt, col_cnt)
        two_scores = torch.stack((dot_product_score, cost_mat_score), dim=4)
        # shape: (batch, head_num, row_cnt, col_cnt, 2)
        two_scores_transposed = two_scores.transpose(1, 2)
        # shape: (batch, row_cnt, head_num, col_cnt, 2)
        ms1 = torch.matmul(two_scores_transposed, self.mix1_weight)
        # shape: (batch, row_cnt, head_num, col_cnt, ms_hidden_dim)
        ms1 = ms1 + self.mix1_bias[None, None, :, None, :]
        # shape: (batch, row_cnt, head_num, col_cnt, ms_hidden_dim)
        ms1_activated = F.relu(ms1)
        ms2 = torch.matmul(ms1_activated, self.mix2_weight)
        # shape: (batch, row_cnt, head_num, col_cnt, 1)
        ms2 = ms2 + self.mix2_bias[None, None, :, None, :]
        # shape: (batch, row_cnt, head_num, col_cnt, 1)
        mixed_scores = ms2.transpose(1, 2)
        # shape: (batch, head_num, row_cnt, col_cnt, 1)
        mixed_scores = mixed_scores.squeeze(4)
        # shape: (batch, head_num, row_cnt, col_cnt)
        weights = nn.Softmax(dim=3)(mixed_scores)
        # shape: (batch, head_num, row_cnt, col_cnt)
        out = torch.matmul(weights, v)
        # shape: (batch, head_num, row_cnt, qkv_dim)
        out_transposed = out.transpose(1, 2)
        # shape: (batch, row_cnt, head_num, qkv_dim)
        out_concat = out_transposed.reshape(batch_size, row_cnt, head_num * qkv_dim)
        # shape: (batch, row_cnt, head_num*qkv_dim)

        return out_concat

