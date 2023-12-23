from build_graph import *
from attention import *


def get_recipe2ingreNeighbor_dict():
    max_length = 33
    out = {}
    neighbor_list = []
    ingre_length_list = []
    total_length_index_list = []
    total_ingre_neighbor_list = []
    total_length_index = 0
    total_length_index_list.append(total_length_index)
    for recipeNodeID in tqdm(range(graph.number_of_nodes('recipe'))):
        _, succs = graph.out_edges(recipeNodeID, etype='r-i')
        succs_list = list(set(succs.tolist()))
        total_ingre_neighbor_list.extend(succs_list)
        cur_length = len(succs_list)
        ingre_length_list.append(cur_length)
        
        total_length_index += cur_length
        total_length_index_list.append(total_length_index)
        while len(succs_list) < max_length:
            succs_list.append(77733)
        neighbor_list.append(succs_list)

    ingre_neighbor_tensor = torch.tensor(neighbor_list)
    ingre_length_tensor = torch.tensor(ingre_length_list)
    total_ingre_neighbor_tensor = torch.tensor(total_ingre_neighbor_list)
    return ingre_neighbor_tensor, ingre_length_tensor, \
        total_length_index_list, total_ingre_neighbor_tensor

ingre_neighbor_tensor, ingre_length_tensor, total_length_index_list, \
    total_ingre_neighbor_tensor = get_recipe2ingreNeighbor_dict()
print('ingre_neighbor_tensor: ', ingre_neighbor_tensor.shape)
print('ingre_length_tensor: ', ingre_length_tensor.shape)
print('total_length_index_list: ', len(total_length_index_list))
print('total_ingre_neighbor_tensor: ', total_ingre_neighbor_tensor.shape)

def find(tensor, values):
    return torch.nonzero(tensor[..., None] == values)

# example of find()
# a = torch.tensor([0, 10, 20, 30])
# b = torch.tensor([[ 0, 30, 20,  10, 77733],[ 0, 30, 20,  10, 77733]])
# find(b, a)[:, 2]

def get_ingredient_neighbors_all_embeddings(blocks, output_nodes, secondToLast_ingre):
    ingreNodeIDs = blocks[1].srcdata['_ID']['ingredient']
    recipeNodeIDs = output_nodes
    batch_ingre_neighbors = ingre_neighbor_tensor[recipeNodeIDs].to(device)
    batch_ingre_length = ingre_length_tensor[recipeNodeIDs]
    valid_batch_ingre_neighbors = find(batch_ingre_neighbors, ingreNodeIDs)[:, 2]
    
    # based on valid_batch_ingre_neighbors each row index
    _, valid_batch_ingre_length = torch.unique(find(
        batch_ingre_neighbors, ingreNodeIDs)[:, 0], return_counts=True)
    batch_sum_ingre_length = np.cumsum(valid_batch_ingre_length.cpu())
    
    total_ingre_emb = None
    for i in range(len(recipeNodeIDs)):
        if i == 0:
            recipeNode_ingres = valid_batch_ingre_neighbors[0:batch_sum_ingre_length[i]]
            a = secondToLast_ingre[recipeNode_ingres]
        else:
            recipeNode_ingres = valid_batch_ingre_neighbors[
                batch_sum_ingre_length[i-1]:batch_sum_ingre_length[i]]
            a = secondToLast_ingre[recipeNode_ingres]
    
        # all ingre instead of average
        a_rows = a.shape[0]
        a_columns = a.shape[1]
        max_rows = 5
        if a_rows < max_rows:
            a = torch.cat([a, torch.zeros(max_rows-a_rows, a_columns).cuda()])
        else:
            a = a[:max_rows, :]
        
        if total_ingre_emb == None:
            total_ingre_emb = a.unsqueeze(0)
        else:
            total_ingre_emb = torch.cat([total_ingre_emb,a.unsqueeze(0)], dim = 0)
            if torch.isnan(total_ingre_emb).any():
                print('Error!')

    return total_ingre_emb

# Set transformer for ingredient representation
class SetTransformer(nn.Module):
    def __init__(self):
        """
        Arguments:
            in_dimension: an integer.  # 2
            out_dimension: an integer. # 5 * K
        """
        super(SetTransformer, self).__init__()
        in_dimension = 46 # 300
        out_dimension = 128 # 600

        d = in_dimension
        m = 46  # number of inducing points
        h = 2  # 4 # number of heads
        k = 4  # number of seed vectors

        self.encoder = nn.Sequential(
            InducedSetAttentionBlock(d, m, h, RFF(d), RFF(d)),
            InducedSetAttentionBlock(d, m, h, RFF(d), RFF(d))
        )
        self.decoder = nn.Sequential(
            PoolingMultiheadAttention(d, k, h, RFF(d)),
            SetAttentionBlock(d, h, RFF(d))
        )

        self.decoder_2 = nn.Sequential(
            PoolingMultiheadAttention(d, k, h, RFF(d))
        )
        self.decoder_3 = nn.Sequential(
            SetAttentionBlock(d, h, RFF(d))
        )

        self.predictor = nn.Sequential(
            nn.Linear(k * d, out_dimension),
            nn.ReLU()
        )
        
        self.dropout = nn.Dropout(0.1)
        

    def forward(self, x):
        """
        Arguments:
            x: a float tensor with shape [batch, n, in_dimension].
        Returns:
            a float tensor with shape [batch, out_dimension].
        """
        x = self.encoder(x) # x = self.encoder(cut_x) # shape [batch, batch_max_len, d]
        x = self.dropout(x)
        x = self.decoder(x)  # shape [batch, k, d]

        b, k, d = x.shape
        x = x.view(b, k * d)

        y = self.predictor(x)
        return y