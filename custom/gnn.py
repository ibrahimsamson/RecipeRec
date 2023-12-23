from imports import *
class custom_HeteroGraphConv(nn.Module):
    def __init__(self, mods, aggregate='sum'):
        super(custom_HeteroGraphConv, self).__init__()
        self.mods = nn.ModuleDict(mods)
        # Do not break if graph has 0-in-degree nodes.
        # Because there is no general rule to add self-loop for heterograph.
        for _, v in self.mods.items():
            set_allow_zero_in_degree_fn = getattr(v, 'set_allow_zero_in_degree', None)
            if callable(set_allow_zero_in_degree_fn):
                set_allow_zero_in_degree_fn(True)
        if isinstance(aggregate, str):
            self.agg_fn = get_aggregate_fn(aggregate)
        else:
            self.agg_fn = aggregate

    def forward(self, g, inputs, mod_args=None, mod_kwargs=None):
        if mod_args is None:
            mod_args = {}
        if mod_kwargs is None:
            mod_kwargs = {}
        outputs = {nty : [] for nty in g.dsttypes}
        if isinstance(inputs, tuple) or g.is_block:
            if isinstance(inputs, tuple):
                src_inputs, dst_inputs = inputs
            else:
                src_inputs = inputs
                dst_inputs = {k: v[:g.number_of_dst_nodes(k)] for k, v in inputs.items()}

            for stype, etype, dtype in g.canonical_etypes:
                rel_graph = g[stype, etype, dtype]
                if rel_graph.number_of_edges() == 0:
                    continue
                if stype not in src_inputs or dtype not in dst_inputs:
                    continue
                dstdata = self.mods[etype](
                    rel_graph,
                    (src_inputs[stype], dst_inputs[dtype], mod_kwargs),
                    *mod_args.get(etype, ())
                    )
                outputs[dtype].append(dstdata)
        else:
            for stype, etype, dtype in g.canonical_etypes:
                rel_graph = g[stype, etype, dtype]
                if rel_graph.number_of_edges() == 0:
                    continue
                if stype not in inputs:
                    continue
                dstdata = self.mods[etype](
                    rel_graph,
                    (inputs[stype], inputs[dtype], mod_kwargs),
                    *mod_args.get(etype, ())
                    )
                outputs[dtype].append(dstdata)
        rsts = {}
        for nty, alist in outputs.items():
            if len(alist) != 0:
                rsts[nty] = self.agg_fn(alist, nty)
        return rsts

def _max_reduce_func(inputs, dim):
    return torch.max(inputs, dim=dim)[0]

def _min_reduce_func(inputs, dim):
    return torch.min(inputs, dim=dim)[0]

def _sum_reduce_func(inputs, dim):
    return torch.sum(inputs, dim=dim)

def _mean_reduce_func(inputs, dim):
    return torch.mean(inputs, dim=dim)

def _stack_agg_func(inputs, dsttype):
    if len(inputs) == 0:
        return None
    return torch.stack(inputs, dim=1)

def _agg_func(inputs, dsttype, fn):
    if len(inputs) == 0:
        return None
    stacked = torch.stack(inputs, dim=0)
    return fn(stacked, dim=0)

def get_aggregate_fn(agg):
    if agg == 'sum':
        fn = _sum_reduce_func
    elif agg == 'max':
        fn = _max_reduce_func
    elif agg == 'min':
        fn = _min_reduce_func
    elif agg == 'mean':
        fn = _mean_reduce_func
    elif agg == 'stack':
        fn = None  # will not be called
    else:
        raise DGLError('Invalid cross type aggregator. Must be one of '
                       '"sum", "max", "min", "mean" or "stack". But got "%s"' % agg)
    if agg == 'stack':
        return _stack_agg_func
    else:
        return partial(_agg_func, fn=fn)
    

class ScorePredictor(nn.Module):
    def forward(self, edge_subgraph, x):
        with edge_subgraph.local_scope():
            edge_subgraph.ndata['x'] = x
            edge_subgraph.apply_edges(
                dgl.function.u_dot_v('x', 'x', 'score'), etype='u-r')
            return edge_subgraph.edata['score'][('user', 'u-r', 'recipe')].squeeze()


class RelationAttention(nn.Module):
    def __init__(self, in_size, hidden_size=16):
        super(RelationAttention, self).__init__()

        self.project = nn.Sequential(
            nn.Linear(in_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1, bias=False)
        )

    def forward(self, z):
        w = self.project(z).mean(0)                    # (M, 1)
        beta = torch.softmax(w, dim=0)                 # (M, 1)
        beta = beta.expand((z.shape[0],) + beta.shape) # (N, M, 1)
        out = (beta * z).sum(1)                        # (N, D * K)
        return out
    
def node_drop(feats, drop_rate, training):
    n = feats.shape[0]
    drop_rates = torch.FloatTensor(np.ones(n) * drop_rate)
    
    if training:
        masks = torch.bernoulli(1. - drop_rates).unsqueeze(1)
        feats = masks.to(feats.device) * feats / (1. - drop_rate)
    else:
        feats = feats
    return feats


class custom_GATConv(nn.Module):
    def __init__(self,
                 in_feats,
                 out_feats,
                 num_heads,
                 feat_drop=0.1,
                 attn_drop=0.,
                 negative_slope=0.2,
                 edge_drop=0.1,
                 residual=False,
                 activation=None,
                 allow_zero_in_degree=False,
                 bias=True):
        super(custom_GATConv, self).__init__()
        self._num_heads = num_heads
        self._in_src_feats, self._in_dst_feats = expand_as_pair(in_feats)
        self._out_feats = out_feats
        self._allow_zero_in_degree = allow_zero_in_degree
        if isinstance(in_feats, tuple):
            self.fc_src = nn.Linear(
                self._in_src_feats, out_feats * num_heads, bias=False)
            self.fc_dst = nn.Linear(
                self._in_dst_feats, out_feats * num_heads, bias=False)
            self.fc_src2 = nn.Linear(
                self._in_src_feats, out_feats * num_heads, bias=False)
            self.fc_dst2 = nn.Linear(
                self._in_dst_feats, out_feats * num_heads, bias=False)
        else:
            self.fc = nn.Linear(
                self._in_src_feats, out_feats * num_heads, bias=False)
            self.fc2 = nn.Linear(
                self._in_src_feats, out_feats * num_heads, bias=False)
        self.attn_l = nn.Parameter(torch.FloatTensor(size=(1, num_heads, out_feats)))
        self.attn_r = nn.Parameter(torch.FloatTensor(size=(1, num_heads, out_feats)))
        self.attn_l2 = nn.Parameter(torch.FloatTensor(size=(1, num_heads, out_feats)))
        self.attn_r2 = nn.Parameter(torch.FloatTensor(size=(1, num_heads, out_feats)))
        self.feat_drop = nn.Dropout(feat_drop)
        self.attn_drop = nn.Dropout(attn_drop)
        self.leaky_relu = nn.LeakyReLU(negative_slope)
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(size=(num_heads * out_feats,)))
        else:
            self.register_buffer('bias', None)
        if residual:
            if self._in_dst_feats != out_feats:
                self.res_fc = nn.Linear(
                    self._in_dst_feats, num_heads * out_feats, bias=False)
            else:
                self.res_fc = Identity()
        else:
            self.register_buffer('res_fc', None)
        self.reset_parameters()
        self.activation = activation
        
        self.edge_drop = edge_drop

    def reset_parameters(self):
        gain = nn.init.calculate_gain('relu')
        if hasattr(self, 'fc'):
            nn.init.xavier_normal_(self.fc.weight, gain=gain)
        else:
            nn.init.xavier_normal_(self.fc_src.weight, gain=gain)
            nn.init.xavier_normal_(self.fc_dst.weight, gain=gain)
        nn.init.xavier_normal_(self.attn_l, gain=gain)
        nn.init.xavier_normal_(self.attn_r, gain=gain)
        nn.init.xavier_normal_(self.attn_l2, gain=gain)
        nn.init.xavier_normal_(self.attn_r2, gain=gain)
        nn.init.constant_(self.bias, 0)
        if isinstance(self.res_fc, nn.Linear):
            nn.init.xavier_normal_(self.res_fc.weight, gain=gain)

    def set_allow_zero_in_degree(self, set_value):
        self._allow_zero_in_degree = set_value

    def forward(self, graph, feat, get_attention=False):
        with graph.local_scope():
            if not self._allow_zero_in_degree:
                if (graph.in_degrees() == 0).any():
                    raise DGLError('There are 0-in-degree nodes in the graph, '
                                   'output for those nodes will be invalid. '
                                   'This is harmful for some applications, '
                                   'causing silent performance regression. '
                                   'Adding self-loop on the input graph by '
                                   'calling `g = dgl.add_self_loop(g)` will resolve '
                                   'the issue. Setting ``allow_zero_in_degree`` '
                                   'to be `True` when constructing this module will '
                                   'suppress the check and let the code run.')
            
            if isinstance(feat, tuple):
                do_edge_drop = feat[2]
                # print('do_edge_drop: ', do_edge_drop)
                h_src = self.feat_drop(feat[0])
                h_dst = self.feat_drop(feat[1])
                h_src2 = h_src.clone()
                h_dst2 = h_dst.clone()
                if not hasattr(self, 'fc_src'):
                    feat_src = self.fc(h_src).view(
                        -1, self._num_heads, self._out_feats)
                    feat_dst = self.fc(h_dst).view(
                        -1, self._num_heads, self._out_feats)
                    feat_src2 = self.fc2(h_src2).view(
                        -1, self._num_heads, self._out_feats)
                    feat_dst2 = self.fc2(h_dst2).view(
                        -1, self._num_heads, self._out_feats)
                else:
                    feat_src = self.fc_src(h_src).view(
                        -1, self._num_heads, self._out_feats)
                    feat_dst = self.fc_dst(h_dst).view(
                        -1, self._num_heads, self._out_feats)
                    feat_src2 = self.fc_src2(h_src2).view(
                        -1, self._num_heads, self._out_feats)
                    feat_dst2 = self.fc_dst2(h_dst2).view(
                        -1, self._num_heads, self._out_feats)
            else:
                h_src = h_dst = self.feat_drop(feat)
                h_src2 = h_dst2 = h_src.clone() # self.feat_drop(feat)
                feat_src = feat_dst = self.fc(h_src).view(
                    -1, self._num_heads, self._out_feats)
                feat_src2 = feat_dst2 = self.fc(h_src).view(
                    -1, self._num_heads, self._out_feats)
                if graph.is_block:
                    feat_dst = feat_src[:graph.number_of_dst_nodes()]
                    feat_dst2 = feat_src2[:graph.number_of_dst_nodes()]

            el = (feat_src * self.attn_l).sum(dim=-1).unsqueeze(-1)
            er = (feat_dst * self.attn_r).sum(dim=-1).unsqueeze(-1)
            
            graph.srcdata.update({'ft': feat_src, 'el': el, 'feat_src2': feat_src2})
            graph.dstdata.update({'er': er, 'feat_dst2': feat_dst2})
            # compute edge attention, el and er are a_l Wh_i and a_r Wh_j respectively.
            graph.apply_edges(fn.u_add_v('el', 'er', 'e'))
            e = self.leaky_relu(graph.edata.pop('e'))
            
            # compute softmax, edge dropout
            if self.training and do_edge_drop and self.edge_drop > 0:
                perm = torch.randperm(graph.number_of_edges(), device=e.device)
                bound = int(graph.number_of_edges() * self.edge_drop)
                eids = perm[bound:]
                graph.edata["a"] = torch.zeros_like(e)
                graph.edata["a"][eids] = self.attn_drop(
                    edge_softmax(graph, e[eids], eids=eids))
            else:
                graph.edata['a'] = self.attn_drop(edge_softmax(graph, e))

            # message passing
            graph.update_all(fn.u_mul_e('ft', 'a', 'm'),
                             fn.sum('m', 'initial_ft'))
            graph.update_all(fn.u_mul_v('feat_src2', 'feat_dst2', 'm2'),
                             fn.sum('m2', 'add_ft'))
            rst = graph.dstdata['initial_ft'] + graph.dstdata['add_ft']
            
            # residual
            if self.res_fc is not None:
                resval = self.res_fc(h_dst).view(
                    h_dst.shape[0], self._num_heads, self._out_feats)
                rst = rst + resval
            # bias
            if self.bias is not None:
                rst = rst + self.bias.view(1, self._num_heads, self._out_feats)
            # activation
            if self.activation:
                rst = self.activation(rst)

            if get_attention:
                return rst, graph.edata['a']
            else:
                return rst

    
class GNN(nn.Module):
    def __init__(self, in_feats, hid_feats, out_feats, rel_names):
        super().__init__()
        
        self.num_heads = 4 # 8
        self.hid_feats = int(hid_feats/self.num_heads)
        self.out_feats = int(out_feats/self.num_heads)
        self.relation_attention = RelationAttention(hid_feats)
        
        self.gatconv1 = custom_HeteroGraphConv({ # dglnn.HeteroGraphConv
            'i-r': custom_GATConv(in_feats, self.hid_feats, num_heads=self.num_heads),
            'r-i': custom_GATConv(in_feats, self.hid_feats, num_heads=self.num_heads),
            'r-r': custom_GATConv(in_feats, self.hid_feats, num_heads=self.num_heads),
            'i-i': custom_GATConv(in_feats, self.hid_feats, num_heads=self.num_heads),
            'u-r': custom_GATConv(in_feats, self.hid_feats, num_heads=self.num_heads),
            'r-u': custom_GATConv(in_feats, self.hid_feats, num_heads=self.num_heads),
            }, aggregate='stack')
        
        self.gatconv2 = custom_HeteroGraphConv({ # dglnn.HeteroGraphConv
            'i-r': custom_GATConv(self.hid_feats*self.num_heads, 
                                  self.out_feats, num_heads=self.num_heads),
            'r-i': custom_GATConv(self.hid_feats*self.num_heads, 
                                  self.out_feats, num_heads=self.num_heads),
            'r-r': custom_GATConv(self.hid_feats*self.num_heads, 
                                  self.out_feats, num_heads=self.num_heads),
            'i-i': custom_GATConv(self.hid_feats*self.num_heads, 
                                  self.out_feats, num_heads=self.num_heads),
            'u-r': custom_GATConv(self.hid_feats*self.num_heads, 
                                  self.out_feats, num_heads=self.num_heads),
            'r-u': custom_GATConv(self.hid_feats*self.num_heads, 
                                  self.out_feats, num_heads=self.num_heads),
            }, aggregate='stack')

        self.dropout = nn.Dropout(0.1)
        
    
    def forward(self, blocks, inputs, do_edge_drop):
        edge_weight_0 = blocks[0].edata['weight']
        edge_weight_1 = blocks[1].edata['weight']
        
        num_users = blocks[-1].dstdata[dgl.NID]['user'].shape[0]
        num_recipes = blocks[-1].dstdata[dgl.NID]['recipe'].shape[0]
    
        h = self.gatconv1(blocks[0], inputs, edge_weight_0, do_edge_drop)
        h = {k: F.relu(v).flatten(2) for k, v in h.items()}
        h = {k: self.relation_attention(v) for k, v in h.items()} 

        first_layer_output = {}
        first_layer_output['user'] = h['user'][:num_users]
        first_layer_output['recipe'] = h['recipe'][:num_recipes]
        
        h = {key: self.dropout(value) for key, value in h.items()}
        h = self.gatconv2(blocks[-1], h, edge_weight_1, do_edge_drop)
        last_ingre_and_instr = h['recipe'].flatten(2)
        h = {k: self.relation_attention(v.flatten(2)) for k, v in h.items()}

        return h
    
#         # combine several layer embs as the final emb
#         combined_output = {}
#         combined_output['user'] = torch.cat([h['user'], first_layer_output['user']], dim=1)
#         combined_output['recipe'] = torch.cat([h['recipe'], first_layer_output['recipe']], dim=1)
#         combined_output['user'] = torch.add(h['user'], first_layer_output['user'])
#         combined_output['recipe'] = torch.add(h['recipe'], first_layer_output['recipe'])
#         return combined_output
