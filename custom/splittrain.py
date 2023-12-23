from .build_graph import *


all_src_dst_weight, train_src_dst_weight, val_src_dst_weight,\
      test_src_dst_weight = torch.load(
          dataset_folder+'/all_train_val_test_edge_u_rate_r_src_and_dst_and_weight.pt')
all_src, all_dst, all_weight = all_src_dst_weight
train_src, train_dst, train_weight = train_src_dst_weight
val_src, val_dst, val_weight = val_src_dst_weight
test_src, test_dst, test_weight = test_src_dst_weight

train_eids = graph.edge_ids(train_src, train_dst, etype='u-r')
val_eids = graph.edge_ids(val_src, val_dst, etype='u-r')
test_eids = graph.edge_ids(test_src, test_dst, etype='u-r')
val_eids_r2u = graph.edge_ids(val_dst, val_src, etype='r-u')
test_eids_r2u = graph.edge_ids(test_dst, test_src, etype='r-u')
print('length of all_src: ', len(all_src))
print('length of train_eids: ', len(train_eids))
print('length of val_eids: ', len(val_eids))
print('length of test_eids: ', len(test_eids))

# get train_graph and val_graph
train_graph = graph.clone()
train_graph.remove_edges(torch.cat([val_eids, test_eids]), etype='u-r')
train_graph.remove_edges(torch.cat([val_eids_r2u, test_eids_r2u]), etype='r-u')
print('training graph: ')
print(train_graph)
print()

val_graph = graph.clone()
val_graph.remove_edges(test_eids, etype='u-r')
val_graph.remove_edges(test_eids, etype='r-u')
print('val graph: ')
print(val_graph)


# edge dataloaders
sampler = dgl.dataloading.MultiLayerNeighborSampler([20, 20])
neg_sampler = dgl.dataloading.negative_sampler.Uniform(5)

class test_NegativeSampler(object):
    def __init__(self, g, k):
        # get the negatives
        self.user2negs_100_dict = {}
        filename = dataset_folder+'/test_negatives_100.txt'
        with open(filename, "r") as f:
            lines = f.readlines()
            for line in tqdm(lines):
                if line == None or line == "":
                    continue
                line = line[:-1] # remove \n
                user = int(line.split('\t')[0].split(',')[0][1:])
                negs = [int(neg) for neg in line.split('\t')[1:]]
                self.user2negs_100_dict[user] = negs
                
        self.k = k

    def __call__(self, g, eids_dict):
        result_dict = {}
        for etype, eids in eids_dict.items():
            src, _ = g.find_edges(eids, etype=etype)
            dst = []
            for each_src in src:
                dst.extend(self.user2negs_100_dict[int(each_src)][:self.k])
            dst = torch.tensor(dst)
            src = src.repeat_interleave(self.k)
            result_dict[etype] = (src, dst)
        return result_dict
    
test_neg_sampler = test_NegativeSampler(graph, n_test_negs)
test_train_neg_sampler = test_NegativeSampler(graph, n_test_negs)
    
train_collator = dgl.dataloading.EdgeCollator(
    train_graph, {'u-r': train_graph.edge_ids(
        train_src, train_dst, etype='u-r')}, sampler, 
    exclude='reverse_types',
    reverse_etypes={'u-r': 'r-u', 'r-u': 'u-r'},
    negative_sampler=neg_sampler)
val_collator = dgl.dataloading.EdgeCollator(
    val_graph, {'u-r': val_graph.edge_ids(val_src, val_dst, etype='u-r')}, sampler, 
    exclude='reverse_types',
    reverse_etypes={'u-r': 'r-u', 'r-u': 'u-r'},
    negative_sampler=neg_sampler)
test_collator = dgl.dataloading.EdgeCollator(
    graph, {('user', 'u-r', 'recipe'): test_eids}, sampler, 
    exclude='reverse_types',
    reverse_etypes={'u-r': 'r-u', 'r-u': 'u-r'},
    negative_sampler=test_neg_sampler)

train_edgeloader = torch.utils.data.DataLoader(
    train_collator.dataset, collate_fn=train_collator.collate,
    batch_size=1024, shuffle=True, drop_last=False, num_workers=0)
val_edgeloader = torch.utils.data.DataLoader(
    val_collator.dataset, collate_fn=val_collator.collate,
    batch_size=128, shuffle=False, drop_last=False, num_workers=0)
test_edgeloader = torch.utils.data.DataLoader(
    test_collator.dataset, collate_fn=test_collator.collate,
    batch_size=128, shuffle=False, drop_last=False, num_workers=0)

print('# of batches in train_edgeloader: ', len(train_edgeloader))
print('# of batches in val_edgeloader: ', len(val_edgeloader))
print('# of batches in test_edgeloader: ', len(test_edgeloader))
print()

for input_nodes, pos_pair_graph, neg_pair_graph, blocks in train_edgeloader:
    print('blocks: ', blocks)
    break
