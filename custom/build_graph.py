from .imports import *

def get_graph():
    print('generating graph ...')
    edge_src, edge_dst, r_i_edge_weight = \
        torch.load(dataset_folder+'/edge_r2i_src_dst_weight.pt')
    recipe_edge_src, recipe_edge_dst, recipe_edge_weight = \
        torch.load(dataset_folder+'/edge_r2r_src_and_dst_and_weight.pt')
    ingre_edge_src, ingre_edge_dst, ingre_edge_weight = \
        torch.load(dataset_folder+'/edge_i2i_src_and_dst_and_weight.pt')
    all_u2r_src_dst_weight, train_u2r_src_dst_weight, val_u2r_src_dst_weight, \
        test_u2r_src_dst_weight = \
            torch.load(dataset_folder+'/all_train_val_test_edge_u_rate_r_src_and_dst_and_weight.pt')
    u_rate_r_edge_src, u_rate_r_edge_dst, u_rate_r_edge_weight = all_u2r_src_dst_weight
    
    # nodes and edges
    graph = dgl.heterograph({
        ('recipe', 'r-i', 'ingredient'): (edge_src, edge_dst),
        ('ingredient', 'i-r', 'recipe'): (edge_dst, edge_src),
        ('recipe', 'r-r', 'recipe'): (recipe_edge_src, recipe_edge_dst),
        ('ingredient', 'i-i', 'ingredient'): (ingre_edge_src, ingre_edge_dst),
        ('user', 'u-r', 'recipe'): (u_rate_r_edge_src, u_rate_r_edge_dst),
        ('recipe', 'r-u', 'user'): (u_rate_r_edge_dst, u_rate_r_edge_src)
    })

    # edge weight
    graph.edges['r-i'].data['weight'] = torch.FloatTensor(r_i_edge_weight)
    graph.edges['i-r'].data['weight'] = torch.FloatTensor(r_i_edge_weight)
    graph.edges['r-r'].data['weight'] = torch.FloatTensor(recipe_edge_weight)
    graph.edges['i-i'].data['weight'] = torch.FloatTensor(ingre_edge_weight)
    graph.edges['u-r'].data['weight'] = torch.FloatTensor(u_rate_r_edge_weight)
    graph.edges['r-u'].data['weight'] = torch.FloatTensor(u_rate_r_edge_weight)
    
    # node features
    recipe_nodes_avg_instruction_features = torch.load(
        dataset_folder+'/recipe_nodes_avg_instruction_features.pt')
    ingredient_nodes_nutrient_features_minus1 = torch.load(
        dataset_folder+'/ingredient_nodes_nutrient_features.pt')
    graph.nodes['recipe'].data['avg_instr_feature'] = recipe_nodes_avg_instruction_features
    graph.nodes['ingredient'].data['nutrient_feature'] = ingredient_nodes_nutrient_features_minus1
    graph.nodes['user'].data['random_feature'] = torch.nn.init.xavier_normal_(torch.ones(7959, 300))
    graph.nodes['recipe'].data['random_feature'] = torch.nn.init.xavier_normal_(torch.ones(68794, 1024))

    return graph

graph = get_graph()
print('graph: ', graph)