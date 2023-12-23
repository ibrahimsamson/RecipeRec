from set_transformer import *
from utils import *
from gnn import *
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.user_embedding = nn.Sequential(
            nn.Linear(300, 128),
            nn.ReLU(),
        )
        self.instr_embedding = nn.Sequential(
            nn.Linear(1024, 128),
            nn.ReLU(),
        )
        self.ingredient_embedding = nn.Sequential(
            nn.Linear(46, 128),
            nn.ReLU()
        )
        self.recipe_combine2out = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU()
        )
        self.gnn = GNN(128, 128, 128, graph.etypes)
        self.pred = ScorePredictor()
        self.setTransformer_ = SetTransformer()

    def forward(self, positive_graph, negative_graph, blocks, input_features):
        user, instr, ingredient, ingredient_of_dst_recipe = input_features
        
        # major GNN
        user_major = self.user_embedding(user)
        user_major = norm(user_major)
        instr_major = self.instr_embedding(instr)
        instr_major = norm(instr_major)
        ingredient_major = self.ingredient_embedding(ingredient)
        ingredient_major = norm(ingredient_major)
        x = self.gnn(blocks, {'user': user_major, 
                              'recipe': instr_major, 
                              'ingredient': ingredient_major}, 
                              torch.Tensor([[0]]))
        
        # contrastive - 1
        user1 = node_drop(user, 0.1, model.training)
        instr1 = node_drop(instr, 0.1, model.training)
        ingredient1 = node_drop(ingredient, 0.1, model.training)

        user1 = self.user_embedding(user1)
        user1 = norm(user1)
        instr1 = self.instr_embedding(instr1)
        instr1 = norm(instr1)
        ingredient1 = self.ingredient_embedding(ingredient1)
        ingredient1 = norm(ingredient1)
        
        x1 = self.gnn(blocks, {'user': user1, 'recipe': instr1, 
                               'ingredient': ingredient1}, torch.Tensor([[1]]))
        
        # contrastive - 2
        user2 = node_drop(user, 0.1, model.training)
        instr2 = node_drop(instr, 0.1, model.training)
        ingredient2 = node_drop(ingredient, 0.1, model.training)
        
        user2 = self.user_embedding(user2)
        user2 = norm(user2)
        instr2 = self.instr_embedding(instr2)
        instr2 = norm(instr2)
        ingredient2 = self.ingredient_embedding(ingredient2)
        ingredient2 = norm(ingredient2)
        
        x2 = self.gnn(blocks, {'user': user2, 'recipe': instr2, 
                               'ingredient': ingredient2}, torch.Tensor([[1]]))
        
        # setTransformer
        all_ingre_emb_for_each_recipe = get_ingredient_neighbors_all_embeddings(
            blocks, blocks[1].dstdata['_ID']['recipe'], ingredient_of_dst_recipe)
        all_ingre_emb_for_each_recipe = norm(all_ingre_emb_for_each_recipe)
        total_ingre_emb = self.setTransformer_(all_ingre_emb_for_each_recipe) # 1
        total_ingre_emb = norm(total_ingre_emb)
        
        # scores
        x['recipe'] = self.recipe_combine2out(total_ingre_emb.add(x['recipe']))
        pos_score = self.pred(positive_graph, x)
        neg_score = self.pred(negative_graph, x)        

        return pos_score, neg_score, x1, x2