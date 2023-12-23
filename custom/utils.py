from imports import *

def norm(input, p=1, dim=1, eps=1e-12):
    return input / input.norm(p, dim, keepdim=True).clamp(min=eps).expand_as(input)

def get_recommendation_loss(pos_score, neg_score):
    n = pos_score.shape[0]
    return (neg_score.view(n, -1) - pos_score.view(n, -1) + 1).clamp(min=0).mean()

def get_contrastive_loss(x1, x2):
    temperature = 0.07
    
    # users
    x1_user, x2_user = F.normalize(x1['user']), F.normalize(x2['user'])
    pos_score_user = torch.mul(x1_user, x2_user).sum(dim=1)
    pos_score_user = torch.exp(pos_score_user/temperature)

    x2_user_neg = torch.flipud(x2_user)
    ttl_score_user = torch.mul(x1_user, x2_user_neg).sum(dim=1)
    ttl_score_user = pos_score_user + torch.exp(ttl_score_user/temperature)
    
    contrastive_loss_user = - torch.log(pos_score_user/ttl_score_user).mean()
    # print('contrastive_loss_user: ', contrastive_loss_user)
    assert not math.isnan(contrastive_loss_user)

    
    # recipes
    x1_recipe, x2_recipe = F.normalize(x1['recipe']), F.normalize(x2['recipe'])
    pos_score_recipe = torch.mul(x1_recipe, x2_recipe).sum(dim=1)
    pos_score_recipe = torch.exp(pos_score_recipe/temperature)

    x2_recipe_neg = torch.flipud(x2_recipe)
    ttl_score_recipe = torch.mul(x1_recipe, x2_recipe_neg).sum(dim=1)
    ttl_score_recipe = pos_score_recipe + torch.exp(ttl_score_recipe/temperature) #.sum(dim=1)
    
    contrastive_loss_recipe = - torch.log(pos_score_recipe/ttl_score_recipe).mean()
    # print('contrastive_loss_recipe: ', contrastive_loss_recipe)
    
    return contrastive_loss_user + contrastive_loss_recipe
    
def get_emb_loss(*params):
    out = None
    for param in params:
        for k,v in param.items():
            if out == None:
                out = (v**2/2).mean()
            else:
                out += (v**2/2).mean()
    return out