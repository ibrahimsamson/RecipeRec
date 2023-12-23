from custom.reciperec_model import *
from splittrain import *
from utils import *
from metrics import *

model = Model().to(device)
opt = torch.optim.Adam(model.parameters(), lr=0.005)
scheduler = torch.optim.lr_scheduler.ExponentialLR(opt, gamma=0.9)

print('start ... ')
for epoch in range(50):
    train_start = time.time()
    epoch_loss = 0
    epoch_contrastive_loss = 0
    epoch_emb_loss = 0
    iteration_cnt = 0
    
    for input_nodes, positive_graph, negative_graph, blocks in train_edgeloader:
        model.train()
        blocks = [b.to(device) for b in blocks]
        positive_graph = positive_graph.to(device)
        negative_graph = negative_graph.to(device)
        
        input_user = blocks[0].srcdata['random_feature']['user']
        input_instr = blocks[0].srcdata['avg_instr_feature']['recipe']
        input_ingredient = blocks[0].srcdata['nutrient_feature']['ingredient']
        ingredient_of_dst_recipe = blocks[1].srcdata['nutrient_feature']['ingredient']
        input_features = [input_user, input_instr, input_ingredient, ingredient_of_dst_recipe]
        
        pos_score, neg_score, x1, x2 = model(positive_graph, negative_graph, blocks, input_features)
        contrastive_loss = get_contrastive_loss(x1, x2)
        # emb_loss = get_emb_loss(x1, x2)
        assert not math.isnan(contrastive_loss)        
        recommendation_loss = get_recommendation_loss(pos_score, neg_score)
        assert not math.isnan(recommendation_loss)
        
        loss = recommendation_loss + 0.1 * contrastive_loss # + 1e-5 * emb_loss
        opt.zero_grad()
        loss.backward()
        opt.step()
        
        epoch_loss += recommendation_loss.item()
        epoch_contrastive_loss += contrastive_loss.item()
        # epoch_emb_loss += emb_loss.item()
        iteration_cnt += 1

        # break
        
    epoch_loss /= iteration_cnt
    epoch_contrastive_loss /= iteration_cnt
    train_end = time.strftime("%M:%S min", time.gmtime(time.time()-train_start))
    
    print('Epoch: {0},  Loss: {l:.4f}, Contrastive: {cl:.4f}, Emb: {el:.4f},  Time: {t}, LR: {lr:.6f}'
          .format(epoch, l=epoch_loss, cl=epoch_contrastive_loss, el=epoch_emb_loss, t=train_end, lr=opt.param_groups[0]['lr']))
    scheduler.step()
    
    # Evaluation
    # For demonstration purpose, only test set result is reported here. Please use val_dataloader for comprehensiveness.
    if epoch >= 4 and epoch % 1 == 0:
        print('testing: ')
        evaluate(model, test_edgeloader, multi_metrics=True)
        print()
        print()
