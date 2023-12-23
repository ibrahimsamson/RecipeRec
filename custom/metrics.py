from custom.utils import *


def compute_auc(pos_score, neg_score):
    scores = torch.cat([pos_score, neg_score]).cpu().numpy()
    labels = torch.cat([torch.ones(pos_score.shape[0]), torch.zeros(neg_score.shape[0])]).cpu().numpy()
    return roc_auc_score(labels, scores)

def precision_at_k(r, k):
    # Relevance is binary (nonzero is relevant).
    assert k >= 1
    r = np.asarray(r)[:k]
    return np.mean(r)

def recall_at_k(r, k, all_pos_num):
    r = np.asfarray(r)[:k]
    return np.sum(r) / all_pos_num

def dcg_at_k(r, k, method=0):
    # method: If 0 then weights are [1.0, 1.0, 0.6309, 0.5, 0.4307, ...]
    #         If 1 then weights are [1.0, 0.6309, 0.5, 0.4307, ...]
    r = np.asfarray(r)[:k]
    if r.size:
        if method == 0:
            return r[0] + np.sum(r[1:] / np.log2(np.arange(2, r.size + 1)))
        elif method == 1:
            return np.sum(r / np.log2(np.arange(2, r.size + 2)))
        else:
            raise ValueError('method must be 0 or 1.')
    return 0.

def ndcg_at_k(r, k, method=0):
    dcg_max = dcg_at_k(sorted(r, reverse=True), k, method)
    if not dcg_max:
        return 0.
    return dcg_at_k(r, k, method) / dcg_max


def average_precision_at_k(r, Ks):
    r = np.asarray(r) != 0
    out = []
    for k in Ks:
        assert k <= len(r)
        # print('[precision_at_k(r, i + 1) for i in range(k) if r[i]]: ', [precision_at_k(r, i + 1) for i in range(k) if r[i]])
        all_precision_before_k = [precision_at_k(r, i + 1) for i in range(k) if r[i]]
        if len(all_precision_before_k) == 0:
            all_precision_before_k = [0]
        out.append(np.mean(all_precision_before_k))
    if not out:
        return 0.
    # return np.array([np.mean(out)])
    return np.array(out)

def get_map_at_k(rs, Ks):
    # examples:
    # average_precision_at_k([1, 1, 0, 1, 0, 1, 0, 0, 0, 1], [5,10])
    # average_precision_at_k([0, 0, 1, 0, 0, 0, 0, 0, 0, 1], [5,10])
    # get_map_at_k([[1, 1, 0, 1, 0, 1, 0, 0, 0, 1,1], [0, 0, 1, 0, 0, 0, 0, 0, 0, 1,1]], [5,10])
    out = np.zeros(len(Ks))
    for r in rs:
        # print('average_precision_at_k: ', average_precision_at_k(r, Ks))
        out += average_precision_at_k(r, Ks)/len(rs)
    return out
    # return np.mean([average_precision(r) for r in rs])

def get_ranklist_for_one_user(user_poss, user_negs, Ks):
    item_scores = {}
    n_pos = len(user_poss)
    n_neg = len(user_negs)
    for i in range(n_pos):
        item_scores[i] = user_poss[i]
    for i in range(n_neg):
        item_scores[i+1] = user_negs[i]
        
    K_max = max(Ks)
    K_max_item_score = heapq.nlargest(K_max, item_scores, key=item_scores.get)
    
    r = []
    for i in K_max_item_score:
        if i < n_pos:
            r.append(1)
        else:
            r.append(0)
    return r

def hit_at_k(r, k):
    r = np.array(r)[:k]
    if np.sum(r) > 0:
        return 1.
    else:
        return 0.

def get_mrr(rs):
    rs = (np.asarray(r).nonzero()[0] for r in rs)
    return np.mean([1. / (r[0] + 1) if r.size else 0. for r in rs])


def get_performance_one_user(user_poss, user_negs, Ks):
    r = get_ranklist_for_one_user(user_poss, user_negs, Ks)
    
    precision, recall, ndcg, hit_ratio = [], [], [], []
    for K in Ks:
        precision.append(precision_at_k(r, K))
        # recall.append(recall_at_k(r, K, len(user_poss))) 
        ndcg.append(ndcg_at_k(r, K))
        hit_ratio.append(hit_at_k(r, K))
    # return {'precision': np.array(precision), 'recall': np.array(recall),
    #         'ndcg': np.array(ndcg), 'hit_ratio': np.array(hit_ratio)}, r    
    return {'precision': np.array(precision), 
            'ndcg': np.array(ndcg), 'hit_ratio': np.array(hit_ratio)}, r


def get_performance_all_users(user2pos_score_dict, user2neg_score_dict, Ks):
    # all_result = {'precision': np.zeros(len(Ks)), 'recall': np.zeros(len(Ks)), 'ndcg': np.zeros(len(Ks)),
    #           'hit_ratio': np.zeros(len(Ks))}
    all_result = {'hit_ratio': np.zeros(len(Ks)), 'ndcg': np.zeros(len(Ks)), 'precision': np.zeros(len(Ks))}
    
    rs = []
    n_test_users = len(user2pos_score_dict)
    
    # one specific user
    for user in user2pos_score_dict.keys():
        user_pos_score = user2pos_score_dict[user]
        user_neg_score = user2neg_score_dict[user]
        one_result, one_r = get_performance_one_user(user_pos_score, user_neg_score, Ks)
        # all_result['recall'] += one_result['recall']/n_test_users
        all_result['hit_ratio'] += one_result['hit_ratio']/n_test_users
        all_result['ndcg'] += one_result['ndcg']/n_test_users
        all_result['precision'] += one_result['precision']/n_test_users
        rs.append(one_r)
        
    # get MRR
    # MRR = get_mrr(rs)
    # all_result['MRR'] = MRR
    
    # get MAP
    MAP = get_map_at_k(rs, Ks)
    all_result['MAP'] = MAP

    return all_result
    
    
def evaluate(model, dataloader, multi_metrics=False):
    # print('start evaluating ...')
    evaluate_start = time.time()
    model.eval()
    total_loss = 0
    epoch_contrastive_loss = 0
    iteration_cnt = 0
    total_pos_score = torch.tensor([]).to(device)
    total_neg_score = torch.tensor([]).to(device)
    
    # for evaluation
    user2pos_score_dict = {}
    user2neg_score_dict = {}
    
    with torch.no_grad():
        for input_nodes, positive_graph, negative_graph, blocks in dataloader:
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
            total_pos_score = torch.cat([total_pos_score, pos_score])
            total_neg_score = torch.cat([total_neg_score, neg_score])

            recommendation_loss = get_recommendation_loss(pos_score, neg_score)
            loss = recommendation_loss # + 0.01 * contrastive_loss      
            total_loss += recommendation_loss.item()
            epoch_contrastive_loss += contrastive_loss.item()
            iteration_cnt += 1
            
            # for evaluation
            global_test_users = blocks[1].dstdata['_ID']['user'] # we need to map the user id in subgraph to the whole graph
            test_users, test_recipes = positive_graph.edges(etype='u-r')
            test_users = test_users.tolist()
            test_recipes = test_recipes.tolist()
            for index in range(len(test_users)):
                test_u = int(global_test_users[test_users[index]])
                test_r = int(test_recipes[index])
                test_score = float(pos_score[index])
                
                if test_u not in user2pos_score_dict:
                    user2pos_score_dict[test_u] = []
                user2pos_score_dict[test_u].append(test_score)
                
                if test_u not in user2neg_score_dict:
                    user2neg_score_dict[test_u] = neg_score[index*n_test_negs:(index+1)*n_test_negs]
                
            # break
            
        total_loss /= iteration_cnt
        epoch_contrastive_loss /= iteration_cnt
        
        # metrics
        auc = compute_auc(total_pos_score, total_neg_score)
        if multi_metrics:
            # evaluation_result = get_performance_all_users(total_pos_score, total_neg_score, Ks)
            evaluation_result = get_performance_all_users(user2pos_score_dict, user2neg_score_dict, Ks)
            evaluation_result['AUC'] = auc
            print('evaluation_result: ', evaluation_result)
            print('epoch_contrastive_loss: ', epoch_contrastive_loss)
        else:
            print('AUC: ', auc)
        
        evalutate_time = time.strftime("%M:%S min", time.gmtime(time.time()-evaluate_start))
        print('evalutate_time: ', evalutate_time)
    return total_loss