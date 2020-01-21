import numpy as np
import copy
def graph_reasoning(dist_p_g, dist_g_g, lambda_rgb=0.01, topk=9):
    #similarity graph reasoning
    dist_p_g = copy.deepcopy(dist_p_g)
    dist_g_g = copy.deepcopy(lambda_rgb* dist_g_g)
    gi_neibor_index = np.argsort(dist_g_g)[:,:20]
    for j in range(dist_p_g.shape[1]):

        dist_p_k = dist_p_g[:,gi_neibor_index[j]] + dist_g_g[j, gi_neibor_index[j]]
        dist_p_gj = np.sort(dist_p_k, axis=1)[:,:topk].mean(axis=1)
        dist_p_g[:,j] = dist_p_gj


    return dist_p_g

def re_ranking(q_g_dist, g_g_dist, k0=12, k1=20, k2=6, lambda_value=0.3):

    original_dist = np.concatenate([q_g_dist.T, g_g_dist], axis=1)
    original_dist = np.power(original_dist, 2).astype(np.float32)
    original_dist = np.transpose(1. * original_dist/np.max(original_dist,axis = 0))

    query_num = q_g_dist.shape[0]
    gallery_num = q_g_dist.shape[1]
    all_num = q_g_dist.shape[0] + q_g_dist.shape[1]
    
    q_g_dist = original_dist[:query_num,]
    g_g_dist = original_dist[query_num:,]

    V_q = np.zeros_like(q_g_dist).astype(np.float32)
    initial_rank_q = np.argsort(q_g_dist).astype(np.int32)

    V_g = np.zeros_like(g_g_dist).astype(np.float32)
    initial_rank_g = np.argsort(g_g_dist).astype(np.int32)    


    ##q g similarity matrix
    for i in range(query_num):
        # k-reciprocal neighbors
        forward_k_neigh_index = initial_rank_q[i,:k0+1]
        k_reciprocal_expansion_index = forward_k_neigh_index

        k_reciprocal_expansion_index = np.unique(k_reciprocal_expansion_index)
        weight = np.exp(-q_g_dist[i,k_reciprocal_expansion_index])
        V_q[i,k_reciprocal_expansion_index] = 1.*weight/np.sum(weight)    


    # g g similarity matrix
    for i in range(gallery_num):
        # k-reciprocal neighbors
        forward_k_neigh_index = initial_rank_g[i,:k1+1]
        backward_k_neigh_index = initial_rank_g[forward_k_neigh_index,:k1+1]
        fi = np.where(backward_k_neigh_index==i)[0]
        k_reciprocal_index = forward_k_neigh_index[fi]
        # import pdb;pdb.set_trace()
        k_reciprocal_expansion_index = k_reciprocal_index
        for j in range(len(k_reciprocal_index)):
            candidate = k_reciprocal_index[j]
            candidate_forward_k_neigh_index = initial_rank_g[candidate,:int(np.around(k1/2.))+1]
            candidate_backward_k_neigh_index = initial_rank_g[candidate_forward_k_neigh_index,:int(np.around(k1/2.))+1]
            fi_candidate = np.where(candidate_backward_k_neigh_index == candidate)[0]
            candidate_k_reciprocal_index = candidate_forward_k_neigh_index[fi_candidate]
            if len(np.intersect1d(candidate_k_reciprocal_index,k_reciprocal_index))> 2./3*len(candidate_k_reciprocal_index):
                k_reciprocal_expansion_index = np.append(k_reciprocal_expansion_index,candidate_k_reciprocal_index)

        k_reciprocal_expansion_index = np.unique(k_reciprocal_expansion_index)
        weight = np.exp(-g_g_dist[i,k_reciprocal_expansion_index])
        V_g[i,k_reciprocal_expansion_index] = 1.*weight/np.sum(weight)

    
    V = np.concatenate([V_q,V_g], axis=0)
    V = np.concatenate([np.zeros((all_num, query_num)),V], axis=1)
    initial_rank = np.concatenate([initial_rank_q, initial_rank_g], axis=0)
    
    original_dist = np.concatenate([np.zeros((query_num, query_num)), q_g_dist], axis=1)

    del initial_rank
    invIndex = []
    for i in range(all_num):
        invIndex.append(np.where(V[:,i] != 0)[0]) # row indx

    jaccard_dist = np.zeros_like(original_dist,dtype = np.float32)


    for i in range(query_num):
        temp_min = np.zeros(shape=[1,all_num],dtype=np.float32)
        indNonZero = np.where(V[i,:] != 0)[0]  #col indx
        indImages = []
        indImages = [invIndex[ind] for ind in indNonZero]
        for j in range(len(indNonZero)):
            temp_min[0,indImages[j]] = temp_min[0,indImages[j]]+ np.minimum(V[i,indNonZero[j]],V[indImages[j],indNonZero[j]])
        jaccard_dist[i] = 1-temp_min/(2.-temp_min)

    final_dist = jaccard_dist*(1-lambda_value) + original_dist*lambda_value
    del original_dist
    del V
    del jaccard_dist
    final_dist = final_dist[:query_num,query_num:]
    return final_dist