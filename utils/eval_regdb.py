import os
import logging
import numpy as np
from sklearn.preprocessing import normalize
from utils.relevancemetric import graph_reasoning, re_ranking

def get_unique(array):
    _, idx = np.unique(array, return_index=True)
    return array[np.sort(idx)]


def get_cmc_multi_cam(Y_gallery, cam_gallery, Y_probe, cam_probe, dist):
    # dist = #probe x #gallery
    num_probes, num_gallery = dist.shape
    gallery_unique_count = get_unique(Y_gallery).shape[0]
    match_counter = np.zeros((gallery_unique_count))

    # sort the distance matrix
    sorted_indices = np.argsort(dist, axis=-1)

    Y_result = Y_gallery[sorted_indices]
    cam_locations_result = cam_gallery[sorted_indices]

    valid_probe_sample_count = 0

    for probe_index in range(num_probes):
        # remove gallery samples from the same camera of the probe
        Y_result_i = Y_result[probe_index, :]
        Y_result_i[cam_locations_result[probe_index, :] == cam_probe[probe_index]] = -1

        # remove the -1 entries from the label result
        # print(Y_result_i.shape)
        Y_result_i = np.array([i for i in Y_result_i if i != -1])
        # print(Y_result_i.shape)

        # remove duplicated id in "stable" manner
        Y_result_i_unique = get_unique(Y_result_i)
        # print(Y_result_i_unique, gallery_unique_count)

        # match for probe i
        match_i = Y_result_i_unique == Y_probe[probe_index]

        if np.sum(match_i) != 0:  # if there is true matching in gallery
            valid_probe_sample_count += 1
            match_counter += match_i

    rankk = match_counter / valid_probe_sample_count
    cmc = np.cumsum(rankk)
    return cmc


def get_mAP_multi_cam(Y_gallery, cam_gallery, Y_probe, cam_probe, dist):
    # dist = #probe x #gallery
    num_probes, num_gallery = dist.shape

    # sort the distance matrix
    sorted_indices = np.argsort(dist, axis=-1)

    Y_result = Y_gallery[sorted_indices]
    cam_locations_result = cam_gallery[sorted_indices]

    valid_probe_sample_count = 0
    avg_precision_sum = 0

    for probe_index in range(num_probes):
        # remove gallery samples from the same camera of the probe
        Y_result_i = Y_result[probe_index, :]
        Y_result_i[cam_locations_result[probe_index, :] == cam_probe[probe_index]] = -1

        # remove the -1 entries from the label result
        # print(Y_result_i.shape)
        Y_result_i = np.array([i for i in Y_result_i if i != -1])
        # print(Y_result_i.shape)

        # match for probe i
        match_i = Y_result_i == Y_probe[probe_index]
        true_match_count = np.sum(match_i)

        if true_match_count != 0:  # if there is true matching in gallery
            valid_probe_sample_count += 1
            true_match_rank = np.where(match_i)[0]

            ap = np.mean(
                np.array(range(1, true_match_count + 1)) / (true_match_rank + 1)
            )
            avg_precision_sum += ap

    mAP = avg_precision_sum / valid_probe_sample_count
    return mAP

from scipy.spatial.distance import cdist

def eval_regdb(query_feats, query_ids, query_cam_ids, gallery_feats, gallery_ids, gallery_cam_ids, gallery_img_paths=None,
              mode='all', num_trials=1):

    
    # query_feats, gallery_feats = normalize(query_feats, axis=1), normalize(gallery_feats, axis=1)
    gallery_feats, query_feats = normalize(query_feats, axis=1), normalize(gallery_feats, axis=1)


    dist_p_g = cdist(query_feats, gallery_feats).astype(np.float16)  #np ng
    dist_g_g = cdist(gallery_feats, gallery_feats).astype(np.float16)  #ng ng
    

    dist = graph_reasoning(dist_p_g, dist_g_g, lambda_rgb=0.1, topk=5)
    dist = re_ranking(dist, dist_g_g, k0=9, lambda_value = 0.7)
    # dist = dist_p_g
    # dist = dist_p_g
    cmc = get_cmc_multi_cam(
        gallery_ids, gallery_cam_ids, query_ids, query_cam_ids, dist
    )
    mAP = get_mAP_multi_cam(
        gallery_ids, gallery_cam_ids, query_ids, query_cam_ids, dist
    )
    r1, r5, r10, r20 = cmc[0], cmc[4], cmc[9], cmc[19]
    perf = 'mAP = %f , r1 precision = %f , r5 precision = %f , r10 precision = %f , r20 precision = %f'
    logging.info(perf % (mAP / num_trials, r1 / num_trials, r5 / num_trials, r10 / num_trials, r20 / num_trials))
    # for lambda_rgb in [0.005, 0.01, 0.05, 0.1,0.2,0.5,1.0]:
    #     print("lambda_rgb: {}".format(lambda_rgb))
    #     dist = graph_reasoning(dist_p_g, dist_g_g, lambda_rgb=lambda_rgb)
    #     # dist = dist_p_g
    #     cmc = get_cmc_multi_cam(
    #         gallery_ids, gallery_cam_ids, query_ids, query_cam_ids, dist
    #     )
    #     mAP = get_mAP_multi_cam(
    #         gallery_ids, gallery_cam_ids, query_ids, query_cam_ids, dist
    #     )
    #     r1, r5, r10, r20 = cmc[0], cmc[4], cmc[9], cmc[19]
    #     perf = 'mAP = %f , r1 precision = %f , r5 precision = %f , r10 precision = %f , r20 precision = %f'
    #     logging.info(perf % (mAP / num_trials, r1 / num_trials, r5 / num_trials, r10 / num_trials, r20 / num_trials))

    return mAP, r1, r5, r10, r20
