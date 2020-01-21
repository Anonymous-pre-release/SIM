import os
import logging
import numpy as np
from sklearn.preprocessing import normalize
from utils.relevancemetric import graph_reasoning, re_ranking
from utils.visualize import visualize_rank_list

def get_gallery_names(perm, cams, ids, trial_id, num_shots=1):
    names = []
    for cam in cams:
        cam_perm = perm[cam - 1][0].squeeze()
        for i in ids:
            instance_id = cam_perm[i - 1][trial_id][:num_shots]
            names.extend(['cam{}/{:0>4d}/{:0>4d}'.format(cam, i, ins) for ins in instance_id.tolist()])

    return names


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
            try:
                match_counter += match_i
            except:
                match_counter[:match_i.shape[0]] += match_i

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


def eval_sysu(query_feats, query_ids, query_cam_ids, gallery_feats, gallery_ids, gallery_cam_ids, gallery_img_paths,
              perm, mode='all', num_shots=10, num_trials=10):
    assert mode in ['indoor', 'all']

    gallery_cams = [1, 2] if mode == 'indoor' else [1, 2, 4, 5]

    # cam2 and cam3 are in the same location
    query_cam_ids[np.equal(query_cam_ids, 3)] = 2

    gallery_indices = np.in1d(gallery_cam_ids, gallery_cams)
    query_feats = normalize(query_feats, axis=1)
    gallery_feats = normalize(gallery_feats[gallery_indices], axis=1)
    gallery_cam_ids = gallery_cam_ids[gallery_indices]
    gallery_ids = gallery_ids[gallery_indices]
    gallery_img_paths = gallery_img_paths[gallery_indices]
    gallery_names = np.array(['/'.join(os.path.splitext(path)[0].split('/')[-3:]) for path in gallery_img_paths])

    gallery_id_set = np.unique(gallery_ids)
    
    mAP, r1, r5, r10, r20 = 0, 0, 0, 0, 0
    for t in range(num_trials):
        print("Evaluate trial {}...".format(t+1))
        names = get_gallery_names(perm, gallery_cams, gallery_id_set, t, num_shots)
        flag = np.in1d(gallery_names, names)

        g_feat = gallery_feats[flag]
        g_ids = gallery_ids[flag]
        g_cam_ids = gallery_cam_ids[flag]

        dist_p_g = cdist(query_feats, g_feat).astype(np.float16)  #np ng
        dist_g_g = cdist(g_feat, g_feat).astype(np.float16)  #ng ng

        dist = graph_reasoning(dist_p_g, dist_g_g, lambda_rgb=0.01, topk=9)
        dist = re_ranking(dist, dist_g_g)

        cmc = get_cmc_multi_cam(
            g_ids, g_cam_ids, query_ids, query_cam_ids, dist
        )
        mAP += get_mAP_multi_cam(
            g_ids, g_cam_ids, query_ids, query_cam_ids, dist
        )
        r1 += cmc[0]
        r5 += cmc[4]
        r10 += cmc[9]
        r20 += cmc[19]

    perf = 'mAP = %f , r1 precision = %f , r5 precision = %f , r10 precision = %f , r20 precision = %f'
    logging.info(perf % (mAP / num_trials, r1 / num_trials, r5 / num_trials, r10 / num_trials, r20 / num_trials))
    return mAP, r1, r5, r10, r20

def visualize_sysu(query_feats, query_ids, query_cam_ids, query_img_paths, gallery_feats, gallery_ids, gallery_cam_ids, gallery_img_paths,
              perm, mode='all', num_shots=10, num_trials=1):
    assert mode in ['indoor', 'all']

    gallery_cams = [1, 2] if mode == 'indoor' else [1, 2, 4, 5]

    # cam2 and cam3 are in the same location
    query_cam_ids[np.equal(query_cam_ids, 3)] = 2

    gallery_indices = np.in1d(gallery_cam_ids, gallery_cams)
    query_feats = normalize(query_feats, axis=1)
    # gallery_feats = gallery_feats[gallery_indices]
    gallery_feats = normalize(gallery_feats[gallery_indices], axis=1)
    gallery_cam_ids = gallery_cam_ids[gallery_indices]
    gallery_ids = gallery_ids[gallery_indices]
    gallery_img_paths = gallery_img_paths[gallery_indices]
    gallery_names = np.array(['/'.join(os.path.splitext(path)[0].split('/')[-3:]) for path in gallery_img_paths])

    gallery_id_set = np.unique(gallery_ids)

    print(query_feats.shape, gallery_feats.shape)

    print("calc performance")


    mAP, r1, r5, r10, r20 = 0, 0, 0, 0, 0
    for t in range(num_trials):
        print("Evaluate trial {}...".format(t+1))
        names = get_gallery_names(perm, gallery_cams, gallery_id_set, t, num_shots)
        flag = np.in1d(gallery_names, names)
        g_feat = gallery_feats[flag]
        g_ids = gallery_ids[flag]
        g_cam_ids = gallery_cam_ids[flag]
        g_paths = gallery_img_paths[flag]
        print("calc distance")
        dist_p_g = cdist(query_feats, g_feat).astype(np.float16)  #np ng
        dist_g_g = cdist(g_feat, g_feat).astype(np.float16)  #ng ng
        print("perform SIM")
        dist = graph_reasoning(dist_p_g, dist_g_g, lambda_rgb=0.01, topk=9)
        dist = re_ranking(dist, dist_g_g)
        print("visualize sysu")
        visualize_rank_list(dist_p_g, dist, query=query_img_paths, gallery=g_paths, query_ids=query_ids, gallery_ids=g_ids, query_cams=query_cam_ids, gallery_cams=g_cam_ids)
        cmc = get_cmc_multi_cam(
            g_ids, g_cam_ids, query_ids, query_cam_ids, dist
        )
        mAP += get_mAP_multi_cam(
            g_ids, g_cam_ids, query_ids, query_cam_ids, dist
        )
        r1 += cmc[0]
        r5 += cmc[4]
        r10 += cmc[9]
        r20 += cmc[19]


    perf = 'mAP = %f , r1 precision = %f , r5 precision = %f , r10 precision = %f , r20 precision = %f'
    logging.info(perf % (mAP / num_trials, r1 / num_trials, r5 / num_trials, r10 / num_trials, r20 / num_trials))
    return mAP, r1, r5, r10, r20