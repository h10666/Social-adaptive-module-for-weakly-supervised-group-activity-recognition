import numpy as np
import pickle
import time
import os
from sklearn.cluster import KMeans
import utils

def cluster(data, n_clusters):
    km=KMeans(n_clusters).fit(data)
    rs_labels=km.labels_
    rs_center_ids=km.cluster_centers_
    return rs_labels

def normlize_boxes(boxes, K, scale=[1, 1]):
    # Normlizing box:(top, bottom, left, right) to box:(left, top, right, bottom). i.e., (x1, y1, x2, y2)
    normlized_boxes = np.zeros((K, 4))
    for idx in range(K):
        top, bottom, left, right = boxes[idx] if idx<len(boxes) else boxes[-1]
        normlized_boxes[idx, :] = np.array([left*scale[1], top*scale[0], right*scale[1], bottom*scale[0]])
    return normlized_boxes

def formate_boxes(self, boxes, K, img_size=[720, 1280]):
    # Formating box:(r_y1, r_x1, r_y2, r_x2) to box:(x1, y1, x2, y2) according to the size of image. 'r' denotes relative
    formated_boxes = torch.Tensor(K, 4)
    H, W = img_size
    for idx in range(K):
        r_y1, r_x1, r_y2, r_x2 = boxes[idx] if idx<len(boxes) else boxes[-1]
        formated_boxes[idx, :] = torch.tensor([r_x1*W, r_y1*H, r_x2*W, r_y2*H])
    return formated_boxes

# if __name__=='__main__':
#     K = 12
#     num_regions = 12
#     dataset_root = '/cache/dataset/VD/videos'
#     # load the bboxes dict
#     bboxes_dict = pickle.load(open(os.path.join(dataset_root, 'bboxes.pkl')))
#     cluster_dict = {}
#     since = time.time()
#     # iterring the boxes
#     total = len(bboxes_dict.keys())
#     for i, key in enumerate(bboxes_dict.keys()):
#         centers = np.zeros((K, 2))
#         bboxes = normlize_boxes(np.array(bboxes_dict[key]), K)
#         frame_cluster_dict={}
#         # box:(left, top, right, bottom)
#         centers[:,0], centers[:,1] = (bboxes[:,0] + bboxes[:,2])/2, (bboxes[:,1] + bboxes[:,3])/2
#         for n in range(1, num_regions+1):
#             frame_cluster_dict[n] = cluster(centers, n)
#         cluster_dict[key] = frame_cluster_dict
#         if i%100==0:
#             print i, '/', total, ', takes', (time.time()-since), 's'

#     # print(cluster_dict)
#     utils.save_pkl(cluster_dict, os.path.join(dataset_root, 'cluster'))
    


if __name__=='__main__':
    K = 12
    num_regions = 12
    dataset_root = '/cache/dataset/VD/videos'
    # load the bboxes dict
    bboxes_dict = pickle.load(open(os.path.join(dataset_root, 'normalized_bboxes.pkl'), 'rb'))
    cluster_dict = {}
    since = time.time()
    # iterring the boxes
    total = len(bboxes_dict.keys())
    for i, video_clip_id in enumerate(bboxes_dict.keys()):
        video_id, clip_id = video_clip_id
        cluster_dict[(video_id, clip_id)] = {}
        for frame_id in bboxes_dict[(video_id, clip_id)].keys():
            centers = np.zeros((K, 2))
            bboxes = normlize_boxes(np.array(bboxes_dict[(video_id, clip_id)][frame_id]), K)
            frame_cluster_dict={}
            # box:(x1, y1, x2, y2)
            centers[:,0], centers[:,1] = (bboxes[:,0] + bboxes[:,2])/2, (bboxes[:,1] + bboxes[:,3])/2
            for n in range(1, num_regions+1):
                frame_cluster_dict[n] = cluster(centers, n)
            cluster_dict[(video_id, clip_id)][frame_id] = frame_cluster_dict
        if i%10==0:
            print(i, '/', total, ', takes', (time.time()-since), 's')

    # print(cluster_dict)
    utils.save_pkl(cluster_dict, os.path.join(dataset_root, 'cluster'))