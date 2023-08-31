# -*- coding:utf-8 -*-
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import time
import matplotlib.pyplot as plt
import itertools
from sklearn import datasets, svm, metrics
from collections import deque
import pickle
from PIL import Image, ImageDraw
from sklearn.cluster import KMeans
from torchvision.ops.roi_align import roi_align 
import os
from operator import truediv

def get_avg_acc(preds, targets):
    confusion = metrics.confusion_matrix(targets, preds)
    each_acc = np.nan_to_num(truediv(np.diag(confusion), np.sum(confusion, axis=1)))
    return each_acc, np.mean(each_acc)

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
        
def str2tuple(v):
    if isinstance(v, tuple):
        return v
    else:
        try:
            x = tuple(map(int, v.split(',')))
            return x
        except:
            raise argparse.ArgumentTypeError("Coordinates must be tuple as x,y,...")

def get_folders(root):
    dir_list = os.listdir(root)
    folder_list = []
    for dir in dir_list:
        if not os.path.isfile(os.path.join(root, dir)):
            folder_list.append(dir)
    return folder_list


def set_bn_eval(m):
    classname = m.__class__.__name__
    if classname.find('BatchNorm') != -1:
        m.eval()

def normlize(matrix):
    matrix = np.asarray(matrix, dtype=float)
    return np.round(matrix/np.sum(matrix, 1).reshape(-1,1)*100, decimals=2)

def save_pkl(obj, name):
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)
        
def load_pkl(name):
    with open(name + '.pkl', 'rb') as f:
        return pickle.load(f)

def write_txt(txtFile, content_str, mode):
    with open(txtFile, mode) as f:
        f.write(content_str)
        
def get_mask(sizes, person_rects, mode='cv2'):
    mask = np.zeros(sizes, np.uint8)
    for person_rect in person_rects:
        top, bottom, left, right = person_rect
        mask[top:bottom, left:right] = 1
    return mask

def exp_lr_scheduler(optimizer, epoch, init_lr=0.001, lr_decay_epoch=100):
    """Decay learning rate by a factor of 0.1 every lr_decay_epoch epochs."""
    lr = init_lr * (0.1**(epoch // lr_decay_epoch))

    if epoch % lr_decay_epoch == 0:
        print('LR is set to {}'.format(lr))

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    return optimizer


def block_shuffle(List, block_length):
    """
        Args
    """
    base_indx = np.arange(0, len(List), block_length)
    np.random.shuffle(base_indx)
    indx = base_indx
    for i in range(block_length - 1):
        new_indx = base_indx + i + 1
        indx = np.column_stack((indx, new_indx))
    indx = indx.reshape(-1)
    # print indx
    # print List.type()
    #shuffled_List = List[indx]
    shuffled_List = type(List)(map(lambda i: List[i], indx))
    return shuffled_List

def get_confusion_matrix(y_true, y_pred):
    matrix = confusion_matrix(y_true, y_pred)
    matrix = matrix * 100 / matrix.astype(np.float).sum(axis=1).reshape(-1, 1)
    return matrix

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='',
                          cmap=plt.cm.Greys):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    # plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45, fontsize=12)
    plt.yticks(tick_marks, classes, fontsize=12)

    fmt = '.2f' if normalize else '.2f'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    #plt.ylabel('True label')
    #plt.xlabel('Predicted label')


def Min_Max_Normlize(vec):
    """
        Args: vec
    """
    Max, Min = torch.max(vec), torch.min(vec)
    eps = 0.0001
    return (vec - Min) / (Max - Min)


def build_multiscale_feas(feature_maps):
    # Building multiscale features
    OH, OW = feature_maps[0].size()[-2:] # Output_H and Output_W
    multiscale_feas=[]
    for features in feature_maps:
        if features.shape[2:4]!=torch.Size([OH, OW]):
            features=F.interpolate(features, size=(OH, OW), mode='bilinear', align_corners=True)
        multiscale_feas.append(features)
    return torch.cat(multiscale_feas,dim=1)  #b*T, D, OH, OW
    
def build_box_feas(feature_maps, boxes_list, output_crop_size=(3,3), resolution=[720, 1280]):
    # Building feas for each bounding box by using RoI Align
    # feature_maps:[N,C,H,W], where N=b*T
    IH, IW = resolution
    FH, FW = feature_maps.size()[-2:] # Feature_H, Feature_W
    box_feas = roi_align(feature_maps, boxes_list, output_crop_size, spatial_scale=float(FW)/IW) # b*T*K, C, S, S; S denotes output_size
    return box_feas.view(box_feas.size(0),-1) # b*T*K, D*S*S
    
def build_frame_feas(box_feas, num_groups=1):
    #print 'box_feas size: ', box_feas.size() # b*T, K, D'
    box_feas = torch.chunk(box_feas, num_groups, dim=-2)
    group_feas = {}
    for g in range(num_groups):
        group_feas[g], _ = torch.max(box_feas[g], dim=-2)
    frame_feas = torch.cat(tuple(group_feas.values()), -1)
    return frame_feas
    
    
def tensor_to_list(boxes_tensor):
    boxes_list = []
    for boxes in boxes_tensor:
        boxes_list.append(boxes)
    return boxes_list


def cluster(data, n_clusters):
    km=KMeans(n_clusters).fit(data)
    rs_labels=km.labels_
    rs_center_ids=km.cluster_centers_
    return rs_labels

def build_region(subset, resolution, offset=30):
    IH, IW = resolution
    region_left, region_top, region_right, region_bottom = 9999,9999,0,0
    region_left = min(region_left, min(subset[:, 0]))
    region_top = min(region_top, min(subset[:, 1]))
    region_right = max(region_right, max(subset[:, 2]))
    region_bottom = max(region_bottom, max(subset[:, 3]))
    return max(region_left-offset, 0), max(region_top-offset, 0), min(region_right+offset, IW), min(region_bottom+offset, IH)


def reshape_list(data_list):
    # this function is to reshape the list type data from Dataloader
    new_list = []
    for b in range(len(data_list[0])):
        for t in range(len(data_list)):
            new_list.append(data_list[t][b])
    return new_list

def get_etities_tensor(cluster_dict, boxes, img_paths, resolution, num_regions=3):
    N, K, _, = boxes.size()
    regions = torch.Tensor(N, num_regions, 4)
    for n in range(N):
        keys_array = img_paths[n].split('/')[-3:]
        video_id, clip_id, frame_id = int(keys_array[0]), int(keys_array[1]), int(keys_array[2].split('.')[0])
        rs_labels = cluster_dict[(video_id, clip_id)][frame_id][num_regions]
        for cls in range(num_regions):
            selected_id = np.where(rs_labels==cls)[0]
            if len(selected_id)==0:
                regions[n, cls] = regions[n, cls-1]
            else:
                regions[n, cls] = torch.tensor(build_region(boxes[n][selected_id], resolution))
    return regions.cuda()


# def get_valid_boxes_tensor(boxes_tensor):
#     # delete error boxes, i.e. (0, 0, 0, 0).
#     error = torch.tensor([0, 0, 0, 0]).float().cuda()
#     output_list = []
#     for b in range(len(boxes_tensor)):
#         output = []
#         for box in boxes_tensor[b]:
#             if not torch.all(torch.eq(box, error)):
#                 output.append(box)
#         output = torch.stack(output, dim=0)
#         output_list.append(output)
#     return torch.stack(output_list, dim=0)

# def get_truth_boxes_tensor(boxes_tensor):
#     # delete error boxes, i.e. (0, 0, 0, 0).
#     error = torch.tensor([0, 0, 0, 0]).float().cuda()
#     output_list = []
#     for b in range(len(boxes_tensor)):
#         output = []
#         boxes = boxes_tensor[b]
#         for i in range(len(boxes)):
#             if not torch.all(torch.eq(boxes[i], error)):
#                 if (i-1)>0 and torch.all(torch.eq(boxes[i], boxes[i-1])):
#                     continue
#                 else:
#                     output.append(boxes[i])
#         output = torch.stack(output, dim=0)
#         output_list.append(output)
#     return torch.stack(output_list, dim=0)

def get_valid_boxes_tensor(boxes_tensor):
    # delete error boxes, i.e. (0, 0, 0, 0).
    error = torch.tensor([0, 0, 0, 0]).float().cuda()
    break_idx = boxes_tensor.size(1)
    boxes = boxes_tensor[0]
    for i in range(len(boxes)):
        if torch.all(torch.eq(boxes[i], error)):
            break_idx = i
            break
    if break_idx !=0:
        return boxes_tensor[:, :break_idx,:] ### ignore ':', we cannot determine the dim of tensor, it will be [b,t,4], [b*t,4]
    else:
        return boxes_tensor

def get_truth_boxes_tensor(boxes_tensor):
    # delete error boxes, i.e. (0, 0, 0, 0), and the filler.
#     print(boxes_tensor.size(), boxes_tensor)
    error = torch.tensor([0, 0, 0, 0]).float().cuda()
    break_idx = 0
    boxes = boxes_tensor[0]
    for i in range(len(boxes)):
        if (i-1)>0 and torch.all(torch.eq(boxes[i], boxes[i-1])):
            break_idx = i
            break
        if torch.all(torch.eq(boxes[i], error)):
            break_idx = i
            break
#     print(boxes_tensor[:, :break_idx, :].size(), boxes_tensor[:, :break_idx, :])
    return boxes_tensor[:, :break_idx, :]


def filtering(boxes_tensor):
    # delete error boxes, i.e. (0, 0, 0, 0).
    error = torch.tensor([0, 0, 0, 0]).float().cuda()
    output_list = []
    for b in range(len(boxes_tensor)):
        output = []
        for box in boxes_tensor[b]:
            if not torch.all(torch.eq(box, error)):
                output.append(box)
        output = torch.stack(output, dim=0)
        output_list.append(output)
    return torch.stack(output_list, dim=0)

def batch_formate_boxes(boxes_tensor, lower_K):
    # Formating box:(r_x1, r_y1, r_x2, r_y2) to box:(x1, y1, x2, y2) according to the size of image. 'r' denotes relative
    formated_boxes = torch.Tensor(boxes_tensor.size(0), lower_K, 4).cuda()
    for b in range(boxes_tensor.size(0)):
        boxes = boxes_tensor[b]
        formated_boxes[b, :len(boxes), :] = boxes
        formated_boxes[b, len(boxes):, :] = boxes[-1]
    return formated_boxes