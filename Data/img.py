"""
 Load data from img source.
"""
import torch
from torch.utils.data import Dataset
from skimage import io, transform
import numpy as np
from PIL import Image
import os
import pickle
import random
import time
import argparse
# import jpeg4py as jpeg

class img(Dataset):
    def __init__(self, data_confs, phase, data_transform=None):
        self.full_input = True
        self.dataset_folder = data_confs.dataset_folder
        self.IH, self.IW = data_confs.resolution
        self.num_players = data_confs.num_players
        self.num_frames = data_confs.num_frames
        self.N_f = data_confs.N_f
        self.phase = phase
        self.data_transform = data_transform
        self.txt_file = os.path.join(self.dataset_folder, phase + '.txt')
        
        parser = argparse.ArgumentParser()
        parser.add_argument('--tracks_mode', type=str)
        parser.add_argument('--mode_arg', type=float)
        opt, unknown = parser.parse_known_args()
        self.tracks_mode = opt.tracks_mode
        self.mode_arg = opt.mode_arg
        
        
        self.boxes_dict = pickle.load(open(os.path.join(self.dataset_folder, 'tracks_normalized_detected_all.pkl'), 'rb'))

        lines = open(self.txt_file)
        self.path_list = []
        self.labels_list = []
        for i, line in enumerate(lines):
            img_path = line.split('\n')[0].split('\t')[0]
            label = line.split('\n')[0].split('\t')[1]
            if ('error' in label) or ('NA' in label):
                pass
            else:
                self.path_list.append(img_path)
                self.labels_list.append(int(label))
    
    
    def __getitem__(self, index):
        '''
            load a random sequence of frames
        '''
        # sample the paths of frames and get activity label for this sequence
        #sample_number = self.N_f if self.phase=='trainval' else self.num_frames
        sample_number = self.N_f
        img_paths, targets = self.random_sample(index, sample_number)

        # read images and boxes
        imgs, boxes = self.load_sequence(img_paths)
        return imgs, boxes, targets, img_paths
        
    def random_sample(self, index, sample_number):
        img_paths = self.path_list[index*self.num_frames:(index+1)*self.num_frames]
        targets = torch.tensor(self.labels_list[index*self.num_frames:(index+1)*self.num_frames])
        sub_img_paths = []
        step = len(img_paths)//sample_number
        for seg_id in range(sample_number):
            seg_paths = img_paths[seg_id*step:(seg_id+1)*step]
            selected_path = random.choice(seg_paths) if self.phase=='trainval' else seg_paths[0]
            sub_img_paths.append(selected_path)
        #sub_img_paths = random.sample(img_paths, sample_number) if sample_number<self.num_frames else img_paths
        sub_targets = targets[:sample_number]
        return sub_img_paths, sub_targets
    
    def load_sequence(self, img_paths):
        """load a sequence of imgs
        Returns:
            pytorch tensors
        """
        img_list, boxes_list = [], []
        for img_path in img_paths:
            img = Image.open(os.path.join(self.dataset_folder,img_path))
#             img = Image.open(img_path)
            
            keys_array = img_path.split('/')[-3:]
            video_id, clip_id, frame_id = int(keys_array[0]), int(keys_array[1]), int(keys_array[2].split('.')[0])
            if frame_id not in self.boxes_dict[(video_id, clip_id)].keys():
                print(video_id, clip_id, frame_id)
            
            ## load boxes by different modes, such as prob, pre, all.
            if self.tracks_mode == 'prob':
                boxes = torch.tensor(self.prob(self.boxes_dict[(video_id, clip_id)][frame_id], self.mode_arg)).float()
            elif self.tracks_mode == 'pre':
                boxes = torch.tensor(self.pre(self.boxes_dict[(video_id, clip_id)][frame_id], self.mode_arg)).float()
            else:
                boxes = torch.tensor(self.boxes_dict[(video_id, clip_id)][frame_id]).float()

            if self.full_input:
                # take a full image as input
                img, boxes = self.get_full_input(img, boxes)
                img_list.append(img)
            else:
                # take a set of person images as inputs
                imgs, boxes = self.get_person_input(img, boxes)
                img_list.append(imgs)
            boxes_list.append(boxes)
        return torch.stack(img_list), torch.stack(boxes_list)
    
    def __len__(self):
        return len(self.labels_list)//self.num_frames
    
    def get_person_input(self, img, boxes):
        """
            imgs: a full image, boxes: a set of personal bounding boxes
            Returns: a list of person images and boxes
        """
        person_imgs = []
        boxes = self.normlize_boxes(boxes, self.num_players)
        for box in boxes:
            person_img = img.crop(box)
            if self.data_transform is not None:
                person_img = self.data_transform(person_img)
            person_imgs.append(person_img)
        return torch.stack(person_imgs), boxes
    
    def get_full_input(self, img, boxes):
        # input full image
        if self.data_transform is not None:
            img = self.data_transform(img)
        boxes = self.adaptive_formate_boxes(boxes, lower_K=self.num_players, img_size=img.size()[-2:])
        return img, boxes
    
    
    def formate_boxes(self, boxes, K, img_size=[720, 1280]):
        # Formating box:(r_x1, r_y1, r_x2, r_y2) to box:(x1, y1, x2, y2) according to the size of image. 'r' denotes relative
        formated_boxes = torch.Tensor(K, 4)
        H, W = img_size
        for idx in range(K):
            r_x1, r_y1, r_x2, r_y2 = boxes[idx] if idx<len(boxes) else boxes[-1]
            formated_boxes[idx, :] = torch.tensor([r_x1*W, r_y1*H, r_x2*W, r_y2*H])
        return formated_boxes

#     def adaptive_formate_boxes(self, boxes, lower_K, upper_K=100, img_size=[720, 1280]):
#         # Formating box:(r_x1, r_y1, r_x2, r_y2) to box:(x1, y1, x2, y2) according to the size of image. 'r' denotes relative
#         formated_boxes = torch.Tensor(upper_K, 4)
#         H, W = img_size
#         for idx in range(upper_K):
#             if idx<len(boxes):
#                 r_x1, r_y1, r_x2, r_y2 = boxes[idx][:4]
#             elif idx<lower_K:
#                 r_x1, r_y1, r_x2, r_y2 = boxes[-1][:4]
#             else:
#                 r_x1, r_y1, r_x2, r_y2 = torch.tensor([0, 0, 0, 0])
#             formated_boxes[idx, :] = torch.tensor([r_x1*W, r_y1*H, r_x2*W, r_y2*H])
#         return formated_boxes

    def adaptive_formate_boxes(self, boxes, lower_K, upper_K=110, img_size=[720, 1280]):
        # Formating box:(r_x1, r_y1, r_x2, r_y2) to box:(x1, y1, x2, y2) according to the size of image. 'r' denotes relative
        error = torch.tensor([0, 0, 0, 0])
        latest_box = torch.tensor([0.1, 0.1, 0.1, 0.1])
        formated_boxes = torch.Tensor(upper_K, 4)
        H, W = img_size
        for idx in range(upper_K):
            if idx<len(boxes):
                if not torch.all(torch.eq(boxes[idx][:4], error)):
                    # check the boxes for error data (i.e., Missing track).
                    r_x1, r_y1, r_x2, r_y2 = boxes[idx][:4]
                    latest_box = boxes[idx][:4]
                else:
                    # fill right boxes into truth boxes.
                    r_x1, r_y1, r_x2, r_y2 = latest_box + (torch.rand((4))/100) # tmp_boxes denote the latest right data in list, we add a small random vector on it to represent the new box.
                if idx==(len(boxes)-1):
                    last_box = torch.tensor([r_x1, r_y1, r_x2, r_y2])
            elif idx<lower_K:
#                 print('idx_test:', idx, len(boxes))
                r_x1, r_y1, r_x2, r_y2 = last_box # records_boxes denote the valid data
            else:
                r_x1, r_y1, r_x2, r_y2 = torch.tensor([0, 0, 0, 0])
            formated_boxes[idx, :] = torch.tensor([r_x1*W, r_y1*H, r_x2*W, r_y2*H])
        return formated_boxes
    
    def pre(self, boxes, pre_N=12):
#         print('pre_N: ', pre_N)
        boxes = sorted(boxes, key=lambda x:(-x[-1]))
        return boxes[:int(pre_N)]

    def prob(self, boxes, prob_threshold=0.7):
#         print('prob_threshold: ', prob_threshold)
        boxes = np.array(boxes)
        selected_id = np.where(boxes[:,-1]>prob_threshold)[0]
        ## the number of proposals whose prob is larger than threshold, will be zero!!!
        if len(selected_id)==0:
            return boxes[[0]]
#         print('selected_id:', len(selected_id))
        return boxes[selected_id]