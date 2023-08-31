#coding=utf-8
import os
import glob
import dlib
from collections import deque
import numpy as np
import cv2
import utils

from mmdet.apis import init_detector

class Frame_Proposal:
    """
        Generating the minimal proposal of each frame image and recording the personal bounding boxes. 
    """
    def __init__(self):
        self.top = 99999
        self.left = 99999
        self.bottom =-1
        self.right = -1
        self.person_rects = []
        
    def update_minimal_proposal(self, top, bottom, left, right):
        if top < self.top:
            self.top = top
        if bottom > self.bottom:
            self.bottom = bottom
        if left < self.left:
            self.left = left
        if right > self.right:
            self.right = right

    def update_person_rects(self):
        Rects_Array = np.asarray(self.person_rects)
        Rects_Array[:,0] = Rects_Array[:,0] - self.top
        Rects_Array[:,1] = Rects_Array[:,1] - self.top
        Rects_Array[:,2] = Rects_Array[:,2] - self.left
        Rects_Array[:,3] = Rects_Array[:,3] - self.left
        self.Rects_Array = Rects_Array
        return Rects_Array
    
    
    def resize_person_rects(self, scale):
        self.Rects_Array[:,0] = self.Rects_Array[:,0]*scale[0]
        self.Rects_Array[:,1] = self.Rects_Array[:,1]*scale[0]
        self.Rects_Array[:,2] = self.Rects_Array[:,2]*scale[1]
        self.Rects_Array[:,3] = self.Rects_Array[:,3]*scale[1]
        return self.Rects_Array
    
    def add(self, top, bottom, left, right):
        self.person_rects.append([top, bottom, left, right])
        self.update_minimal_proposal(top, bottom, left, right)
        
class Track(object):
    """This class is used for tracking the persons in video clip"""
    def __init__(self, dataset_root, data_confs, dataset_name):
        super(Track, self).__init__()
        self.dataset_root = dataset_root
        self.dataset_folder = os.path.join(dataset_root, dataset_name, 'videos')
        self.num_players = data_confs.num_players
        self.num_videos = data_confs.num_videos
        self.num_frames = data_confs.num_frames
        
        self.phases = ['trainval','test']
        self.videos = {phase: data_confs.splits[phase] for phase in self.phases}
        
        self.activity_list = data_confs.activity_list
        
        self.tracker = dlib.correlation_tracker()
        self.track_phases = ['pre', 'back']
        self.save_folder = os.path.join(dataset_root, dataset_name, 'videos')
        print ('the tracklets are saved at', self.save_folder)
        
        # build the model from a config file and a checkpoint file
        config_file = '/mnt/tangjinhui/10117_yanrui/mmdetection/configs/faster_rcnn_r50_fpn_1x.py'
        checkpoint_file = '/mnt/tangjinhui/10117_yanrui/mmdetection/weights/faster_rcnn_r50_fpn_1x_20181010-3d1b3351.pth'
        self.model = init_detector(config_file, checkpoint_file, device='cuda:0')

    def annotation_parse(self, line):
        keywords = deque(line.strip().split(' '))
        frame_id = keywords.popleft().split('.')[0]
        activity = self.activity_list.index(keywords.popleft())
        rects = []
        while keywords:
            x = int(keywords.popleft())
            y = int(keywords.popleft())
            w = int(keywords.popleft())
            h = int(keywords.popleft())
            _ = keywords.popleft() # action not used in our model
            rects.append([x,y,w,h])
        rects = np.asarray(rects)
        # sort Rects by the first col
        rects = rects[np.lexsort(rects[:,::-1].T)]
        return frame_id, rects, activity

    def track(self, person_rects, imgs, tracker, save_path=None):
        #frame_proposal_list = [Frame_Proposal() for i in range(self.num_frames)]
        #person_rects_list = [[] for i in range(self.num_frames)]
        person_rects_dict = {}
        for i, person_rect in enumerate(person_rects):
            for j, phase in enumerate(self.track_phases):
                if j == 0:
                    j = -1
                for k, f in enumerate(imgs[phase]):
                    #print("Processing Frame {}".format(k))
                    frame_img = cv2.cvtColor(cv2.imread(f), cv2.COLOR_BGR2RGB)
                    if k == 0:
                        x, y, w, h = person_rect
                        #print x,y,w,h
                        tracker.start_track(frame_img, dlib.rectangle(int(x),int(y),int(x+w),int(y+h)))
                    else:
                        tracker.update(frame_img)
                    
                    # save imgs
                    pos = tracker.get_position()
                    top, bottom, left, right = max(int(pos.top()),0),max(int(pos.bottom()),0),max(int(pos.left()),0),max(int(pos.right()),0)
                    
                    # take records for tracked person_rects
                    if j==1 and k==0:
                        pass
                    else:
                        #frame_proposal_list[4+j*k].add(top, bottom, left, right)
                        #person_rects_list[4+j*k].append([top, bottom, left, right])
                        key = '/'.join(f.split('/')[-3:])
                        if key not in person_rects_dict.keys():
                            person_rects_dict[key] = []
                        person_rects_dict[key].append([top, bottom, left, right])
                        
        return person_rects_dict
        #return self.crop_mask_frame_image(imgs, frame_proposal_list, save_path, str(activity_label), mask=False)
    
    def crop_mask_frame_image(self, imgs, frame_proposal_list, save_path, activity_label, output_size=(1280,720), crop=True, mask=True):
        """
            crop the frame image acorrding to the frame_proposal, and mask it via bounding boxes
        """
        rects_dict = {}
        for j, phase in enumerate(self.track_phases):
            if j == 0:
                j = -1
            for k, f in enumerate(imgs[phase]):
                frame_img = cv2.cvtColor(cv2.imread(f), cv2.COLOR_BGR2RGB)
                frame_proposal = frame_proposal_list[4+j*k]
                if crop:
                    cropped_frame_img = frame_img[frame_proposal.top:frame_proposal.bottom, frame_proposal.left:frame_proposal.right]
                    updated_person_rects = frame_proposal.update_person_rects()
                
                img_name = f.split('/')[-1].split('.')[0] + '_' + activity_label + '.' + f.split('/')[-1].split('.')[-1]
                img_path = os.path.join(save_path, img_name)
                
                if mask:
                    mask = utils.get_mask(cropped_frame_img.shape, updated_person_rects)
                    saved_frame_img = cropped_frame_img * mask
                else:
                    saved_frame_img = cropped_frame_img
                cv2.imwrite(img_path, cv2.resize(saved_frame_img, output_size))
                
                #print frame_img.shape, output_size, cropped_frame_img.shape
                scale=[float(output_size[1])/cropped_frame_img.shape[0], float(output_size[0])/cropped_frame_img.shape[1]]
                resized_person_rects = frame_proposal.resize_person_rects(scale)
                rects_dict['/'.join(img_path.split('/')[-3:])] = resized_person_rects

        return rects_dict
    
    
                    
    def write_list(self, source_list, block_size, phase):
        if phase is not 'test':
            source_list = utils.block_shuffle(source_list, block_size)
        txtFile = os.path.join(self.save_folder, phase + '.txt')
        open(txtFile, 'w')
        print(phase +'_size:' + str(len(source_list)/(block_size)))
        for i in range(len(source_list)):
            with open(txtFile, 'a') as f:
                f.write(source_list[i])
