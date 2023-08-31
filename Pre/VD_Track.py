#coding=utf-8
import os
import glob
import utils
from .Track import *
from mmdet.apis import inference_detector

class VD_Track(Track):
    """docstring for VD_Preprocess"""
    def __init__(self, dataset_root, data_confs, ranked=False):
        super(VD_Track, self).__init__(dataset_root, data_confs, 'VD')
        '''
        # track the persons
        if not(os.path.exists(os.path.join(self.save_folder, 'bboxes.pkl'))):
            person_rects_dict = self.getPersons()
            utils.save_pkl(person_rects_dict, os.path.join(self.save_folder, 'bboxes'))
        else:
            print "%s exist!" % (os.path.join(self.save_folder, 'bboxes.pkl'))
        '''
        # detection and track persons
        boxes_dict = self.detection_track()
        utils.save_pkl(boxes_dict, os.path.join(self.save_folder, 'tracks_normalized_detected_p1'))
        # write the train_test file
#         self.getTrainTest()


#     def getPersons(self):
#         # Create the correlation tracker - the object needs to be initialized
#         # before it can be used
#         person_rects_dict = {}
#         for video_id in range(self.num_videos):
#             self.joints_dict = {}
#             video_id = str(video_id)
#             annotation_file = os.path.join(self.dataset_folder, video_id, 'annotations.txt')
#             f = open(annotation_file)
#             lines = f.readlines()
#             imgs={}
#             for line in lines:
#                 frame_id, rects, _ = self.annotation_parse(line)
#                 img_list = glob.glob(os.path.join(self.dataset_folder, video_id, frame_id, "*.jpg"))
#                 img_list = sorted(img_list, key=lambda x:int(x.split('/')[-1].split('.')[0]))[20-int((self.num_frames-1)/2):21+int(np.ceil(float(self.num_frames-1)/2.0))]
#                 imgs['pre'] = img_list[:5][::-1] # reverse
#                 imgs['back'] = img_list[4:]
                
#                 if len(rects)<=self.num_players:
#                     person_rects_dict.update(self.track(rects, imgs, self.tracker))
#                     '''
#                     #print 'video_id: ', video_id, 'frame_id: ', frame_id
#                     save_path = os.path.join(self.save_folder, video_id, frame_id)
                    
#                     if not(os.path.exists(save_path)):
#                         os.makedirs(save_path)
#                         # We will track the frames as we load them off of disk
#                         person_rects_dict.update(self.track(rects, imgs, self.tracker, save_path))
#                     else:
#                         print "%s exist!" % (save_path)
#                     '''
#         return person_rects_dict
    
    def detection_track(self):
        boxes_dict = {}
        for video_id in range(0, 10):
            # traverse the videos
            imgs = {}
            for root, dirs, files in os.walk(os.path.join(self.dataset_folder, str(video_id))):
                if len(dirs) != 0:
                    for dir in dirs:
                        clip_id = int(dir)
                        middle_frame = os.path.join(root, dir, dir + '.jpg')
                        # detect persons for the middle frame
                        boxes = inference_detector(self.model, middle_frame)
#                         boxes = boxes[0][:12,:-1]
#                         selected_id = np.where(boxes[0][:,-1]>0.7)[0]
#                         boxes = boxes[0][selected_id,:-1]
        
                        boxes = boxes[0]
                        
                        # track
                        img_list = glob.glob(os.path.join(root, dir, "*.jpg"))
                        img_list = sorted(img_list, key=lambda x: int(x.split('/')[-1].split('.')[0]))
                        middle_point = int(len(img_list) / 2)  # 20
                        imgs['pre'] = img_list[:middle_point + 1][::-1]  # reverse
                        imgs['back'] = img_list[middle_point:]
                        boxes_dict[(video_id, clip_id)] = self.track(boxes, imgs)
                        print('%d/%d'%(video_id, clip_id))
        return boxes_dict

    def track(self, boxes, imgs):
        frame_boxes_dict = {}
        for i, box in enumerate(boxes):
            for phase in self.track_phases:
                for k, f in enumerate(imgs[phase]):
                    #print("Processing Frame {}".format(k))
                    frame_img = cv2.cvtColor(cv2.imread(f), cv2.COLOR_BGR2RGB)
                    H, W = frame_img.shape[:2]
                    if k == 0:
                        x1, y1, x2, y2, prob = box # left, top, right, bottom
                        self.tracker.start_track(frame_img, dlib.rectangle(int(x1),int(y1),int(x2),int(y2)))
                    else:
                        self.tracker.update(frame_img)
                        
                    pos = self.tracker.get_position()
                    top, bottom, left, right = max(int(pos.top()),0),max(int(pos.bottom()),0),max(int(pos.left()),0),max(int(pos.right()),0)
                    r_x1, r_y1, r_x2, r_y2 = float(left)/W, float(top)/H, float(right)/W, float(bottom)/H
                        
                    # take records for tracked boxes
                    if phase=='back' and k==0:
                        pass
                    else:
                        keys_array = f.split('/')[-3:]
                        video_id, clip_id, frame_id = int(keys_array[0]), int(keys_array[1]), int(keys_array[2].split('.')[0])
                        if frame_id not in frame_boxes_dict.keys():
                            frame_boxes_dict[frame_id] = []
                        frame_boxes_dict[frame_id].append([r_x1, r_y1, r_x2, r_y2, prob])
        return frame_boxes_dict

    def getTrainTest(self):
        # split train-test following [CVPR 16]
        for phase in self.phases:
            print(phase + ' videos:', self.videos[phase])
            activity_list = []
            for video_id in self.videos[phase]:
                imgs_folder = os.path.join(self.save_folder, str(video_id))
                annotation_file = os.path.join(imgs_folder, 'annotations.txt')
                f = open(annotation_file)
                lines = f.readlines()
                for line in lines:
                    frame_id, _, activity = self.annotation_parse(line)
                    if activity>=2:
                        activity-=2 # merge l_pass and l_set, r_pass and r_set
                    img_list = glob.glob(os.path.join(imgs_folder, frame_id, "*.jpg"))
                    img_list = sorted(img_list, key=lambda x:int(x.split('/')[-1].split('.')[0]))[20-int((self.num_frames-1)/2):21+int(np.ceil(float(self.num_frames-1)/2.0))]
                    for img in img_list:
                        activity_list.append(img + '\t' + str(activity) + '\n')
                        #print img + '\t' + str(activity) + '\n'
                '''
                for root, dirs, files in os.walk(imgs_folder):
                    activity_label = ''
                    print
                    if len(files)!=0:
                        files.sort()
                        for i in xrange(self.num_frames):
                            if i<len(files):
                                # parse
                                filename = files[i]
                                #activity_label = filename.split('_')[1].split('.')[0]
                                activity_label = activity_dict['/'.join((video_id, frame_id))]
                                file_path = os.path.join(root, filename)
                                activity_list.append(file_path + '\t' + activity_label + '\n')
                            else:
                                print 'error!!!'
                                exit(0)
                '''

            self.write_list(activity_list, self.num_frames, phase)
