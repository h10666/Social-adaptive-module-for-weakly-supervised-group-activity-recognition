# -*- coding:utf-8 -*-
import torch
from torch.autograd import Variable
import numpy as np
import time
import os
import gc
import sys
from torch.optim import lr_scheduler
import utils
import torch.nn.functional as F
from sklearn import datasets, svm, metrics
import argparse
from torch.utils.tensorboard import SummaryWriter


class Solver:
    def __init__(self, net, data_confs, solver_confs):
        # data args
        self.data_loaders = solver_confs.data_loaders
        self.data_sizes = solver_confs.data_sizes
        self.num_frames = data_confs.num_frames
#         self.num_selected_frames = data_confs.num_selected_frames
        #self.K = data_confs.num_players
        # net args
        self.net = net
        # model training args
        self.gpu = solver_confs.gpu
        self.num_epochs = solver_confs.num_epochs
        self.optimizer = solver_confs.optimizer
        self.criterion = solver_confs.criterion
        self.scheduler = solver_confs.exp_lr_scheduler
        
        
        parser = argparse.ArgumentParser()
        parser.add_argument('--save_path', type=str)
        parser.add_argument('--person', type=utils.str2bool, nargs='?', const=True, default=False)
        parser.add_argument('--frame', type=utils.str2bool, nargs='?', const=True, default=False)
        opt, unknown = parser.parse_known_args()
        self.save_path = opt.save_path
        self.person = opt.person
        self.frame = opt.frame
#         self.save_path = os.path.join('./weights', solver_confs.dataset_name)

    def training(self, inputs, labels, phase):
        # Zero the parameter gradients
        self.optimizer.zero_grad()
        # Forward
        _, outputs = self.net(inputs)

        if phase == 'test':
            _, preds = torch.max(torch.squeeze(torch.mean(outputs.data, 1), 1), -1)
        else:
            _, preds = torch.max(outputs, -1)
            preds = preds.view(-1)
            
        outputs = outputs.view(-1, outputs.size(-1)) # b*t, c
        targets = labels[::len(labels)//len(outputs)]
#         print('outputs:', outputs.size(), 'labels', labels.size(), 'preds', preds.size())
        loss = self.criterion(outputs, targets)

        # Backward + optimize(update parameters) only if in training phase
        if phase == 'trainval':
            loss.backward()
            self.optimizer.step()
        
        # statistics
        self.running_loss += loss.item()
        self.mini_batch_running_loss += loss.item()
        targets = labels[::len(labels)//len(preds)]
        self.running_corrects += torch.sum(preds == targets)
        return preds, targets
    
    def two_training(self, inputs, labels, phase):
        lamda_f, lamda_p = (0.5, 1)
        # Zero the parameter gradients
        self.optimizer.zero_grad()
        # Forward
        frame_scores, person_scores = self.net(inputs)
#         print('frame_scores:', frame_scores.size(), 'person_scores:', person_scores.size())
        
        scores = lamda_f*frame_scores + lamda_p*person_scores
        preds = torch.argmax(scores, -1)
#         preds = torch.argmax(person_scores, -1)
        
#         frame_preds = torch.argmax(frame_scores, -1)
#         person_preds = torch.argmax(person_scores, -1)
        
#         print('f preds:', frame_preds, 'p preds:', person_preds)
#         preds =  lamda_p*person_preds
        targets = labels[::len(labels)//len(person_scores)]
#         print('frame_scores:', frame_scores.size(), 'person_scores', person_scores.size(), 'targets', targets.size())

        loss = self.criterion(scores, targets)
#         frame_loss = self.criterion(frame_scores, targets)
#         person_loss = self.criterion(person_scores, targets)
#         loss = lamda_f*frame_loss + lamda_p*person_loss

        # Backward + optimize(update parameters) only if in training phase
        if phase == 'trainval':
            loss.backward()
            self.optimizer.step()
        
        # statistics
        self.running_loss += loss.item()
        self.mini_batch_running_loss += loss.item()
        targets = labels[::len(labels)//len(preds)]
        self.running_corrects += torch.sum(preds == targets)
        return preds, targets
    
    def single_training(self, inputs, labels, phase):
        # Zero the parameter gradients
        self.optimizer.zero_grad()
        # Forward
        scores = self.net(inputs)
#         preds = torch.argmax(torch.squeeze(torch.mean(scores.data, 1), 1), -1)
#         scores = scores.view(-1, scores.size(-1)) # b*t, c
        preds = torch.argmax(scores, -1)
        
        targets = labels[::len(labels)//len(scores)]
        loss = self.criterion(scores, targets)

        # Backward + optimize(update parameters) only if in training phase
        if phase == 'trainval':
            loss.backward()
            self.optimizer.step()
        
        # statistics
        self.running_loss += loss.item()
        self.mini_batch_running_loss += loss.item()
        targets = labels[::len(labels)//len(preds)]
        self.running_corrects += torch.sum(preds == targets)
        return preds, targets

    def train_model(self, phases=['trainval', 'test']):
        if not(os.path.exists(self.save_path)):
            os.makedirs(self.save_path)
        self.writer = SummaryWriter(self.save_path)
            
        best_acc = 0.0
        best_avg_acc = 0.0
        for epoch in range(self.num_epochs):
            print('Epoch {}/{}'.format(epoch, self.num_epochs - 1))
            print('-' * 10)
            # Each epoch has a training and evaluate phase
            for phase in phases:
                since = time.time()
                preds = []
                targets = []
                if phase == 'trainval':
                    self.scheduler.step()
                    self.net.train()  # Set model to training mode
#                     self.net.apply(utils.set_bn_eval)
                else:
                    self.net.eval()  # Set model to evaluate mode
                
                self.running_loss = 0.0
                self.running_corrects = 0.0
                self.mini_batch_running_loss = 0.0

                # Iterate over data.
                for i, data in enumerate(self.data_loaders[phase]):
                    # get the inputs                    
                    if len(data)>=2:
                        inputs = (data[0].float().cuda(), data[1].float().cuda(), data[-1]) if self.gpu else (data[0].float(), data[1].float())
                    else:
                        inputs = data[0].float().cuda() if self.gpu else data[0].float()
                    labels = data[-2].view(-1).cuda() if self.gpu else data[-2]

#                     self.training(inputs, labels, phase)
#                     since = time.time()
                    if self.frame and self.person:
                        pred, target = self.two_training(inputs, labels, phase)
                    else:
                        pred, target = self.single_training(inputs, labels, phase)
#                     print('training, takes', time.time()-since)

                    pred, target = pred.view(-1), target.view(-1)
                    preds.extend(pred.cpu().numpy())
                    targets.extend(target.cpu().numpy())
                    
                    ## every 100 mini-batch printer
                    # ...log the running loss
                    if phase == 'trainval' and i % 1000 == 999:
                        self.writer.add_scalar('training loss',
                                self.mini_batch_running_loss / 1000,
                                epoch * len(self.data_loaders['trainval']) + i)
                        self.mini_batch_running_loss = 0.0
                
#                 if phase == 'trainval':
#                     self.scheduler.step()
                
#                 num_frames = self.num_selected_frames if phase=='trainval' else self.num_frames
#                 num_frames = self.num_selected_frames if phase=='trainval' else 1
                num_frames = 1
                print(self.running_corrects, self.data_sizes[phase])
                epoch_loss = float(self.running_loss) / (self.data_sizes[phase]*num_frames)
                epoch_acc = float(self.running_corrects) / (self.data_sizes[phase]*num_frames)
                
                ##################
                # added by Mr. Yan
                preds_array = np.asarray(preds, dtype=int).reshape(-1)
                targets_array = np.asarray(targets, dtype=int).reshape(-1)
                epoch_each_acc, epoch_avg_acc = utils.get_avg_acc(preds_array, targets_array)

                # display related Info(Loss, Acc, Time, etc.)
                
                ## every epoch printer
                print('Epoch: {} phase: {} Loss: {} Acc: {} Avg_acc: {}'.format(
                    epoch, phase, epoch_loss, epoch_acc, epoch_avg_acc))
                if phase == 'test':
                    print('Each acc:', epoch_each_acc)
                time_elapsed = time.time() - since
                print('Running this epoch in {:.0f}m {:.0f}s'.format(
                    time_elapsed // 60, time_elapsed % 60))

                if phase == 'test' and epoch_acc > best_acc:
                    best_acc = epoch_acc
                    torch.save(self.net.state_dict(), os.path.join(self.save_path, 'best_acc_wts.pth'))
                if phase == 'test' and epoch_avg_acc > best_avg_acc:
                    best_avg_acc = epoch_avg_acc
                    torch.save(self.net.state_dict(), os.path.join(self.save_path, 'best_avg_acc_wts.pth'))
        print('Best test Acc: {:4f}'.format(best_acc))
        print('Best test Avg_acc: {:4f}'.format(best_avg_acc))

        
    def evaluate(self):
        preds = []
        targets = []
        spatial_dense_relation_dict= {}
        temporal_dense_relation_dict= {}
        with torch.no_grad():
            for i, data in enumerate(self.data_loaders['test']):
                inputs = (data[0].float().cuda(), data[1].float().cuda(), data[2])
                target = data[-2].view(-1)[0].cuda()
                img_paths = data[-1]
                # compute output
                spatial_dense_relation, temporal_dense_relation, outputs = self.net(inputs)
                probs = F.softmax(outputs.data)
                # print probs
                _, pred = torch.max(torch.mean(outputs.data, 0).view(1, -1), 1)
                preds.append(pred.cpu().numpy())
                targets.append(target)
                
                # get spatial relation
                spatial_dense_relation = spatial_dense_relation.cpu().numpy()
                temporal_dense_relation = temporal_dense_relation.cpu().numpy()
                for i, img_path in enumerate(img_paths):
                    video_id, clip_id, frame_id = (map(int, img_path[0].split('.')[0].split('/')))
                    if (video_id, clip_id) not in spatial_dense_relation_dict.keys():
                        spatial_dense_relation_dict[(video_id, clip_id)] = {}
#                     print(video_id, clip_id, frame_id, dense_relation[i].shape)
                    spatial_dense_relation_dict[(video_id, clip_id)][frame_id] = spatial_dense_relation[i]
                
                # get temporal relation
                temporal_dense_relation_dict[(video_id, clip_id)] = temporal_dense_relation

        ### show result
        preds = np.asarray(preds, dtype=int)
        #preds = label_map(preds)
        targets = np.asarray(targets, dtype=int)
        #labels = label_map(labels)
        preds, targets = preds.reshape(-1,1), targets.reshape(-1,1)
        
        utils.save_pkl(spatial_dense_relation_dict, 'spatial_dense_relation_dict')
        utils.save_pkl(temporal_dense_relation_dict, 'temporal_dense_relation_dict')
        
        print("Classification report for classifier \n %s" % (metrics.classification_report(targets, preds)))
        print("Confusion matrix:\n%s" % utils.normlize(metrics.confusion_matrix(targets, preds)))
        print(np.sum(preds == targets) / float(targets.shape[0]))

        # Compute confusion matrix
        cnf_matrix = metrics.confusion_matrix(targets, preds)
        print(cnf_matrix)
