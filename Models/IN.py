import math
import os
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
from torch.autograd import Variable
from .backbones.lib.Panoramic_Reasoning import PanoramicReasoningBlock2D, PanoramicReasoningBlock1D
# from .backbones.lib.non_local_dot_product import NONLocalBlock2D, NONLocalBlock1D
from .backbones.lib.non_local_embedded_gaussian import NONLocalBlock2D, NONLocalBlock1D
import utils
import Models
from .temporal_shift import TemporalShift
from .TRNmodule import RelationModule
import time
import argparse

__all__ = ['IN', 'iN']

fea_map_dim = {
        'myInception_v3': 768,
        'inception_v3': 2048,
        'inception_I3D': 1024,
        'resNet18_I3D': 2048,##?????
        'resNet18_I3D_NLN': 2048,##?????
        'resNet18': 512,
        'alexNet': 256,
}

class iN(nn.Module):
    def __init__(self, dataset_name, data_confs, model_confs):
        super(iN, self).__init__()
        self.num_players = data_confs.num_players
        self.num_classes = data_confs.num_classes
#         self.N_f = data_confs.N_f # number of input frames
        self.num_frames = data_confs.num_frames
        self.num_groups = data_confs.num_groups
        self.resolution = data_confs.resolution
        self.T = data_confs.N_f # number of input frames
        self.IH, self.IW = data_confs.resolution
        self.data_confs = data_confs
        #############################################################
        # model structure parameter
        self.model_confs = model_confs
        self.fea_map_dim = fea_map_dim[self.model_confs.backbone_name]
        self.fea_map_H, self.fea_map_W = 6, 10
        self.entity_feas_size = 1024
#         self.frame_feas_size = 4096
        self.output_size = 1024 #1000 for alexnet; 512 for resnet
#         self.output_crop_size = [3, 3] # the size of the output after the cropping is performed, as (height, width), used for RoIAlign

        parser = argparse.ArgumentParser()
#         parser.add_argument('--entity_choices', nargs='+', type=float, default=1)
#         parser.add_argument('--fusion_weights', nargs='+', type=float, default=1)
        parser.add_argument('--person', type=utils.str2bool, nargs='?', const=True, default=False)
        parser.add_argument("--person_SAM", type=utils.str2bool, nargs='?', const=True, default=False)
        parser.add_argument('--mode_arg', type=float) # N_p
        parser.add_argument("--temporal_SAM", type=utils.str2bool, nargs='?', const=True, default=False)
        parser.add_argument("--L", type=int)
        
        parser.add_argument('--frame', type=utils.str2bool, nargs='?', const=True, default=False)
        parser.add_argument("--frame_SAM", type=utils.str2bool, nargs='?', const=True, default=False)
        parser.add_argument("--frame_TRN", type=utils.str2bool, nargs='?', const=True, default=False)
        parser.add_argument('--N_f', type=int) #
        parser.add_argument('--K_f', type=int) #
        
        
        opt, unknown = parser.parse_known_args()
#         self.entity_choices = opt.entity_choices
#         self.fusion_weights = opt.fusion_weights
        self.person = opt.person
        self.person_SAM = opt.person_SAM
        self.temporal_SAM = opt.temporal_SAM
        self.L = opt.L

        self.frame = opt.frame
        self.frame_SAM = opt.frame_SAM
        self.frame_TRN = opt.frame_TRN

#         self.entity_choices = [0.8] # [0, 0.5, 0.8, 1], '0' denotes that treating the whole frame as a entity; '1' denotes that treating each people as entities; '0.5' or '0.8' denote that clustering people into N*0.5 or N*0.8 groups as the input entities.
#         self.fusion_weights = [1] # weights
        
        self.crop_size_dict = {0:[1,1], #[6,10],
                               0.2:[7,7], 0.3:[7,7], 0.4:[6,6], 0.5:[6,6], 
                               0.6:[6,6], 0.7:[5,5], 0.8:[5,5], 1:[5,5]}
        self.entity_feas_size_dict = {0:self.fea_map_dim, #1024, 
                                      0.2:1024, 0.3:1024, 0.4:1024, 0.5:1024, 
                                      0.6:1024, 0.7:1024, 0.8:1024, 1:1024}
        
#         self.crop_size_dict = {1:[6,10], 2:[13,13], 3:[9,9], 4:[7,7], 6:[5,5], 12:[5,5]}
        #############################################################
#         print('entity_choices: ', self.entity_choices, ' fusion_weights: ',  self.fusion_weights)
#         print('person_SAM: ', self.person_SAM, 'frame_SAM: ', self.frame_SAM)

        
#         # load cluster dictionary
#         cluster_file = 'dataset/'+dataset_name+'/videos/small_cluster_prob_09.pkl'
#         if os.path.exists(cluster_file):
#             self.cluster_dict = pickle.load(open(cluster_file, 'rb'))
            
        # cnn backbone for feature representation
        if self.model_confs.backbone_name=='inceptionI3d':
            self.CNN_backbone = Models.inceptionI3d(in_channels=3)
            self.CNN_backbone.load_state_dict(torch.load('weights/rgb_imagenet.pt'))
        else:
            self.CNN_backbone = eval('Models.'+self.model_confs.backbone_name)(pretrained=True, dataset_name=dataset_name, data_confs=data_confs, model_confs=model_confs)

        if self.frame:
            if self.frame_SAM:
                self.T = opt.K_f
                print('frame_SAM(N_f:{}-->K_f:{})'.format(opt.N_f, opt.K_f))
                self.frame_Dense_Relation = NONLocalBlock1D(self.fea_map_dim, sub_sample=False, bn_layer=True)
                self.fc_fusion = nn.Sequential(
                        nn.Linear(self.T * self.fea_map_dim, self.fea_map_dim),
                        nn.ReLU(inplace=True),
                        nn.Dropout(0.5)
                    )
#                 self.fused_classifier = nn.Linear(self.fea_map_dim+1024, self.num_classes)
            else:
                print('frame_only(N_f:{})'.format(opt.N_f))
            if self.frame_TRN:
                self.frame_classifier = RelationModule(self.fea_map_dim, opt.N_f, self.num_classes)
            else:
                self.frame_classifier = nn.Linear(self.fea_map_dim, self.num_classes)
        
        if self.person:
            N_p = opt.mode_arg # prob or pre
            if self.person_SAM:
                K_p = self.num_players
                print('person_SAM(N_p:{}-->K_p:{})'.format(N_p, K_p))
                self.person_Dense_Relation = NONLocalBlock1D(self.entity_feas_size, sub_sample=False, bn_layer=True)
                if self.temporal_SAM:
                    print('person_temporal_SAM(T:{}-->L:{})'.format(self.T, self.L))
                    self.temporal_Dense_Relation = NONLocalBlock1D(self.entity_feas_size, sub_sample=False, bn_layer=True)
                    self.temporal_fusion = nn.Sequential(
                        nn.Linear(self.L * self.entity_feas_size, self.entity_feas_size),
                        nn.ReLU(inplace=True),
                        nn.Dropout(0.5)
                    )
            else:
                print('person_only(N_p:{})'.format(N_p))
            self.person_embedding = nn.Sequential(
                    nn.Linear(self.fea_map_dim * 5 * 5, 1024),
                    nn.ReLU(inplace=True),
                    nn.Dropout(0.3)
                )
            self.person_classifier = nn.Linear(1024, self.num_classes)
            
            
#         if self.entity_choices:
#             self.entity_embedding_list = nn.ModuleList([
#                 nn.Sequential(
#                     nn.Linear(self.fea_map_dim * self.crop_size_dict[num_entities][0] * self.crop_size_dict[num_entities][1], self.entity_feas_size_dict[num_entities]),
#                     nn.ReLU(inplace=True),
#                     nn.Dropout(0.3)
#                 )
#                     for i, num_entities in enumerate(self.entity_choices)])

                
        # classify
#         self.classifier = nn.Linear(self.output_size*self.model_confs.num_groups, self.num_classes)
#         if self.entity_choices:
#             self.classifier_list = nn.ModuleList([nn.Linear(self.entity_feas_size_dict[num_entities]*self.num_groups, self.num_classes) for i, num_entities in enumerate(self.entity_choices)])
        
        # load backbone and fix it
#         if not self.model_confs.train_backbone:
#         if self.temporal_SAM:
#             for p in self.CNN_backbone.parameters():
#                 p.requires_grad=False
#             for p in self.person_Dense_Relation.parameters():
#                 p.requires_grad=False
#             for p in self.person_embedding.parameters():
#                 p.requires_grad=False
#             self.load_backbone(self.model_confs.backbone_path)

        # init for params
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                nn.init.zeros_(m.bias)

    def load_backbone(self, filepath):
        state = torch.load(filepath)
        CNN_backbone = {'.'.join(k.split('.')[2:]): v for k, v in state.items() if k.split('.')[1] in ['CNN_backbone']}
        person_Dense_Relation = {'.'.join(k.split('.')[2:]): v for k, v in state.items() if k.split('.')[1] in ['person_Dense_Relation']}
        person_embedding = {'.'.join(k.split('.')[2:]): v for k, v in state.items() if k.split('.')[1] in ['person_embedding']}
#         box_feas_embedding_state = {'.'.join(k.split('.')[2:]): v for k, v in state.items() if k.split('.')[1]=='box_feas_embedding'}
        self.CNN_backbone.load_state_dict(CNN_backbone)
        self.person_Dense_Relation.load_state_dict(person_Dense_Relation)
        self.person_embedding.load_state_dict(person_embedding)
#         self.box_feas_embedding.load_state_dict(box_feas_embedding_state)
        print('Load model states from: ', filepath)
    
    def forward(self, inputs):
        #self.T = self.num_selected_frames if self.training else self.num_frames
        imgs, boxes_tensor, img_paths = inputs
        boxes_tensor = boxes_tensor.view(-1, *boxes_tensor.size()[2:]) # B*T, K, 4
        
#         print('input imgs size: ', imgs.size()) # B, T, C, H, W
        input_T =imgs.size(1)
        if 'I3D' in self.model_confs.backbone_name:
            imgs = torch.transpose(imgs, 1, 2) # B, C, T, H, W
            self.T = 1
        else:
            imgs = imgs.view(-1, *imgs.size()[-3:]) # B*T, C, H, W
        
        # extract CNN feature map
#         since = time.time()
        feature_maps = self.CNN_backbone(imgs)
#         print('feature_maps size: ', feature_maps.size())
#         print('extract CNN feature map, takes', time.time()-since)
        
#         since = time.time()
        ##################################################################################################
        if self.frame:
            # build frame-level feature
            if self.frame_SAM:
                # Extracting feas for key frames.
                sparse_feature_maps, sparse_frame_feas, sparse_boxes_tensor = self.build_sparse_frame_maps_feas(feature_maps, boxes_tensor, input_T, self.T)
                # sparse_frame_feas (B, L, C)
                sparse_frame_feas = sparse_frame_feas.view(sparse_frame_feas.size(0), -1) #(B, T_*C)
                sparse_frame_feas = self.fc_fusion(sparse_frame_feas) # (B, C)
                feature_maps, boxes_tensor = sparse_feature_maps, sparse_boxes_tensor
                frame_scores = self.frame_classifier(sparse_frame_feas)#.unsqueeze(1) #(B, cls)
            else:
                # Extracting feas for all frames directly.
                frame_feature_maps = F.adaptive_avg_pool2d(feature_maps, (1, 1))
                frame_feas = torch.flatten(frame_feature_maps, 1) # b*t, c*h*w
                frame_feas = frame_feas.view(int(frame_feas.size(0)/self.T), self.T, -1) # B，T, c*h*w
                frame_scores = self.frame_classifier(frame_feas) #(B, T, cls)
                if not self.frame_TRN:
#                     print('frame_scores size:', frame_scores.size())
                    frame_scores = torch.squeeze(torch.mean(frame_scores, 1), 1)
            
        ##################################################################################################
#         print('frame_SAM, takes', time.time()-since)

        if self.person:
            # build person-level feature
            if self.person_SAM:
                spatial_dense_relation, person_feas = self.batch_build_sparse_entity_feas(feature_maps, boxes_tensor, img_paths, [5,5], self.resolution) # B*T, K, D'
#                 frame_feas = self.build_pairwise_frame_feas(entity_feas, i)
                frame_feas = utils.build_frame_feas(person_feas)#(B, T, C)
                if self.temporal_SAM:
                    # build dense relation
                    frame_feas = torch.transpose(frame_feas, -2, -1) #(B, C, T)
                    frame_feas, temporal_dense_relation = self.temporal_Dense_Relation(frame_feas)
                    # pruning T to L
                    frame_feas = torch.transpose(frame_feas, -2, -1) # (B, T, C)
                    sparse_frame_feas = self.Pruning(frame_feas, temporal_dense_relation, self.L) # (B, L, C)
                    # temporal fusion
                    sparse_frame_feas = sparse_frame_feas.view(sparse_frame_feas.size(0), -1) #(B, L*C)
#                     print(sparse_frame_feas.size())
                    frame_feas = self.temporal_fusion(sparse_frame_feas) # (B, C)
                    frame_feas = frame_feas.unsqueeze(1) #(B, cls)
                    
            else:
                boxes_tensor = utils.get_valid_boxes_tensor(boxes_tensor)
                person_feas = utils.build_box_feas(feature_maps, utils.tensor_to_list(boxes_tensor), [5,5], self.resolution) #(b*T*K, C)
                person_feas = person_feas.view(-1, self.T, self.num_players, person_feas.size(-1)) # B, T, N, C
                person_feas = self.person_embedding(person_feas) # B, T, N, C
                frame_feas = utils.build_frame_feas(person_feas) #(B, T, C)
            person_scores = self.person_classifier(frame_feas) #(B, T, cls)
#           person_scores = self.frame_classifier(frame_feas)
            person_scores = torch.squeeze(torch.mean(person_scores, 1), 1)

#         # build feature from entities
#         feas_list = self.build_multi_feas(feature_maps, boxes_tensor, img_paths, self.crop_size_dict, self.resolution, self.entity_choices)

#         # Predicting the category of group activity.
#         entity_score_list = []
#         for i, feas in enumerate(feas_list):
#             entity_score_list.append(self.fusion_weights[i]*self.classifier_list[i](feas))
#         entity_scores = torch.sum(torch.stack(entity_score_list), dim=0)
# #         print('entity_scores', entity_scores.size())
#         if self.T !=1:
#             entity_scores = torch.squeeze(torch.mean(entity_scores, 1), 1)
            
            
        if self.frame:
            if self.person:
                return frame_scores, person_scores
#                 scores = self.fused_classifier(torch.cat((sparse_frame_feas, person_based_frame_feas), -1))#.unsqueeze(1) #(B, T, cls)
#                 return torch.squeeze(torch.mean(scores, 1), 1)
            return frame_scores
        else:
            if self.person:
                return spatial_dense_relation, temporal_dense_relation, person_scores
            print('error')
            
#         if self.person and not self.frame:
#             return entity_scores
#         if not self.person_SAM and self.frame_SAM:
#             return frame_scores
#         if self.person_SAM and self.frame_SAM:
#             return frame_scores, entity_scores


    def Pruning(self, box_feas, dense_relation, K=6):
        # matrix: b, s,s
        b, S, S = dense_relation.size()
        relation_degree = torch.zeros(b, S)
        for j in range(S):
            relation_degree[:, j] = torch.sum(dense_relation[:, j, :], -1) + torch.sum(dense_relation[:, :, j], -1)
        _, indices = torch.topk(relation_degree, K, dim=-1, largest=True)
        idx, _ = torch.sort(indices, dim=-1)
        idx = idx.unsqueeze(2).expand(idx.size(0), idx.size(1), box_feas.size(-1)).cuda()
        return torch.gather(box_feas, dim=1, index=idx)
    

    
    def build_multi_feas(self, feature_maps, boxes_tensor, img_paths, crop_size_dict, resolution=[720, 1280], entity_choices=[1,2,3]):
        # init
        feas_list = []
        img_paths = utils.reshape_list(img_paths)
        # iterate the choices to get a list of feas
        for i, scale in enumerate(entity_choices):
            if scale==0:
                # build frame-level features
                # Extracting feas for each frame directly, which retains the spatial structure of group activity.
                frame_feature_maps = F.adaptive_avg_pool2d(feature_maps, crop_size_dict[scale])
                frame_feas = torch.flatten(frame_feature_maps, 1) # b*t, c*h*w
                frame_feas = frame_feas.view(int(frame_feas.size(0)/self.T), self.T, -1) # b， t, c*h*w
            else:
                if self.person_SAM:
                    entity_feas = self.batch_build_sparse_entity_feas(feature_maps, boxes_tensor, img_paths, crop_size_dict[scale], resolution, i, scale) # B*T, K, D'
#                     frame_feas = self.build_pairwise_frame_feas(entity_feas, i)
                    frame_feas = utils.build_frame_feas(entity_feas)
                else:
                    boxes_tensor = utils.get_valid_boxes_tensor(boxes_tensor)
                    entity_feas = utils.build_box_feas(feature_maps, utils.tensor_to_list(boxes_tensor), crop_size_dict[scale], resolution) #(b*T*K, C)
                    entity_feas = entity_feas.view(-1, self.T, self.num_players, entity_feas.size(-1)) # b, T, K, C
                    entity_embeddings = self.entity_embedding_list[i](entity_feas) # b, T, N, C
                    frame_feas = utils.build_frame_feas(entity_embeddings)

            feas_list.append(frame_feas)
        return feas_list # M, b*T, N, D', M denote the num of entityies_choice, where N denote the number of entities(i.e., persons, regions), D' is new dim of feature

    def build_sparse_entity_feas(self, feature_maps, boxes_tensor, img_paths, crop_size, resolution, scale=1):
        # extract features for N entities from the feature map, and normalize them into K entities. 
        # For convient, we fill many <0,0,0,0> in boxes_tensor, thus N=100. And K is a constant depending on the dataset.
        # feature_maps: (B*T, C, H, W), boxes_tensor: (b*T, N=100, 4).
        # output: (b, T, K, D')
        
        feature_maps = feature_maps.view(-1, self.T, *feature_maps.size()[-3:]) # B, T, C, H, W
        boxes_tensor = boxes_tensor.view(-1, self.T, *boxes_tensor.size()[-2:]) # B, T, N, 4
        outputs = []
        
        for i in range(feature_maps.size(0)):
            # get valid boxes.
            entities_tensor = utils.get_valid_boxes_tensor(boxes_tensor[i]) # T, N, 4
            N, K = entities_tensor.size(-2), self.num_players
            
            # construct different entities.
#             since = time.time()
            if not scale==1:
                truth_boxes_tensor = utils.get_truth_boxes_tensor(boxes_tensor[i])
                N = truth_boxes_tensor.size(-2)
                entities_tensor = utils.get_etities_tensor(self.cluster_dict, truth_boxes_tensor, img_paths[i*self.T:(i+1)*self.T], resolution, int(N*scale))
                # we need set int(N*scale)>K*scale, where K*scale is num of nodes of the sparse relation graph.
                if entities_tensor.size(-2)<int(K*scale):
                    entities_tensor = utils.batch_formate_boxes(entities_tensor, lower_K=int(K*scale))
                N, K = entities_tensor.size(-2), int(K*scale)
#             print('read pre-clustering info for one sequence, takes', (time.time()-since), 's')
            
            # extract N box-feature from feature map by valid boxes.
            entity_feas = utils.build_box_feas(feature_maps[i], utils.tensor_to_list(entities_tensor), crop_size, self.resolution) # (T*N, C')
            entity_feas = entity_feas.view(self.T, entity_feas.size(0)//self.T, entity_feas.size(-1)) # T, N, C'
            entity_embeddings = self.person_embedding(entity_feas) # T, N, C''
            entity_embeddings = torch.transpose(entity_embeddings, 1, -1) # (T, C'', N)
            
            # compute relateness
#             _, dense_relation = self.Dense_Relation(entity_embeddings)
            entity_embeddings, dense_relation = self.person_Dense_Relation(entity_embeddings) # Added by Mr. Yan
            
            # pruning N to K
            entity_embeddings = torch.transpose(entity_embeddings, 1, -1) # (T, N, C'')
            sparse_entity_feas = self.Pruning(entity_embeddings, dense_relation, K).unsqueeze(0)

            #sparse_box_feas = box_feas.unsqueeze(0)
            outputs.append(sparse_entity_feas)
        return torch.cat(outputs, dim=0)
    
    def batch_build_sparse_entity_feas(self, feature_maps, boxes_tensor, img_paths, crop_size, resolution):
        # extract features for N entities from the feature map, and normalize them into K entities. 
        # For convient, we fill many <0,0,0,0> in boxes_tensor, thus N=100. And K is a constant depending on the dataset.
        # feature_maps: (b*T, C, H, W), boxes_tensor: (b*T, N=100, 4).
        # output: (b*T, K, C'')
        
        # get valid boxes.
        entities_tensor = utils.get_valid_boxes_tensor(boxes_tensor) # b*T, N, 4
        N, K = entities_tensor.size(-2), self.num_players

        # extract N box-feature from feature map by valid boxes.
        entity_feas = utils.build_box_feas(feature_maps, utils.tensor_to_list(entities_tensor), crop_size, resolution) # (b*T*N, C')
        entity_feas = entity_feas.view(-1, N, entity_feas.size(-1)) # B*T, N, C
#         entity_feas = entity_feas.view(self.T, entity_feas.size(0)//self.T, entity_feas.size(-1)) # T, N, C'
#         entity_embeddings = self.entity_embedding_list[num_layer](entity_feas) # B*T, N, C''
        entity_embeddings = self.person_embedding(entity_feas) # B*T, N, C''
        entity_embeddings = torch.transpose(entity_embeddings, -2, -1) # (B*T, C'', N)
#         entity_embeddings = entity_embeddings.view(-1, *entity_embeddings.size()[-2:])

        # compute relateness
        entity_embeddings, dense_relation = self.person_Dense_Relation(entity_embeddings)

        # pruning N to K
        entity_embeddings = torch.transpose(entity_embeddings, 1, -1) # (b*T, N, C'')
        sparse_entity_feas = self.Pruning(entity_embeddings, dense_relation, K) # (b*T, K, C'')
        return dense_relation, sparse_entity_feas.view(-1, self.T, *sparse_entity_feas.size()[-2:])
    
    def build_pairwise_frame_feas(self, feas, num_layer, position=False):
        # feas: b, T, K, C
        b, T, K, C = feas.size()
        pairwise_frame_feas = torch.zeros((b, T, int(((K-1)*K)/2), C*2)).cuda()
        idx = 0
        for i in range(K):
            for j in range(i+1, K):
                pairwise_frame_feas[:, :, idx, :] = torch.cat((feas[:, :, i, :], feas[:, :, j, :]), dim=-1)
                idx += 1
        pairwise_frame_feas = self.pairwise_frame_feas_embedding_list[num_layer](pairwise_frame_feas)
        pairwise_frame_feas, _ = torch.max(pairwise_frame_feas, dim=-2)
        return pairwise_frame_feas
    
    def build_sparse_frame_maps_feas(self, feature_maps, boxes_tensor, T, L):
        # feature_maps (B*T, C, H, W)
        _, C, H, W = feature_maps.size()
#         since = time.time()
        # pooling feature
        frame_feas = torch.flatten(F.adaptive_avg_pool2d(feature_maps, (1, 1)), 1) # b*t, c*h*w
        frame_feas = frame_feas.view(-1, T, frame_feas.size(-1)) #(B, T, C)
#         print('--------------pooling feature, takes', time.time()-since)
        
#         since = time.time()
        ## compute relateness
        frame_feas = torch.transpose(frame_feas, -2, -1) #(B, C, T)
        frame_feas, dense_relation = self.frame_Dense_Relation(frame_feas)
#         print('--------------compute relateness, takes', time.time()-since)

#         since = time.time()
        ## pruning T to L
        frame_feas = torch.transpose(frame_feas, -2, -1) # (B, T, C)
        sparse_frame_feas = self.Pruning(frame_feas, dense_relation, L)
#         print('--------------pruning frame_feas, takes', time.time()-since)
        
#         since = time.time()
        # feature_maps (B, T, C, H, W)
        feature_maps = feature_maps.view(-1, T, C*H*W) # (B, T, CHW)
        sparse_feature_maps = self.Pruning(feature_maps, dense_relation, L)
        sparse_feature_maps = sparse_feature_maps.view(-1, C, H, W) # (B*L, C, H, W)
#         print('--------------pruning feature_maps, takes', time.time()-since)
        
        # boxes
        _, N, _ = boxes_tensor.size() #(B*T, N, 4)
        boxes_tensor = boxes_tensor.view(boxes_tensor.size(0)//T, T, -1) # (B, T, 4)
        sparse_boxes_tensor = self.Pruning(boxes_tensor, dense_relation, L)
        sparse_boxes_tensor = sparse_boxes_tensor.view(-1, N, 4) # (B*L, C, H, W)

        return sparse_feature_maps, sparse_frame_feas, sparse_boxes_tensor

            
    
def IN(pretrained=False, **kwargs):
    model = iN(**kwargs)
    if pretrained:
        pretrained_dict = torch.load('****.pkl')
        model.load_state_dict(pretrained_dict)
    return model