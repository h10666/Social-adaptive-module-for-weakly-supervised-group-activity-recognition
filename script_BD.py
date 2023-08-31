import os
import sys
sys.path.append(".")

# modes = ['pre', 'prob', 'all']

# modes = ['prob']

# modes = ['pre']
# frame_SAM = False
# person_SAM = True

# num_selected_frames = 20
# for mode in modes:
#     #####################################################################
#     if mode == 'pre':
# #         start_K, end_K, step = [16, 16, -1]
#         start_K, end_K, step = [12, 5, -1]
#         mode_arg = start_K ##### modified by Mr. Yan
#     elif mode == 'prob':
#         start_K, end_K, step = [16, 15, -1] #[16, 8, -1]
#         mode_arg = 0.9
#     elif mode == 'all':
#         start_K, end_K, step = [32, 17, -3]

#     save_root = 'weights/BD/resnet18/'+mode+'/w_pSAN'
#     num_epochs = 30
#     for K in range(start_K, end_K, step):
#         save_folder = os.path.join(save_root, 'K_' + str(K))
#         if not os.path.exists(save_folder):
#             os.makedirs(save_folder)
#             print(save_folder, 'is created!')
#         else:
#             print(save_folder, 'is existed!')
        
#         ##########
# #         mode_arg = K
#         os.system('python GAR.py --dataset_name BD --resolution 224,224 --frame_SAN {} --person_SAN {} --entity_choices 1 --fusion_weights 1 --tracks_mode {} --mode_arg {} --num_players {} --num_epochs {} --num_selected_frames {} --save_path {} > {}/log'.format(frame_SAN, person_SAN, mode, mode_arg, K, num_epochs, num_selected_frames, save_folder, save_folder))
#     ######################################################################



# # 10 frames based single frame classification
# python GAR.py --resolution 224,224 --SAN False --entity_choices 0 --num_selected_frames 10 --num_epochs 30 --save_path weights/BD/resnet18/single_frame/> weights/BD/resnet18/single_frame/log

# # 10 frames based PreN w/o SAN
# python GAR.py --SAN False --entity_choices 1 --tracks_mode pre --mode_arg 10 --num_players 10 --num_selected_frames 10 --num_epochs 30 --save_path weights/BD/resnet18/Pre_N_wo_SAN/> weights/BD/resnet18/Pre_N_wo_SAN/log

# # 10 frames, low resolution based PreN w/o SAN
# python GAR.py --resolution 224,224 --SAN False --entity_choices 1 --tracks_mode pre --mode_arg 10 --num_players 10 --num_selected_frames 10 --num_epochs 30 --save_path weights/BD/resnet18/low_res_Pre_N_wo_SAN/> weights/BD/resnet18/low_res_Pre_N_wo_SAN/log

# # 10 frames, low resolution based PreN w/ SAN
# python GAR.py --resolution 224,224 --SAN True --entity_choices 1 --tracks_mode pre --mode_arg 12 --num_players 6 --num_selected_frames 10 --num_epochs 30 --save_path weights/BD/resnet18/low_res_Pre_N_w_SAN/> weights/BD/resnet18/low_res_Pre_N_w_SAN/log

# # 20 frames, low resolution based PreN w/ person, frame SAN
# python GAR.py --resolution 224,224 --frame_SAN True --num_selected_frames 20 --num_epochs 30 --save_path weights/BD/resnet18/low_res_frame_w_SAN/> weights/BD/resnet18/low_res_frame_w_SAN/log

# # 20 frames, low resolution based PreN w/ frame SAN(20->20)
# python GAR.py --resolution 224,224 --frame_SAN True --num_selected_frames 20 --num_epochs 30 --save_path weights/BD/resnet18/low_res_bmp_frame_w_SAN/> weights/BD/resnet18/low_res_bmp_frame_20_20_w_SAN/log





#### SAM final version scripts ###
# # 20 frames, low resolution based PreN w/ person SAM(12->8), frame SAM(20->10)
# python GAR.py --resolution 224,224 --frame True --frame_SAM True --N_f 20 --K_f 10 --person True --person_SAM True --tracks_mode pre --mode_arg 12 --num_players 8 --num_epochs 30 --save_path weights/BD/resnet18/pre/w_fSAM_Nf20_Kf10_pSAM_Np12_Kp8/> weights/BD/resnet18/pre/w_fSAM_Nf20_Kf10_pSAM_Np12_Kp8/log

# # low resolution based PreN w/ only frame SAM(20->10)
# python GAR.py --resolution 224,224 --frame True --frame_SAM True --N_f 20 --K_f 10 --num_epochs 30 --save_path weights/BD/resnet18/pre/w_fSAM_Nf20_Kf10/> weights/BD/resnet18/pre/w_fSAM_Nf20_Kf10/log

# # low resolution based PreN w/ frame SAM(20->10), person SAM(12->8)
# os.system('python GAR.py --resolution 224,224 --frame True --frame_SAM True --N_f 20 --K_f 10 --person True --person_SAM True --tracks_mode pre --mode_arg 12 --num_players 8 --num_epochs 30 --save_path weights/BD/resnet18/pre/bt16_w_feafused_fpSAM_Nf20_Kf10_Np12_Kp8/> weights/BD/resnet18/pre/bt16_w_feafused_fpSAM_Nf20_Kf10_Np12_Kp8/log')


# ## low res, PreN All Frames Classification, (72)
# N_f = 72
# save_folder = 'weights/BD/resnet18/pre/bt4_allframes_cls/Nf_'+str(N_f)+'/'
# if not os.path.exists(save_folder):
#     os.makedirs(save_folder)
# os.system('python GAR.py --resolution 224,224 --frame True --N_f {} --num_epochs 15 --save_path {} > {}/log'.format(N_f, save_folder, save_folder))



##########################################BASELINES##########################################################
#
# ## low res, PreN Frame Classification, TSN-type(72->20)
# for N_f in range(20, 10, -1):
#     save_folder = 'weights/BD/resnet18/pre/bt16_frame_cls_TSN_testN/Nf_'+str(N_f)+'/'
#     if not os.path.exists(save_folder):
#         os.makedirs(save_folder)
#     os.system('python GAR.py --resolution 224,224 --frame True --N_f {} --num_epochs 15 --save_path {} > {}/log'.format(N_f, save_folder, save_folder))

# ## low res, PreN Frame Classification, TSN-type(72->20) based single-scale TRN
# N_f = 20
# save_folder = 'weights/BD/resnet18/pre/bt16_frame_cls/TRN/Nf_'+str(N_f)+'/'
# if not os.path.exists(save_folder):
#     os.makedirs(save_folder)
# os.system('python GAR.py --resolution 224,224 --frame True --N_f {} --frame_TRN True --num_epochs 15 --save_path {} > {}/log'.format(N_f, save_folder, save_folder))


## low res, PreN Frame Classification, TSN-type(72->20) based resNet18_I3D
# N_f = 20
# save_folder = 'weights/BD/resnet18/pre/bt16_frame_cls/resNet18_I3D/Nf_'+str(N_f)+'/'
# if not os.path.exists(save_folder):
#     os.makedirs(save_folder)
# os.system('python GAR.py --resolution 224,224 --backbone_name resNet18_I3D --frame True --N_f {} --num_epochs 15 --save_path {} > {}/log'.format(N_f, save_folder, save_folder))

## low res, PreN Frame Classification, TSN-type(72->20) based resNet18_I3D_NLN
# N_f = 20
# save_folder = 'weights/BD/resnet18/pre/bt16_frame_cls/resNet18_I3D_NLN/Nf_'+str(N_f)+'/'
# if not os.path.exists(save_folder):
#     os.makedirs(save_folder)
# os.system('python GAR.py --resolution 224,224 --backbone_name resNet18_I3D_NLN --frame True --N_f {} --num_epochs 15 --save_path {} > {}/log'.format(N_f, save_folder, save_folder))

############################################BASELINES###########################################################







# #####################################################################
# # fix N_f = 21, test K_f = [21, 0], step=-3
# modes = ['pre']
# frame, frame_SAM = [True, True]
# person, person_SAM = [True, True]

# N_f = 20 # the number of input frames, sampled from the whole video clip.
# N_p = 12
# K_p = 8
# for mode in modes:
#     if mode == 'pre':
#         start_K_f, end_K_f, step = [20, 5, -1]

#     save_root = 'weights/BD/resnet18/'+mode+'/w_05_fSAM_1_pSAM_test_Kf'
#     num_epochs = 15
#     for K_f in range(start_K_f, end_K_f, step):
#         save_folder = os.path.join(save_root, 'Kf_' + str(K_f))
#         if not os.path.exists(save_folder):
#             os.makedirs(save_folder)
#             print(save_folder, 'is created!')
#         else:
#             print(save_folder, 'is existed!')
        
#         os.system('python GAR.py --dataset_name BD --resolution 224,224 --frame {} --frame_SAM {} --N_f {} --K_f {} --person {} --person_SAM {} --tracks_mode {} --mode_arg {} --num_players {} --num_epochs {} --save_path {} > {}/log'.format(frame, frame_SAM, N_f, K_f, person, person_SAM, mode, N_p, K_p, num_epochs, save_folder, save_folder))
# ######################################################################



#############################################VARIANTS###########################################################
#
##### QUAN-N

## w/o SAM 
## low res, N_f=20, Pre-N w/o SAM, person only (N_p=8)
# save_folder = 'weights/BD/resnet18/pre/bt16_wo_SAM/Np8_Kp8/'
# if not os.path.exists(save_folder):
#     os.makedirs(save_folder)
# os.system("python GAR.py --resolution 224,224 --N_f 20 --person True --tracks_mode pre --mode_arg 8 --num_players 8 --num_epochs 15 --save_path {}> {}/log".format(save_folder, save_folder))

# w/ SAM(p)
# w/ SAM(f+p)




#### Prob-N
## w/ SAM(p) N_p(theta)=0.9 K_p=8
# low res, N_f=20, Prob-N w/ SAM(p)
# import numpy as np
# for theta in np.arange(0.95, 0, -0.05):
#     theta = np.around(theta, decimals=2)
#     save_folder = 'weights/BD/resnet18/prob/bt16_w_pSAM/test_theta1/theta_'+(''.join(str(theta).split('.')))
#     if not os.path.exists(save_folder):
#         os.makedirs(save_folder)
#     os.system("python GAR.py --resolution 224,224 --N_f 20 --person True --person_SAM True --tracks_mode prob --mode_arg {} --num_players 8 --num_epochs 15 --save_path {}> {}/log".format(theta, save_folder, save_folder))

# w/ SAM(f+p) N_f=72 K_f=20 N_p(theta)=0.9 K_p=8
## low res, Prob-N w/ SAM(f+p)
# N_f = 20
# for K_f in range(20, 1, -1):
#     save_folder = 'weights/BD/resnet18/prob/bt16_w_fSAM_pSAM/Nf20_Np_theta09_Kp8_test_Kf/Kf_'+str(K_f)
#     if not os.path.exists(save_folder):
#         os.makedirs(save_folder)
#     os.system("python GAR.py --resolution 224,224 --frame True --frame_SAM True --N_f {} --K_f {} --person True --person_SAM True --tracks_mode prob --mode_arg 0.9 --num_players 8 --num_epochs 15 --save_path {}> {}/log".format(N_f, K_f, save_folder, save_folder))



# # #####################################################################
# # test N_f = [72, 20] step=-4, fix K_f = 20, step=-3
# modes = ['pre']
# frame, frame_SAM = [True, True]
# person, person_SAM = [True, True]

# #N_f = 20 # the number of input frames, sampled from the whole video clip.
# K_f = 20
# N_p = 12
# K_p = 8
# for mode in modes:
#     if mode == 'pre':
# #         start_N_f, end_N_f, step = [72, 19, -4]
#         start_N_f, end_N_f, step = [20, 41, 4]

#     save_root = 'weights/BD/resnet18/'+mode+'/bt16_w_0floss_1ploss_fSAM_pSAM_test_Nf'
#     num_epochs = 15
#     for N_f in range(start_N_f, end_N_f, step):
#         save_folder = os.path.join(save_root, 'Nf_' + str(N_f))
#         if not os.path.exists(save_folder):
#             os.makedirs(save_folder)
#             print(save_folder, 'is created!')
#         else:
#             print(save_folder, 'is existed!')
        
#         os.system('python GAR.py --dataset_name BD --resolution 224,224 --frame {} --frame_SAM {} --N_f {} --K_f {} --person {} --person_SAM {} --tracks_mode {} --mode_arg {} --num_players {} --num_epochs {} --save_path {} > {}/log'.format(frame, frame_SAM, N_f, K_f, person, person_SAM, mode, N_p, K_p, num_epochs, save_folder, save_folder))
# ######################################################################



# confusion script
# os.system('python GAR.py --dataset_name BD --resolution 224,224 --frame True --frame_SAM True --N_f 21 --K_f 18 --person True --person_SAM True --tracks_mode pre --mode_arg 12 --num_players 8')
# os.system('python GAR.py --dataset_name BD --resolution 224,224 --N_f 20 --person True --person_SAM True --tracks_mode pre --mode_arg 14 --num_players 8')

# # #####################################################################
# # test N_f = [72, 20] step=-4, fix K_f = 20, step=-3
# modes = ['pre']
# frame, frame_SAM = [True, True]
# # person, person_SAM = [True, True]

# N_f = 20 # the number of input frames, sampled from the whole video clip.
# for mode in modes:
#     if mode == 'pre':
#         start, end, step = [10, 21, 2]

#     save_root = 'weights/BD/resnet18/'+mode+'/bt16_w_fSAM_test_Kf'
#     num_epochs = 15
#     for K_f in range(start, end, step):
#         save_folder = os.path.join(save_root, 'Nf_' + str(K_f))
#         if not os.path.exists(save_folder):
#             os.makedirs(save_folder)
#             print(save_folder, 'is created!')
#         else:
#             print(save_folder, 'is existed!')
        
#         os.system('python GAR.py --dataset_name BD --resolution 224,224 --frame {} --frame_SAM {} --N_f {} --K_f {} --num_epochs {} --save_path {} > {}/log'.format(frame, frame_SAM, N_f, K_f, num_epochs, save_folder, save_folder))
# ######################################################################


# # temporal modeling for person-based frame feature
# N_f = 20
# for L in range(2, N_f+1, 2):
#     save_folder = 'weights/BD/resnet18/pre/bt16_w_pSAM_temporalSAM_Nf20_test_L_fixedbackbone/L_'+str(L)
#     if not os.path.exists(save_folder):
#         os.makedirs(save_folder)
#         print(save_folder, 'is created!')
#     else:
#         print(save_folder, 'is existed!')
#     os.system('python GAR.py --dataset_name BD --resolution 224,224 --N_f {} --person True --person_SAM True --temporal_SAM True --L {} --tracks_mode pre --mode_arg 14 --num_players 8 --num_epochs {} --save_path {} > {}/log'.format(N_f, L, 15, save_folder, save_folder))
    

    
# # temporal modeling for person-based frame feature
# N_f = 20
# for L in range(2, N_f+1, 2):
#     save_folder = 'weights/BD/resnet18/pre/bt16_w_pSAM_temporalSAM_Nf20_test_L/L_'+str(L)
#     if not os.path.exists(save_folder):
#         os.makedirs(save_folder)
#         print(save_folder, 'is created!')
#     else:
#         print(save_folder, 'is existed!')
#     os.system('python GAR.py --dataset_name BD --resolution 224,224 --N_f {} --person True --person_SAM True --temporal_SAM True --L {} --tracks_mode pre --mode_arg 14 --num_players 8 --num_epochs {} --save_path {} > {}/log'.format(N_f, L, 15, save_folder, save_folder))

    
# 'python GAR.py --dataset_name BD --resolution 224,224 --N_f 20 --person True --person_SAM True --temporal_SAM True --L 6 --tracks_mode pre --mode_arg 14 --num_players 8'



