import os
import sys
sys.path.append(".")


# mode = 'all' # 'pre12', 'prob07', 'all'
# modes = ['pre', 'prob', 'all']

# modes = ['prob']

modes = ['prob']
SAN = True

num_selected_frames = 3
for mode in modes:
    #####################################################################
    if mode == 'pre':
#         start_K, end_K, step = [16, 16, -1]
        start_K, end_K, step = [12, 11, -1]
        mode_arg = start_K ##### modified by Mr. Yan
    elif mode == 'prob':
        start_K, end_K, step = [16, 15, -1] #[16, 8, -1]
        mode_arg = 0.9
    elif mode == 'all':
        start_K, end_K, step = [32, 17, -3]

    save_root = 'weights/VD/detect_' + mode + '_6_cls/resnet18/nopretrain_Prob_w_SAN_test_frame'
    num_epochs = 30
    for K in range(start_K, end_K, step):
        save_folder = os.path.join(save_root, 'K_' + str(K))
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)
            print(save_folder, 'is created!')
        else:
            print(save_folder, 'is existed!')
        
        ##########
#         mode_arg = K
        os.system('python GAR.py --SAN {} --entity_choices 0 --fusion_weights 1 --tracks_mode {} --mode_arg {} --num_players {} --num_epochs {} --num_selected_frames {} --save_path {} > {}/log'.format(SAN, mode, mode_arg, K, num_epochs, num_selected_frames, save_folder, save_folder))
#         os.system('mv weights/VD/best_wts.pth %s/'%(save_folder))
    ######################################################################

    
    
# python GAR.py --tracks_mode 'pre' --mode_arg 16 --num_players 16 --num_epochs 30 
# python GAR.py --tracks_mode 'prob' --mode_arg 0.9 --num_players 16 --num_epochs 30 > cluster_choice2


'''
modes = ['prob']
entity_choices_list = [
    [0.8], 
    [0.7],
    [0.6],
    [0.5],
    [0.4],
    [0.3],
    [0.2],
]
num_epochs = 30
save_root = 'weights/VD/detect_prob_6_cls/resnet18/pruning/K_16/'
for entity_choices in entity_choices_list:
    #####################################################################
    save_folder = os.path.join(save_root, '')
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
        print(save_folder, 'is created!')
    else:
        print(save_folder, 'is existed!')
        
        print('python GAR.py --entity_choices %s --tracks_mode %s --mode_arg %f --num_players %d --num_epochs %d > %s/log'%(entity_choices, mode, mode_arg, K, num_epochs, save_folder))
#         os.system('python GAR.py --entity_choices %s --tracks_mode %s --mode_arg %f --num_players %d --num_epochs %d > %s/log'%(entity_choices, mode, mode_arg, K, num_epochs, save_folder))
#         os.system('mv weights/VD/best_wts.pth %s/'%(save_folder))
    ######################################################################
    
    
# python GAR.py --entity_choices 1 --fusion_weights 1 --tracks_mode 'prob' --mode_arg 0.9 --num_players 16 --num_epochs 30 > multi_scale_08

'python GAR.py --entity_choices {} --tracks_mode {} --mode_arg {} --num_players {} --num_epochs {} > {}/log'.format(entity_choices, mode, mode_arg, K, num_epochs, save_folder)

'''

# STAGE A: 10 frames based single frame classification
# python GAR.py --SAN False --entity_choices 0 --fusion_weights 1 --num_selected_frames 10 --num_epochs 10 --save_path xxx> xxx/log

# STAGE B: 10 frames based prob_09_pruning(N->K, K=12) classification + pairwise frame_feas
# python GAR.py --entity_choices 1 --fusion_weights 1 --tracks_mode 'prob' --mode_arg 0.9 --num_players 12 --num_selected_frames 10