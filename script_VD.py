import os
import sys
sys.path.append(".")


########################### Quan-N #################################
## w/ SAM(p) N_p=16 K_p=12 alexNet
# low res, N_f=20, Prob-N w/ SAM(p)
# save_folder = 'weights/VD/alexnet/pre/bt16_w_pSAM/Nf10_Np16_Kp12'
# if not os.path.exists(save_folder):
#     os.makedirs(save_folder)
# os.system("python GAR.py --dataset_name VD --backbone_name alexNet --N_f 10 --person True --person_SAM True --tracks_mode pre --mode_arg 16 --num_players 12 --num_epochs 30 --save_path {}> {}/log".format(save_folder, save_folder))

# ## w/ SAM(p) N_p=16 K_p=12 inception
# # low res, N_f=20, Prob-N w/ SAM(p)
# save_folder = 'weights/VD/inception/pre/bt16_w_pSAM/Np16_Kp12'
# if not os.path.exists(save_folder):
#     os.makedirs(save_folder)
# os.system("python GAR.py --dataset_name VD --backbone_name myInception_v3 --N_f 3 --person True --person_SAM True --tracks_mode pre --mode_arg 16 --num_players 12 --num_epochs 30 --save_path {}> {}/log".format(save_folder, save_folder))

## w/ SAM(p) N_f=10, N_p=16 K_p=12 resnet18
# low res, N_f=10, Prob-N w/ SAM(p)
for K_f in range(3, 11, 1):
    save_folder = 'weights/VD/resnet18/pre/bt16_w_pSAM_fSAM_testKf/Kf_' + str(K_f)
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    os.system("python GAR.py --resolution 224,224 --dataset_name VD --frame True --frame_SAM True --N_f 10 --K_f {} --person True --person_SAM True --tracks_mode pre --mode_arg 16 --num_players 12 --num_epochs 30 --save_path {}> {}/log".format(K_f, save_folder, save_folder))


##### Prob-N
### w/ SAM(p) N_p(theta)=0.9 K_p=8
## low res, N_f=20, Prob-N w/ SAM(p)
# save_folder = 'weights/BD/resnet18/prob/bt16_w_pSAM/Np_theta09_Kp8'
# if not os.path.exists(save_folder):
#     os.makedirs(save_folder)
# os.system("python GAR.py --resolution 224,224 --N_f 10 --person True --person_SAM True --tracks_mode prob --mode_arg 0.9 --num_players 8 --num_epochs 15 --save_path {}> {}/log".format(save_folder, save_folder))