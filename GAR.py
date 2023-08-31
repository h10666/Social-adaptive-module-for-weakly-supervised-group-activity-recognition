"""
	Group Activity Recognition
"""
import argparse
from Piplines import *
import torch
import Pre
import time

# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
torch.backends.cudnn.enabled = True

# torch.set_printoptions(profile="full")

parser = argparse.ArgumentParser()
parser.add_argument('--dataset_root', type=str, default='./dataset/', help='Please set the root folder of datasets')
parser.add_argument('--dataset_name', type=str, default='BD', choices=['VD', 'BD'], help='Please choose one of the dataset')
# parser.add_argument('--num_players', type=int, default=12, help='Please set the certain num_players for construct relation graph')

opt, unknown = parser.parse_known_args()


# Step Zero: Dataset Preprocessing
# print('Please wait for tracking, cropping and resizing!')
# print('It will take about 300+ min for VD')
# track_since = time.time()
# Pre.Processing(opt.dataset_root, opt.dataset_name, operation='track')
# print('Tracking, cropping and resizing {} in {:.0f}m {:.0f}s'.format(opt.dataset_name,
#             (time.time() - track_since) // 60, (time.time() - track_since) % 60))


# Step One: Group activity recognition
# pipline_A = Piplines(opt.dataset_root, opt.dataset_name, 'stage_A')
# pipline_A.trainval()
# weights_path = os.path.join('weights', opt.dataset_name)
# os.system('mv {} {}'.format(weights_path+'/best_wts.pth', weights_path+'/MODEL_STAGE_A.pth'))

# pipline_B = Piplines(opt.dataset_root, opt.dataset_name, 'stage_B')
# pipline_B.trainval()

# Step Two: Evaluate
pipline = Piplines(opt.dataset_root, opt.dataset_name, 'stage_B')
# dataset_size = 1337 if opt.dataset_name == 'VD' else 621
# since = time.time()
pipline.evaluate(model_path='weights/BD/resnet18/pre/bt16_w_pSAM_temporalSAM_test_L/L_6/best_acc_wts.pth')
# print('infer one sequence, takes', (time.time()-since)/dataset_size, 's')
