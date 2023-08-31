"""
    Pre_Configs
"""
import argparse
import os
import utils

class Data_Configs(object):
    """docstring for Pre_Configs"""

    def __init__(self, dataset_root, dataset_name):
        super(Data_Configs, self).__init__()
        self.dataset_root = dataset_root
        self.dataset_name = dataset_name
        self.confs_dict = {
            'VD': {
                'data_type': 'img',
                'cur_folder': 'videos_bmp',
                'num_videos': 55,
                'num_frames': 10,
                'N_f': 3,
                'num_players': 12,
                'num_classes':6,
                'num_groups': 1,
                'resolution': (720, 1280), #(224, 224)

                #'action_list': ['blocking', 'digging', 'falling', 'jumping', 'moving', 'setting', 'spiking', 'standing', 'waiting'],
                'activity_list': ['l-pass', 'r-pass', 'l_set', 'r_set', 'l-spike', 'r_spike', 'l_winpoint', 'r_winpoint'],
                'splits': {
                    'trainval': [0, 1, 2, 3, 6, 7, 8, 10, 12, 13, 15, 16, 17, 18, 19, 22, 23, 24, 26, 27, 28, 30, 31, 32, 33, 36, 38, 39, 40, 41, 42, 46, 48, 49, 50, 51, 52, 53, 54],
                    'test': [4, 5, 9, 11, 14, 20, 21, 25, 29, 34, 35, 37, 43, 44, 45, 47]
                },                
                'batch_size': {'trainval': 4, 'test': 8}
                
            },
            'BD': {
                'data_type': 'img',
                'cur_folder': 'videos_bmp',
                'num_videos': 181,
                'num_frames': 72,
                'N_f': 10, # the number of input frames
                'num_players': 10,
                'num_classes': 9,
                'num_groups': 1,
                'resolution': (720, 1280),
                'activity_list': ['three-others-shot-success', 'three-others-shot-failure-off-rebound', 'three-others-shot-failure-def-rebound',
                 'two-layup-shot-success', 'two-layup-shot-failure-off-rebound', 'two-layup-shot-failure-def-rebound',
                 'two-others-shot-success', 'two-others-shot-failure-off-rebound', 'two-others-shot-failure-def-rebound',
#                  'two-dunk-shot-success', 'two-dunk-shot-failure-off-rebound', 'two-dunk-shot-failure-def-rebound'
                                 ],

                'splits': {
                    'trainval': [21801058,21801127,21801017,21801211,21801043,21800980,21801213,21801110,21801220,21801141,21801107,21801086,21801051,21801155,21800995,21801111,21800991,21801217,21801095,21801218,21801150,21801044,21801121,21800960,21801172,21801039,21801054,21801171,21801015,21801124,21801037,21801144,21801098,21801013,21801019,21801008,21801004,21801106,21800934,21801115,21800999,21801149,21801224,21800972,21801108,21801072,21801161,21801065,21801090,21800979,21800919,21800994,21801053,21801114,21801151,21801216,21801074,21800938,21801168,21800909,21801012,21800929,21801135,21800982,21801160,21800949,21801228,21801097,21801226,21800952,21801140,21800983,21801052,21801061,21800997,21800975,21801126,21801189,21801064,21801125,21801209,21800987,21801089,21801163,21800992,21801219,21801136,21800976,21801128,21801068,21800966,21801139,21800981,21801011,21801085,21801116,21801221,21801119,21801214,21801048,21800989,21801113,21800985,21801158,21800964,21801215,21801102,21801046,21800984,21801091,21801164,21801002,21801056,21801175,21801167,21801069,21801087,21801060,21801100,21800963,21800973,21800971,21801112,21801018,21801157,21801003,21801006,21801067,21801001,21801147,21801094,21801057,21801230,21801131,21801154,21801042,21801156,21801225,21801045,21801104,21800990,21801210,21801014,21801145,21801148,21801099,21800996,21801223,21801088,21801134,21800968],
                    'test': [21800974,21801076,21801079,21800970,21800965,21801165,21801071,21801049,21801070,21800977,21801162,21801120,21801007,21801078,21801123,21801188,21801038,21800978,21801159,21801152,21801204,21801229,21800988,21801077,21801153,21801063,21801096,21801105,21801050,21801129]
                },
                'batch_size': {'trainval': 4*4, 'test': 1} # 4*4
            },
            'CAD': {
                'data_type': 'img',
                'cur_folder': 'videos',
                
                'num_videos': 44,
                'num_frames': 10,
                'num_players': 5,
                'num_classes':4,
                'num_groups': 1,
                'resolution': (720, 1280),
                
                #'action_list': ['Walking', 'Crossing', 'Waiting', 'Queuing', 'Talking'],
                'activity_list': ['Moving', 'Waiting', 'Queuing', 'Talking'],
                'splits': {
                    'trainval': [7, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 26, 27, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44],
                    'test': [1, 2, 3, 4, 5, 6, 8, 9, 10, 11, 25, 28, 29]
                },
                'batch_size': {'trainval': 300, 'test': 10}
            }
        }

    def configuring(self):
        dataset_confs = self.confs_dict[self.dataset_name]
        parser = argparse.ArgumentParser()
        parser.add_argument('--dataset_folder', type=str, default=os.path.join(self.dataset_root, self.dataset_name, dataset_confs['cur_folder']), help='')
        parser.add_argument('--data_type', type=str, default=dataset_confs['data_type'], choices=['img', 'hdf5', 'npy'], help='the story type for data')
        parser.add_argument('--num_videos', type=dict, default=dataset_confs['num_videos'])
        parser.add_argument('--num_frames', type=int, default=dataset_confs['num_frames'])
        parser.add_argument('--N_f', type=int, default=dataset_confs['N_f'])
        parser.add_argument('--num_players', type=int, default=dataset_confs['num_players'])
#         parser.add_argument('--num_players', type=int) # for excuting scripts in batch, this param is set by K.
        parser.add_argument('--num_classes', type=int, default=dataset_confs['num_classes'])
        parser.add_argument('--num_groups', type=int, default=dataset_confs['num_groups'])
        parser.add_argument('--resolution', type=utils.str2tuple, nargs='?', default=dataset_confs['resolution'])
        parser.add_argument('--activity_list', type=list, default=dataset_confs['activity_list'])
        parser.add_argument('--splits', type=dict, default=dataset_confs['splits'])
        parser.add_argument('--batch_size', type=dict, default=dataset_confs['batch_size'])
        
        args, unknown = parser.parse_known_args()
        return args
