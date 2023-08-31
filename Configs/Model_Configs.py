"""
	Model_Configs
"""
import argparse


class Model_Configs(object):
    """docstring for Model_Configs"""

    def __init__(self, dataset_name, stage_name='stage_B'):
        super(Model_Configs, self).__init__()
        self.dataset_name = dataset_name
        self.stage_name = stage_name
        self.confs_dict = {
            'VD': {
                'stage_A': {
                    'backbone_name': 'resNet18', # resNet18, myInception_v3, inception_v3
                    'train_backbone': True,
                    'backbone_path': '',
                },
                'stage_B': {
                    'backbone_name': 'resNet18',
                    'train_backbone': True,
                    'backbone_path': '', # 'weights/VD/MODEL_STAGE_A.pth',
                }
            },
            'BD': {
                'stage_A': {
                    'backbone_name': 'myInception_v3', # resNet18, myInception_v3, inception_v3
                    'train_backbone': True,
                    'backbone_path': '',
                },
                'stage_B': {
                    'backbone_name': 'resNet18', #resNet18, resNet50_I3D, inceptionI3d
                    'train_backbone': True,
                    'backbone_path': 'weights/BD/resnet18/pre/bt16_w_pSAM/test_Np/N_14/best_avg_acc_wts.pth', # 'weights/VD/MODEL_STAGE_A.pth',
                }
            }
        }

    def configuring(self):
        parser = argparse.ArgumentParser()
        model_confs = self.confs_dict[self.dataset_name][self.stage_name]
        parser.add_argument('--backbone_name', type=str, default=model_confs['backbone_name'])
        parser.add_argument('--train_backbone', type=bool, default=model_confs['train_backbone'])
        parser.add_argument('--backbone_path', type=str, default=model_confs['backbone_path'])
        
        args, unknown = parser.parse_known_args()
        return args
