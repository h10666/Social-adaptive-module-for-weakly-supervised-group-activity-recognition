"""
	Common Piplines
"""
from abc import ABCMeta, abstractmethod
from torchvision import transforms
import torch

import Configs
import Data
import Models
from Solver import *


class Piplines(object):
    """docstring for Piplines"""

    def __init__(self, dataset_root, dataset_name, stage_name):
        super(Piplines, self).__init__()
        self.dataset_root = dataset_root
        self.dataset_name = dataset_name
        self.stage_name = stage_name
        self.configuring()
        
    def configuring(self):
        # Dataset configs:
        self.data_confs = Configs.Data_Configs(self.dataset_root, self.dataset_name).configuring()
        
        print(self.dataset_name, 'data_confs: \n', self.data_confs)
        
        # Model configs:
        self.model_confs = Configs.Model_Configs(self.dataset_name, self.stage_name).configuring()
        
        self.data_loaders, self.data_sizes = self.loadData(self.data_confs)
        self.net = self.loadModel()
        
        if torch.cuda.is_available():
#             self.net = self.net.cuda()
            self.net = torch.nn.DataParallel(self.net).cuda()
        print(self.net)

        # Solver configs:
        self.solver_confs = Configs.Solver_Configs(self.dataset_name, self.data_loaders, self.data_sizes, self.net, self.data_confs).configuring()
        print('solver_confs: \n', self.solver_confs)
        self.solver = Solver(self.net, self.data_confs, self.solver_confs)

    def loadModel(self, pretrained=False):
        net = Models.IN(pretrained, dataset_name=self.dataset_name, data_confs=self.data_confs, model_confs=self.model_confs)
        return net

    def trainval(self):
        self.solver.train_model()

    def test(self):
        self.solver.test_model()

    def loadData(self, data_confs, phases=['trainval', 'test']):
        if data_confs.data_type == 'img':
            data_transforms = {
                'trainval': transforms.Compose([
                    transforms.Resize(self.data_confs.resolution),
                    #transforms.RandomResizedCrop(224),
                    #transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    #transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                ]),
                'test': transforms.Compose([
                    transforms.Resize(self.data_confs.resolution),
                    # transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    #transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                ]),
            }
        else:
            data_transforms = None

        data = {phase: eval('Data.' + data_confs.data_type)(data_confs, phase, data_transforms[phase] if data_transforms else None) for phase in phases}
        data_loaders = {phase: torch.utils.data.DataLoader(data[phase], batch_size=data_confs.batch_size[phase], num_workers=16, shuffle=True) for phase in phases}
        print(data_transforms)
        # num_workers=8
        # 'end_to_end':
        data_sizes = {phase: len(data[phase]) for phase in phases}

        return data_loaders, data_sizes

    
    def evaluate(self, model_path=None):
        if model_path:
            pretrained_dict = torch.load(model_path)
        else:
            pretrained_dict = torch.load('./weights/'+self.dataset_name+'/MODEL_STAGE_A.pth')
        self.net.load_state_dict(pretrained_dict)
        self.net.eval()
        self.solver.evaluate()