"""
	Solver_Configs
"""
import argparse
import torch
import torch.nn as nn
from torch.optim import lr_scheduler


class Solver_Configs(object):
    """docstring for Solver_Configs"""

    def __init__(self, dataset_name, data_loaders, data_sizes, net, dataset_confs):
        super(Solver_Configs, self).__init__()
        self.dataset_name = dataset_name
        self.dataset_confs = dataset_confs
        
        self.data_loaders = data_loaders
        self.data_sizes = data_sizes

        self.net = net

        self.confs_dict = {
            'VD': {
                'num_epochs': 50,
                'lr_scheduler': {'step_size': 5, 'gamma': 0.5},
                'optimizer': {'method': 'Adam', 'lr': 0.0001, 'arg': (0.9, 0.9)}
            },
            'BD': {
                'num_epochs': 50,
                'lr_scheduler': {'step_size': 5, 'gamma': 0.5},
                'optimizer': {'method': 'Adam', 'lr': 1e-4, 'arg': (0.9, 0.9)}
            },
            'CAD': {
                'num_epochs': 20,
                'lr_scheduler': {'step_size': 10, 'gamma': 0.1},
                'optimizer': {'method': 'Adam', 'lr': 0.0001, 'arg': (0.9, 0.9)}
            }
        }

    def configuring(self):
        solver_confs = self.confs_dict[self.dataset_name]
        parser = argparse.ArgumentParser()
        parser.add_argument('--num_epochs', type=int, default=solver_confs['num_epochs'])
        parser.add_argument('--gpu', type=bool, default=torch.cuda.is_available(), help='*****')

        criterion = nn.CrossEntropyLoss()
        parser.add_argument('--criterion', type=type(criterion), default=criterion)

        
        
        optim = solver_confs['optimizer']
        optimizer = eval('torch.optim.' + optim['method'])(self.net.parameters(), optim['lr'], optim['arg'], weight_decay=1e-4) #weight_decay=1e-4
        parser.add_argument('--optimizer', type=type(optimizer), default=optimizer)

        
        
        # Decay LR by a factor of 'gamma' every 'step_size' epochs
        lr_sch = solver_confs['lr_scheduler']
        print('lr_scheduler: ', lr_sch)
        exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=lr_sch['step_size'], gamma=lr_sch['gamma'])
        parser.add_argument('--exp_lr_scheduler', type=type(exp_lr_scheduler), default=exp_lr_scheduler)
        
        parser.add_argument('--data_loaders', type=type(self.data_loaders), default=self.data_loaders)
        parser.add_argument('--data_sizes', type=type(self.data_sizes), default=self.data_sizes)
        parser.add_argument('--dataset_name', type=str, default=self.dataset_name)

        args, unknown = parser.parse_known_args()
        return args
