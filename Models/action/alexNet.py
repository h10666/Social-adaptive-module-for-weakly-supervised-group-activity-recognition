import torch.nn as nn
import torch
import torch.utils.model_zoo as model_zoo
import os

__all__ = ['AlexNet', 'alexNet']


model_urls = {
    'alexnet': 'https://download.pytorch.org/models/alexnet-owt-4df8aa71.pth',
}


class AlexNet(nn.Module):

    def __init__(self, model_confs, do_Classify=True):
        super(AlexNet_LSTM, self).__init__()
        self.do_LSTM = do_LSTM
        self.do_Classify = do_Classify
        self.T = model_confs.num_frames
        self.K = model_confs.num_players
        
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        
        self.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        feature_map = self.features(x)
        print feature_map.size()
        return feature_map


def alexNet(pretrained=False, dataset_name=None, **kwargs):
    r"""AlexNet model architecture from the
    `"One weird trick..." <https://arxiv.org/abs/1404.5997>`_ paper.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = AlexNet(**kwargs)
    if pretrained:
        model_dict = model.state_dict()
        pretrained_dict_file = './weights/'+dataset_name+'/action/alexnet-owt-4df8aa71.pth'
        if os.path.exists(pretrained_dict_file):
            pretrained_dict = torch.load(pretrained_dict_file)
        else:
            pretrained_dict = model_zoo.load_url(model_urls['alexnet'])
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k.split('.')[0] != 'classifier'}
        '''for k,v in pretrained_dict.items():
            print k.split('.')[0]'''
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)

    return model
