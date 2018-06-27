#/usr/env/bin/python
# -*- coding:utf-8 -*-

import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo

__all__ = ['AlexNet', 'alexnet']


model_urls = {
    'alexnet': 'https://download.pytorch.org/models/alexnet-owt-4df8aa71.pth',
}


class AlexNet(nn.Module):
    def __init__(self, num_classes=1000):
        super(AlexNet, self).__init__()
        self.feature = nn.Sequential(
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
            nn.MaxPool2d(kernel_size=3, stride=2)
        )

        self.classifier = nn.Sequential(
            nn.Dropout(p=0.4),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes)
        )

    
    def forward(self, x):
        x = self.feature(x)
        x = x.view(-1, 256 * 6 * 6)
        x = self.classifier(x)
        return x



def alexnet(pretrained=False, model_root="", **kwargs):
    model = AlexNet(**kwargs)
    if(pretrained):
        # load pretrained model
        pretrained_model = AlexNet(100)
        pretrained_model.load_state_dict(torch.load(model_root + './models/alex_model_final.pkl'))
        # pretrained_model = torch.load(model_root + './models/alex_model_final.pkl')
        pretrained_dict = pretrained_model.state_dict()
        # print pretrained_dict

        # filter out unnecessary keys
        model_dict = model.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)

    return model