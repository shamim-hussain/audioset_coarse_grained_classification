
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..scheme_base import AudiosetTraining
from lib.training.training_mixins import ReduceLR

class VGG16(nn.Module):
    def __init__(self, 
                 input_features     = 64, 
                 num_feature_layers = -1,
                 linear_layers      = [64],
                 features_path      = 'data/vgg16_features.pth',
                 ):
        super(VGG16, self).__init__()
        self.input_features = input_features
        self.num_feature_layers = num_feature_layers
        self.features_path = features_path
        self.linear_layers = linear_layers
        
        vgg_features = torch.load(features_path)
        if num_feature_layers == -1:
            self.feature_layers = vgg_features
        else:
            self.feature_layers = vgg_features[:num_feature_layers]
        
        linear_dims = [512] + linear_layers + [3]
        
        self.classifier_layers = sum(([nn.Linear(linear_dims[i], linear_dims[i+1]),
                                       nn.ReLU(inplace=True)] 
                                       for i in range(len(linear_dims)-2)), [])
        self.classifier_layers.append(nn.Linear(linear_dims[-2], linear_dims[-1]))
        self.classifier_layers = nn.Sequential(*self.classifier_layers)
    
    def forward(self, inputs):
        x = inputs['log_mfb']
        x = (1. + x/16.).unsqueeze(1).expand(-1, 3, -1, -1)
        
        x = self.feature_layers(x)
        x = torch.mean(x, dim=[2,3])
        x = self.classifier_layers(x)
        
        x = F.log_softmax(x, dim=1)
        return x



class VGG16Training(ReduceLR,AudiosetTraining):
    def get_default_config(self):
        config = super().get_default_config()
        config.update(
            model_name         = 'vgg16',
            num_feature_layers = -1,
            linear_layers      = [64],
            features_path      = 'data/vgg16_features.pth',
        )
        return config
    
    def get_model_config(self):
        model_config, _ = super().get_model_config()
        model_config.update(
            num_feature_layers = self.config.num_feature_layers,
            linear_layers      = self.config.linear_layers,
            features_path      = self.config.features_path,
        )
        return model_config, VGG16

