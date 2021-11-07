import torch
import torch.nn as nn
import torch.nn.functional as F
import random

from ..scheme_base import AudiosetTraining

class SwishNetV2(nn.Module):
    def __init__(self, 
                 input_features = 64, 
                 base_width     = 20, 
                 model_width    = 16,
                 dropout_rate   = 0.1):
        super().__init__()
        self.input_features = input_features
        self.base_width = base_width
        self.model_width = model_width
        self.dropout_rate = dropout_rate
        
        self.base = nn.Sequential(
            nn.ZeroPad2d((0,0,5,5)),
            nn.Conv2d(in_channels=1, 
                      out_channels=self.base_width, 
                      kernel_size=(7, self.input_features-4)),
            nn.MaxPool2d(kernel_size=5, stride=1)
        )
        
        self.sep_convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(in_channels=self.base_width*2,
                          groups=self.base_width*2, 
                          out_channels=self.base_width*4,
                          kernel_size=6,
                          bias=False),
                nn.Conv1d(in_channels=self.base_width*4,
                          out_channels=self.model_width*2,
                          kernel_size=1),
            ),
            nn.Sequential(
                nn.Conv1d(in_channels=self.model_width,
                          groups=self.model_width, 
                          out_channels=self.model_width*2,
                          kernel_size=6,
                          bias=False),
                nn.Conv1d(in_channels=self.model_width*2,
                          out_channels=self.model_width,
                          kernel_size=1),
            ),
            nn.Sequential(
                nn.Conv1d(in_channels=self.model_width//2,
                          groups=self.model_width//2, 
                          out_channels=self.model_width,
                          kernel_size=6,
                          bias=False),
                nn.Conv1d(in_channels=self.model_width,
                          out_channels=self.model_width,
                          kernel_size=1),
            ),
        ])
        
        self.convs = nn.ModuleList([
            nn.Conv1d(in_channels=self.model_width//2,
                      out_channels=self.model_width,
                      kernel_size=3,
                      dilation=2),
            nn.Conv1d(in_channels=self.model_width//2,
                      out_channels=self.model_width,
                      kernel_size=3,
                      dilation=4),
            nn.Conv1d(in_channels=self.model_width//2,
                      out_channels=self.model_width,
                      kernel_size=3,
                      dilation=8),
            nn.Conv1d(in_channels=self.model_width//2,
                      out_channels=self.model_width,
                      kernel_size=3,
                      dilation=16),
        ])
        self.final_conv = nn.Conv1d(in_channels=self.model_width//2,
                                    out_channels=self.model_width*2,
                                    kernel_size=3,
                                    dilation=32)
        
        self.final_linear = nn.Linear(self.model_width*3, 3)
    
    
    def _sig_tanh(self, x):
        x1, x2 = x.chunk(2, dim=1)
        return torch.sigmoid(x1) * torch.tanh(x2)
    
    def forward(self, inputs):
        x = inputs['log_mfb']
        x = 1. + x/16.
        
        x = x.unsqueeze(1)
        x = self.base(x)
        x = x.squeeze(3)
        
        x1, x2 = x.chunk(2, dim=2)
        x = torch.cat((x1, x2), dim=1)
        
        for layer in self.sep_convs:
            x = F.pad(x, (5,0))
            x = self._sig_tanh(layer(x))
        
        connections = []
        for i, layer in enumerate(self.convs):
            y = x
            x = F.pad(x, (4<<i,0))
            x = self._sig_tanh(layer(x))
            connections.append(x)
            x = x + y
        
        x = F.pad(x, (4<<len(self.convs),0))
        x = self._sig_tanh(self.final_conv(x))
        connections.append(x)
        
        x = torch.cat(connections, dim=1)
        
        if self.training:
            t_ind = random.randint(x.shape[2]//2, x.shape[2]-1)
            x = x[:,:,t_ind]
        else:
            x = x[:,:,-1]
        
        x = self.final_linear(x)
        x = F.log_softmax(x, dim=1)
        return x
        

class SwishNetV2Training(AudiosetTraining):
    def get_default_config(self):
        config = super().get_default_config()
        config.update(
            model_name   = 'swishnetv2',
            base_width   = 20,
        )
        return config
    
    def get_model_config(self):
        model_config, _ = super().get_model_config()
        model_config.update(
            base_width   = self.config.base_width,
        )
        return model_config, SwishNetV2
    

    
    
