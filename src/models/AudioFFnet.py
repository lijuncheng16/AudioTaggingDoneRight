import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy
import math
import numpy as np


import re
from scipy import linalg


class FFNetInput(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.layer_norm = nn.LayerNorm(config['input_size'], eps=config['layer_norm_eps'])
        self.hidden_mapping = nn.Linear(config['input_size'], config['hidden_size'])
        self.dropout = nn.Dropout(config['dropout_rate'])

    def forward(self, x):
        x = self.layer_norm(x)
        x = self.hidden_mapping(x)
        x = self.dropout(x)

        return x



class FourierFFTLayer(nn.Module):
    '''
    Default fft
    '''
    def __init__(self):
        super().__init__()

    @torch.cuda.amp.autocast(enabled=False)
    def forward(self, hidden_states, dim=-1):
        return torch.fft.fft(hidden_states.float(), dim=dim).real


class FFNetLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.fft =  FourierFFTLayer()
        self.mixing_layer_norm = nn.LayerNorm(config['hidden_size'], eps=config['layer_norm_eps'])
        self.feed_forward = nn.Linear(config['hidden_size'], config['intermediate_size'])
        self.output_dense = nn.Linear(config['intermediate_size'], config['hidden_size'])
        self.activation = nn.GELU()
        
        self.mixing_layer_norm2 = nn.LayerNorm(config['hidden_size'], eps=config['layer_norm_eps'])
        self.feed_forward2 = nn.Linear(config['hidden_size'], config['intermediate_size'])
        self.output_dense2 = nn.Linear(config['intermediate_size'], config['hidden_size'])
        self.activation2 = nn.GELU()
        
        
        self.output_layer_norm = nn.LayerNorm(config['hidden_size'], eps=config['layer_norm_eps'])
        self.dropout = nn.Dropout(config['dropout_rate'])
        

    def forward(self, hidden_states):
        fft_output = self.fft(hidden_states, dim=-1)
        fft_output = self.mixing_layer_norm(fft_output + hidden_states)
        intermediate_output = self.feed_forward(fft_output)
        intermediate_output = self.activation(intermediate_output)
        output = self.output_dense(intermediate_output)
        
        fft_output = self.fft(output, dim=-2)
        fft_output = self.mixing_layer_norm2(fft_output + hidden_states)
        intermediate_output = self.feed_forward2(fft_output)
        intermediate_output = self.activation2(intermediate_output)
        output = self.output_dense2(intermediate_output)
        
        output = self.dropout(output)
        output = self.output_layer_norm(output + fft_output)
        
        return output


class FFNetEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.layer = nn.ModuleList([FFNetLayer(config) for _ in range(config['num_hidden_layers'])])

    def forward(self, hidden_states):
        for i, layer_module in enumerate(self.layer):
            hidden_states = layer_module(hidden_states)

        return hidden_states


class FFNetPooler(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.att = nn.Linear(config['hidden_size'], num_classes)
        self.fc = nn.Linear(config['hidden_size'], num_classes)
        

    def forward(self, hidden_states):
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
#         first_token_tensor = hidden_states[:, 0]
#         pooled_output = self.dense(first_token_tensor)
#         pooled_output = self.activation(pooled_output)
        
        
        x = hidden_states
        frame_prob = torch.sigmoid(self.fc(x))
        frame_prob = torch.clamp(frame_prob, 1e-7, 1 - 1e-7)
        frame_att = torch.sigmoid(self.att(x))
        frame_att = torch.clamp(frame_att, 1e-7, 1 - 1e-7)
        frame_att = frame_att / frame_att.sum(dim=1).unsqueeze(1)
        global_prob = (frame_prob * frame_att).sum(dim=1) 
        #global_prob = torch.clamp(global_prob, 1e-7, 1 - 1e-7)
        return global_prob, None, None

        
        return pooled_output


class FFNet(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.input = FFNetInput(config)
        self.encoder = FFNetEncoder(config)
        self.pooler = FFNetPooler(config)

    def forward(self, x):
        input_output = self.input(x)
        sequence_output = self.encoder(input_output)
        pooled_output = self.pooler(sequence_output)

        return sequence_output, pooled_output





def get_default_config( fourier_type="fft",
                              layer_norm_eps=1e-12,
                              dropout_rate=0.1):
    return {
        "num_hidden_layers": 10,
        "input_size": 64,
        "hidden_size": 128,
        "intermediate_size": 256,
        "fourier": 'fft',
        "layer_norm_eps": layer_norm_eps,
        "dropout_rate": dropout_rate,
    }

def get_ffnet():
    config = get_default_config()
    return FFNet(config)