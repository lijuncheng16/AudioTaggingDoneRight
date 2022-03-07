import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy
import math
import numpy as np


import re
from scipy import linalg


class FNetInput(nn.Module):
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


class FourierMMLayer(nn.Module):
    '''
    Matmul to accelerate
    '''
    def __init__(self, config):
        super().__init__()

        self.dft_mat_seq = torch.tensor(linalg.dft(config['max_position_embeddings']))
        self.dft_mat_hidden = torch.tensor(linalg.dft(config['hidden_size']))

    def forward(self, hidden_states):
        hidden_states_complex = hidden_states.type(torch.complex128)
        return torch.einsum(
            "...ij,...jk,...ni->...nk",
            hidden_states_complex,
            self.dft_mat_hidden,
            self.dft_mat_seq
        ).real.type(torch.float32)


class FourierFFTLayer(nn.Module):
    '''
    Default fft
    '''
    def __init__(self):
        super().__init__()

    @torch.cuda.amp.autocast(enabled=False)
    def forward(self, hidden_states):
        return torch.fft.fft(torch.fft.fft(hidden_states.float(), dim=-1), dim=-2).real


class FNetLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.fft = FourierMMLayer(config) if config['fourier'] == 'matmul' else FourierFFTLayer()
        self.mixing_layer_norm = nn.LayerNorm(config['hidden_size'], eps=config['layer_norm_eps'])
        self.feed_forward = nn.Linear(config['hidden_size'], config['intermediate_size'])
        self.output_dense = nn.Linear(config['intermediate_size'], config['hidden_size'])
        self.output_layer_norm = nn.LayerNorm(config['hidden_size'], eps=config['layer_norm_eps'])
        self.dropout = nn.Dropout(config['dropout_rate'])
        self.activation = nn.GELU()

    def forward(self, hidden_states):
        fft_output = self.fft(hidden_states)
        fft_output = self.mixing_layer_norm(fft_output + hidden_states)
        intermediate_output = self.feed_forward(fft_output)
        intermediate_output = self.activation(intermediate_output)
        output = self.output_dense(intermediate_output)
        output = self.dropout(output)
        output = self.output_layer_norm(output + fft_output)
        return output


class FNetEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.layer = nn.ModuleList([FNetLayer(config) for _ in range(config['num_hidden_layers'])])

    def forward(self, hidden_states):
        for i, layer_module in enumerate(self.layer):
            hidden_states = layer_module(hidden_states)

        return hidden_states


class FNetPooler(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.att = nn.Linear(config['hidden_size'], config['num_classes'] )
        self.fc = nn.Linear(config['hidden_size'], config['num_classes'] )
        

    def forward(self, hidden_states):
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
#         first_token_tensor = hidden_states[:, 0]
#         pooled_output = self.dense(first_token_tensor)
#         pooled_output = self.activation(pooled_output)
        
        
        x = hidden_states
        frame_prob = self.fc(x)
        frame_prob = torch.clamp(frame_prob, 1e-7, 1 - 1e-7)
        frame_att = F.softmax(self.att(x),dim=1)
        frame_att = torch.clamp(frame_att, 1e-7, 1 - 1e-7)
        frame_att = frame_att / frame_att.sum(dim=1).unsqueeze(1)
        global_prob = (frame_prob * frame_att).sum(dim=1) 
        #global_prob = torch.clamp(global_prob, 1e-7, 1 - 1e-7)
        return global_prob, None, None

        
        return pooled_output


class FNet(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.input = FNetInput(config)
        self.encoder = FNetEncoder(config)
        self.pooler = FNetPooler(config)

    def forward(self, x):
        input_output = self.input(x)
        sequence_output = self.encoder(input_output)
        pooled_output = self.pooler(sequence_output)

        return pooled_output[0]


class FNetForPreTraining(nn.Module):
    def __init__(self, config):
        super(FNetForPreTraining, self).__init__()
        self.encoder = FNet(config)

        self.input_size = config['input_size']
        self.vocab_size = config['vocab_size']
        self.hidden_size = config['hidden_size']
        self.num_layers = config['num_hidden_layers']

        self.mlm_intermediate = nn.Linear(self.hidden_size, self.input_size)
        self.activation = nn.GELU()
        self.mlm_layer_norm = nn.LayerNorm(self.input_size)
        self.mlm_output = nn.Linear(self.input_size, self.vocab_size)

        self.nsp_output = nn.Linear(self.hidden_size, 2)

    def _mlm(self, x):
        x = self.mlm_intermediate(x)
        x = self.activation(x)
        x = self.mlm_layer_norm(x)
        x = self.mlm_output(x)
        return x

    def forward(self, input_ids, type_ids, mlm_positions=None):
        sequence_output, pooled_output = self.encoder(input_ids, type_ids)

        if mlm_positions is not None:
            mlm_input = sequence_output.take_along_dim(mlm_positions.unsqueeze(-1), dim=1)
        else:
            mlm_input = sequence_output

        mlm_logits = self._mlm(mlm_input)
        nsp_logits = self.nsp_output(pooled_output)
        return {"mlm_logits": mlm_logits, "nsp_logits": nsp_logits}


def get_default_config( fourier_type="fft",
                              layer_norm_eps=1e-12,
                              dropout_rate=0.1):
    return {
        "num_hidden_layers": 50,
        "input_size": 128,
        "hidden_size": 256,
        "intermediate_size": 512,
        "fourier": 'fft',
        "layer_norm_eps": layer_norm_eps,
        "dropout_rate": dropout_rate,
        "num_classes":527
    }


    
def get_config_from_statedict(state_dict,
                              fourier_type="fft",
                              pad_token_id=0,
                              layer_norm_eps=1e-12,
                              dropout_rate=0.1):
    is_pretraining_checkpoint = 'mlm_output.weight' in state_dict.keys()
    
    def prepare(key):
        if is_pretraining_checkpoint: 
            return f"encoder.{key}"
        return key

    regex = re.compile(prepare(r'encoder.layer.\d+.feed_forward.weight'))
    num_layers = len([key for key in state_dict.keys() if regex.search(key)])

    return {
        "num_hidden_layers": num_layers,
        "vocab_size": state_dict[prepare('embeddings.word_embeddings.weight')].shape[0],
        "embedding_size": state_dict[prepare('embeddings.word_embeddings.weight')].shape[1],
        "hidden_size": state_dict[prepare('encoder.layer.0.output_dense.weight')].shape[0],
        "intermediate_size": state_dict[prepare('encoder.layer.0.feed_forward.weight')].shape[0],
        "max_position_embeddings": state_dict[prepare('embeddings.position_embeddings.weight')].shape[0],
        "type_vocab_size": state_dict[prepare('embeddings.token_type_embeddings.weight')].shape[0],
        # the following parameters can not be inferred from the state dict and must be given manually
        "fourier": fourier_type,
        "pad_token_id": pad_token_id,
        "layer_norm_eps": layer_norm_eps,
        "dropout_rate": dropout_rate,
    }

def get_fnet():
    config = get_default_config()
    return FNet(config)