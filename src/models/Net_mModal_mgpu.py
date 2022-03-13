import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy
import math
import numpy as np
# from model import * # Poyao's model.py


class ConvBlock(nn.Module):
    def __init__(self, n_input_feature_maps, n_output_feature_maps, kernel_size, batch_norm = False, pool_stride = None):
        super(ConvBlock, self).__init__()
        assert all(x % 2 == 1 for x in kernel_size)
        self.n_input = n_input_feature_maps
        self.n_output = n_output_feature_maps
        self.kernel_size = kernel_size
        self.batch_norm = batch_norm
        self.pool_stride = pool_stride
        self.conv = nn.Conv2d(self.n_input, self.n_output, self.kernel_size, padding = tuple(int(x/2) for x in self.kernel_size), bias = ~batch_norm)
        if batch_norm: self.bn = nn.BatchNorm2d(self.n_output)
        # std = math.sqrt((4 * (1.0 - dropout)) / kernel_size[0] * n_input_feature_maps)
        # self.conv.weight.data.normal_(mean=0, std=std)
        # self.conv.bias.data.zero_()
        nn.init.xavier_uniform(self.conv.weight)

    def forward(self, x):
        x = self.conv(x)
        if self.batch_norm: x = self.bn(x)
        x = F.relu(x)
        if self.pool_stride is not None: x = F.max_pool2d(x, self.pool_stride)
        return x

class TALNet(nn.Module):
    def __init__(self, args):
        super(TALNet, self).__init__()
        self.__dict__.update(args.__dict__)     # Instill all args into self
#         print(self.n_conv_layers)
        assert self.n_conv_layers % self.n_pool_layers == 0
        self.input_n_freq_bins = n_freq_bins = args.n_mels
        self.output_size = 527
        self.conv = nn.ModuleList()
#         self.conv = []
        pool_interval = self.n_conv_layers / self.n_pool_layers
        n_input = 1
        for i in range(self.n_conv_layers):
            if (i + 1) % pool_interval == 0:        # this layer has pooling
                n_freq_bins = int(n_freq_bins / 2)
                n_output = int(self.embedding_size / n_freq_bins)
                pool_stride = (2, 2) if i < pool_interval * 2 else (1, 2)
            else:
                n_output = int(self.embedding_size * 2 / n_freq_bins)
                pool_stride = None
            layer = ConvBlock(n_input, n_output, self.kernel_size, batch_norm = self.batch_norm, pool_stride = pool_stride)
            self.conv.append(layer)
            self.__setattr__('conv' + str(i + 1), layer)
            n_input = n_output
        half_embedding_size = int(self.embedding_size/2)
        self.gru = nn.GRU(self.embedding_size, half_embedding_size, 1, batch_first = True, bidirectional = True)
        self.fc_prob = nn.Linear(self.embedding_size, self.output_size)
        if self.pooling == 'att':
            self.fc_att = nn.Linear(self.embedding_size, self.output_size)
        # Better initialization
        nn.init.orthogonal(self.gru.weight_ih_l0); nn.init.constant(self.gru.bias_ih_l0, 0)
        nn.init.orthogonal(self.gru.weight_hh_l0); nn.init.constant(self.gru.bias_hh_l0, 0)
        nn.init.orthogonal(self.gru.weight_ih_l0_reverse); nn.init.constant(self.gru.bias_ih_l0_reverse, 0)
        nn.init.orthogonal(self.gru.weight_hh_l0_reverse); nn.init.constant(self.gru.bias_hh_l0_reverse, 0)
        nn.init.xavier_uniform(self.fc_prob.weight); nn.init.constant(self.fc_prob.bias, 0)
        if self.pooling == 'att':
            nn.init.xavier_uniform(self.fc_att.weight); nn.init.constant(self.fc_att.bias, 0)

    def forward(self, x):
        #print('x shape:', x.shape)
        x = x.view((-1, 1, x.size(1), x.size(2)))                                                           # x becomes (batch, channel, time, freq)
        for i in range(len(self.conv)):
            if self.dropout > 0: x = F.dropout(x, p = self.dropout, training = self.training)
            x = self.conv[i](x)                                                                             # x becomes (batch, channel, time, freq)
        #print('x shape:', x.shape)
        x = x.permute(0, 2, 1, 3).contiguous()                                                              # x becomes (batch, time, channel, freq)
        #print('x shape:', x.shape)
        x = x.view((-1, x.size(1), x.size(2) * x.size(3)))                                                  # x becomes (batch, time, embedding_size)
        if self.dropout > 0: x = F.dropout(x, p = self.dropout, training = self.training)
        #print('x shape:', x.shape)
        x, _ = self.gru(x)                                                                                  # x becomes (batch, time, embedding_size)
        if self.dropout > 0: x = F.dropout(x, p = self.dropout, training = self.training)
        frame_prob = torch.sigmoid(self.fc_prob(x))                                                 # shape of frame_prob: (batch, time, output_size)
        frame_prob = torch.clamp(frame_prob, 1e-7, 1 - 1e-7)
        if self.pooling == 'max':
            global_prob, _ = frame_prob.max(dim = 1)
            return global_prob, frame_prob
        elif self.pooling == 'ave':
            global_prob = frame_prob.mean(dim = 1)
            return global_prob, frame_prob
        elif self.pooling == 'lin':
            global_prob = (frame_prob * frame_prob).sum(dim = 1) / frame_prob.sum(dim = 1)
            return global_prob, frame_prob
        elif self.pooling == 'exp':
            global_prob = (frame_prob * frame_prob.exp()).sum(dim = 1) / frame_prob.exp().sum(dim = 1)
            return global_prob, frame_prob
        elif self.pooling == 'att':
            frame_att = F.softmax(self.fc_att(x), dim = 1)
            global_prob = (frame_prob * frame_att).sum(dim = 1)
            return global_prob, frame_prob, frame_att

    def predict(self, x, verbose = True, batch_size = 100):
        # Predict in batches. Both input and output are numpy arrays.
        # If verbose == True, return all of global_prob, frame_prob and att
        # If verbose == False, only return global_prob
        result = []
        for i in range(0, len(x), batch_size):
            with torch.no_grad():
                input = Variable(torch.from_numpy(x[i : i + batch_size])).cuda()
                output = self.forward(input)
                #frame = output[1].cpu().numpy()
                #np.save('TALframe_516.npy', frame)
                if not verbose: output = output[:2]
                result.append([var.data.cpu().numpy() for var in output])
        result = tuple(numpy.concatenate(items) for items in zip(*result))
        #return result if verbose else result[0]
        if verbose:
            return result 
        return result[0], result[1]

class MultiHeadAttention(nn.Module):
    ''' Multi-Head Attention module '''

    def __init__(self, n_head, d_model, d_k, d_v, dropout=0.0):
        super(MultiHeadAttention, self).__init__()

        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        self.w_qs = nn.Linear(d_model, n_head * d_k)
        self.w_ks = nn.Linear(d_model, n_head * d_k)
        self.w_vs = nn.Linear(d_model, n_head * d_v)
        nn.init.normal_(self.w_qs.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_k)))
        nn.init.normal_(self.w_ks.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_k)))
        nn.init.normal_(self.w_vs.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_v)))
        # nn.init.xavier_uniform_(self.w_qs.weight)
        # nn.init.xavier_uniform_(self.w_ks.weight)
        # nn.init.xavier_uniform_(self.w_vs.weight)
        # nn.init.constant_(self.w_qs.bias, 0.)
        # nn.init.constant_(self.w_ks.bias, 0.)
        # nn.init.constant_(self.w_vs.bias, 0.)

        self.attention = ScaledDotProductAttention(temperature=np.power(d_k, 0.5),
                                                   attn_dropout=dropout)
        self.layer_norm = nn.LayerNorm(d_model)

        self.fc = nn.Linear(n_head * d_v, d_model)
        nn.init.xavier_normal_(self.fc.weight)
        #nn.init.constant_(self.fc.bias, 0.)

        self.dropout = nn.Dropout(dropout)


    def forward(self, q, k, v, mask=None):

        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head

        sz_b, len_q, _ = q.size()
        sz_b, len_k, _ = k.size()
        sz_b, len_v, _ = v.size()

        residual = q

        q = self.w_qs(q).view(sz_b, len_q, n_head, d_k)
        k = self.w_ks(k).view(sz_b, len_k, n_head, d_k)
        v = self.w_vs(v).view(sz_b, len_v, n_head, d_v)

        q = q.permute(2, 0, 1, 3).contiguous().view(-1, len_q, d_k) # (n*b) x lq x dk
        k = k.permute(2, 0, 1, 3).contiguous().view(-1, len_k, d_k) # (n*b) x lk x dk
        v = v.permute(2, 0, 1, 3).contiguous().view(-1, len_v, d_v) # (n*b) x lv x dv

        if mask is not None:
            mask = mask.repeat(n_head, 1, 1) # (n*b) x .. x ..

        output, attn = self.attention(q, k, v, mask=mask)

        output = output.view(n_head, sz_b, len_q, d_v)
        output = output.permute(1, 2, 0, 3).contiguous().view(sz_b, len_q, -1) # b x lq x (n*dv)

        output = self.dropout(self.fc(output))
        output = self.layer_norm(output + residual)

        return output, attn


class ScaledDotProductAttention(nn.Module):
    ''' Scaled Dot-Product Attention '''

    def __init__(self, temperature, attn_dropout=0.1):
        super(ScaledDotProductAttention, self).__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, q, k, v, mask=None):

        attn = torch.bmm(q, k.transpose(1, 2))
        attn = attn / self.temperature

        if mask is not None:
            attn = attn.masked_fill(mask, -np.inf)

        attn = self.softmax(attn)
        attn = self.dropout(attn)
        output = torch.bmm(attn, v)

        return output, attn

class PositionalEncoding(nn.Module):
    """Implement the positional encoding (PE) function.
    PE(pos, 2i)   = sin(pos/(10000^(2i/dmodel)))
    PE(pos, 2i+1) = cos(pos/(10000^(2i/dmodel)))
    """

    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model, requires_grad=False)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                             -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, input):
        """
        Args:
            input: N x T x D
        """
        length = input.size(1)
        return self.pe[:, :length]

class PositionwiseFeedForward(nn.Module):
    """Implements position-wise feedforward sublayer.
    FFN(x) = max(0, xW1 + b1)W2 + b2
    """

    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, x):
        residual = x
        output = self.w_2(F.relu(self.w_1(x)))
        output = self.dropout(output)
        output = self.layer_norm(output + residual)
        return output


class TransformerLayer(nn.Module):
    def __init__(self, hidden_size, dropout=0.1):
        super(TransformerLayer, self).__init__()
        # parameters
        #self.hidden_size = hidden_size
        #self.pe_maxlen = pe_maxlen

        #self.linear_in = nn.Linear(hidden_size, hidden_size)
        #self.layer_norm_in = nn.LayerNorm(hidden_size)
        #self.positional_encoding = PositionalEncoding(hidden_size, max_len=pe_maxlen)
        #self.dropout = nn.Dropout(dropout)

        self.slf_attn = MultiHeadAttention(
            8, hidden_size, hidden_size/8, hidden_size/8, dropout=dropout)
        self.pos_ffn = PositionwiseFeedForward(
             hidden_size,  hidden_size, dropout=dropout)

    def forward(self, hidden_states):
       
        #enc_output = self.dropout(
        #    self.layer_norm_in(self.linear_in(hidden_states)) +
        #    self.positional_encoding(hidden_states))
        #enc_output = hidden_states + self.positional_encoding(hidden_states)

        enc_output, enc_slf_attn = self.slf_attn(
            hidden_states, hidden_states, hidden_states)

        enc_output = self.pos_ffn(enc_output)
        
        return enc_output

class NewNet(nn.Module):
    def __init__(self, args):
        super(NewNet, self).__init__()
        self.__dict__.update(args.__dict__)     # Instill all args into self
        assert self.n_conv_layers % self.n_pool_layers == 0
        self.input_n_freq_bins = n_freq_bins = args.n_mels
        self.output_size = 527
#         self.conv = []
        self.conv = nn.ModuleList()
        pool_interval = self.n_conv_layers / self.n_pool_layers
        n_input = 1
        for i in range(self.n_conv_layers):
            if (i + 1) % pool_interval == 0:        # this layer has pooling
                n_freq_bins = int(n_freq_bins / 2)
                n_output = int(self.embedding_size / n_freq_bins)
                pool_stride = (2, 2) if i < pool_interval * 2 else (1, 2)
            else:
                n_output = int(self.embedding_size * 2 / n_freq_bins)
                pool_stride = None
            layer = ConvBlock(n_input, n_output, self.kernel_size, batch_norm = self.batch_norm, pool_stride = pool_stride)
            self.conv.append(layer)
            self.__setattr__('conv' + str(i + 1), layer)
            n_input = n_output
        half_embedding_size = int(self.embedding_size / 2)
        self.gru = nn.GRU(self.embedding_size, half_embedding_size, 1, batch_first = True, bidirectional = True)
        #self.position_embeddings = nn.Embedding(400, 64)
        #self.positional_encoding = PositionalEncoding(64, max_len=400)
        #self.self_att = BERTSelfAttention(self.embedding_size)
        #self.transformer = TransformerLayer(self.embedding_size, dropout=self.dropout)
        #self.layer_stack = nn.ModuleList([
        #    EncoderLayer(self.embedding_size, self.embedding_size*2, dropout=self.dropout)
        #    for _ in range(self.n_trans_layers)])
        self.fc_prob = nn.Linear(self.embedding_size, self.output_size)
        #self.proj_back = nn.Linear(self.embedding_size*2, self.embedding_size)
        if self.pooling == 'att' or self.pooling == 'all':
            self.fc_att = nn.Linear(self.embedding_size, self.output_size)
        # Better initialization
        nn.init.orthogonal(self.gru.weight_ih_l0); nn.init.constant(self.gru.bias_ih_l0, 0)
        nn.init.orthogonal(self.gru.weight_hh_l0); nn.init.constant(self.gru.bias_hh_l0, 0)
        nn.init.orthogonal(self.gru.weight_ih_l0_reverse); nn.init.constant(self.gru.bias_ih_l0_reverse, 0)
        nn.init.orthogonal(self.gru.weight_hh_l0_reverse); nn.init.constant(self.gru.bias_hh_l0_reverse, 0)
        nn.init.xavier_uniform(self.fc_prob.weight); nn.init.constant(self.fc_prob.bias, 0)
        #nn.init.xavier_uniform(self.proj_back.weight); nn.init.constant(self.proj_back.bias, 0)
        if self.pooling == 'att' or self.pooling == 'all':
            nn.init.xavier_uniform(self.fc_att.weight); nn.init.constant(self.fc_att.bias, 0)
        if self.pooling == 'h-att':
            self.stride_pool = nn.AvgPool1d(5)
        if self.pooling == 'all':
            self.ens = nn.Linear(5, 5)
            nn.init.xavier_uniform(self.ens.weight); nn.init.constant(self.ens.bias, 0)

    def forward(self, x):
        #batch_size, seq_length, _ = x.shape
        #position_ids = torch.arange(seq_length, dtype=torch.long, device=x.device)
        #position_ids = position_ids.unsqueeze(0).repeat(batch_size, 1)
        #position_embeddings = self.position_embeddings(position_ids)
        #x = x + position_embeddings
        #x = x + self.positional_encoding(x)
#         print('x shape:', x.shape)
        x = x.view((-1, 1, x.size(1), x.size(2)))                                                           # x becomes (batch, channel, time, freq)
        for i in range(len(self.conv)):
            if self.dropout > 0: x = F.dropout(x, p = self.dropout, training = self.training)
            x = self.conv[i](x)                                                                             # x becomes (batch, channel, time, freq)
#         print('x shape:', x.shape)

        x = x.permute(0, 2, 1, 3).contiguous()                                                              # x becomes (batch, time, channel, freq)
#         print('x shape:', x.shape)
        x = x.view((-1, x.size(1), x.size(2) * x.size(3)))                                                  # x becomes (batch, time, embedding_size)
#         print('x shape:', x.shape)

        if self.dropout > 0: x = F.dropout(x, p = self.dropout, training = self.training)
#         print('x shape:', x.shape)

        #x = self.transformer(x)      
        x, _ = self.gru(x)                                                                                  # x becomes (batch, time, embedding_size)
        if self.dropout > 0: x = F.dropout(x, p = self.dropout, training = self.training)
        frame_prob = torch.sigmoid(self.fc_prob(x))                                                             # shape of frame_prob: (batch, time, output_size)
        frame_prob = torch.clamp(frame_prob, 1e-7, 1 - 1e-7)
        if self.pooling == 'max':
            global_prob, _ = frame_prob.max(dim = 1)
            return global_prob, frame_prob
        elif self.pooling == 'ave':
            global_prob = frame_prob.mean(dim = 1)
            return global_prob, frame_prob
        elif self.pooling == 'lin':
            global_prob = (frame_prob * frame_prob).sum(dim = 1) / frame_prob.sum(dim = 1)
            return global_prob, frame_prob
        elif self.pooling == 'exp':
            global_prob = (frame_prob * frame_prob.exp()).sum(dim = 1) / frame_prob.exp().sum(dim = 1)
            return global_prob, frame_prob
        elif self.pooling == 'att':
            frame_att = F.softmax(self.fc_att(x), dim = 1)
            global_prob = (frame_prob * frame_att).sum(dim = 1)
            return global_prob, frame_prob, frame_att
        elif self.pooling == 'h-att':
            segment_prob = (frame_prob * frame_prob)
            frame_prob = frame_prob.permute(0, 2, 1)
            segment_prob = segment_prob.permute(0, 2, 1)
            xj = self.stride_pool(segment_prob)/self.stride_pool(frame_prob)
            wj = self.stride_pool(frame_prob)
            xj = xj.permute(0, 2, 1)
            wj = wj.permute(0, 2, 1)
            global_prob = (xj*wj).sum(dim=1)/wj.sum(dim=1)
            return global_prob, frame_prob.permute(0, 2, 1)
        elif self.pooling == 'all':
            max_prob, _ = frame_prob.max(dim = 1)
            max_prob = max_prob.unsqueeze(-1)
            ave_prob = frame_prob.mean(dim=1).unsqueeze(-1)
            lin_prob = (frame_prob * frame_prob).sum(dim = 1) / frame_prob.sum(dim = 1)
            lin_prob = lin_prob.unsqueeze(-1)
            exp_prob = (frame_prob * frame_prob.exp()).sum(dim = 1) / frame_prob.exp().sum(dim = 1)
            exp_prob = exp_prob.unsqueeze(-1)
            frame_att = F.softmax(self.fc_att(x), dim = 1)
            att_prob = (frame_prob * frame_att).sum(dim = 1)
            att_prob = att_prob.unsqueeze(-1)
            all_prob = torch.cat([max_prob, ave_prob, lin_prob, exp_prob, att_prob], dim=2)
            global_weights = F.softmax(self.ens(all_prob), dim=2)
            global_prob = (all_prob * global_weights).sum(dim=2)
            global_weights = global_weights.permute(0, 2, 1)
            return global_prob, frame_prob, frame_att, global_weights

    def predict(self, x, verbose = True, batch_size = 100):
        # Predict in batches. Both input and output are numpy arrays.
        # If verbose == True, return all of global_prob, frame_prob and att
        # If verbose == False, only return global_prob
        result = []
        for i in range(0, len(x), batch_size):
            with torch.no_grad():
                input = Variable(torch.from_numpy(x[i : i + batch_size])).cuda()
                output = self.forward(input)
                if not verbose: output = output[:1]
                result.append([var.data.cpu().numpy() for var in output])
        result = tuple(numpy.concatenate(items) for items in zip(*result))
        return result if verbose else result[0]


class EncoderLayer(nn.Module):
    """Compose with two sub-layers.
        1. A multi-head self-attention mechanism
        2. A simple, position-wise fully connected feed-forward network.
    """

    def __init__(self, d_model, d_inner, dropout=0.1):
        super(EncoderLayer, self).__init__()
        self.slf_attn = MultiHeadAttention(
            8, d_model, int(d_model/8), int(d_model/8), dropout=dropout)
        self.pos_ffn = PositionwiseFeedForward(
            d_model, d_inner, dropout=dropout)

    def forward(self, enc_input):
        enc_output, enc_slf_attn = self.slf_attn(
            enc_input, enc_input, enc_input)

        enc_output = self.pos_ffn(enc_output)

        return enc_output

class TransformerEncoder(nn.Module):
    """Encoder of Transformer including self-attention and feed forward.
    """

    def __init__(self, args):
        super(TransformerEncoder, self).__init__()
        self.__dict__.update(args.__dict__) 
        assert self.n_conv_layers % self.n_pool_layers == 0
        self.input_n_freq_bins = n_freq_bins = args.n_mels
        self.output_size = 527
#         self.conv = []
        self.conv = nn.ModuleList()
        pool_interval = int(self.n_conv_layers / self.n_pool_layers)
        n_input = 1
        for i in range(self.n_conv_layers):
            if (i + 1) % pool_interval == 0:        # this layer has pooling
                n_freq_bins = int(n_freq_bins/2)
                n_output = int(self.embedding_size / n_freq_bins)
                pool_stride = (2, 2) if i < pool_interval * 2 else (1, 2)
            else:
                n_output = int(self.embedding_size * 2 / n_freq_bins)
                pool_stride = None
            layer = ConvBlock(n_input, n_output, self.kernel_size, batch_norm = self.batch_norm, pool_stride = pool_stride)
            self.conv.append(layer)
            self.__setattr__('conv' + str(i + 1), layer)
            n_input = n_output
        # use linear transformation with layer norm to replace input embedding
        #self.linear_in = nn.Linear(self.input_n_freq_bins, self.embedding_size)
        #self.layer_norm_in = nn.LayerNorm(self.embedding_size)
        if self.addpos:
            self.positional_encoding = PositionalEncoding(args.n_mels, max_len=args.target_length)
        self.enc_layer = EncoderLayer(self.embedding_size, self.embedding_size, dropout=self.transformer_dropout)
        # self.layer_stack = nn.ModuleList([
        #     EncoderLayer(self.embedding_size, self.embedding_size, dropout=self.transformer_dropout)
        #     for _ in range(self.n_trans_layers)])
        self.fc_prob = nn.Linear(self.embedding_size, self.output_size)
        if self.pooling == 'att':
            self.fc_att = nn.Linear(self.embedding_size, self.output_size)
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        nn.init.xavier_uniform(self.fc_prob.weight); nn.init.constant(self.fc_prob.bias, 0)
        if self.pooling == 'att':
            nn.init.xavier_uniform(self.fc_att.weight); nn.init.constant(self.fc_att.bias, 0)

    def forward(self, x):
        """
        Args:
            padded_input: N x T x D   
            input_lengths: N
        Returns:
            enc_output: N x T x H
        """
        # Forward
        if self.addpos:
            x = x*8 + self.positional_encoding(x)
        x = x.view((-1, 1, x.size(1), x.size(2)))                                                           # x becomes (batch, channel, time, freq)
        for i in range(len(self.conv)):
            if self.dropout > 0: x = F.dropout(x, p = self.dropout, training = self.training)
            x = self.conv[i](x)                                                                             # x becomes (batch, channel, time, freq)
        x = x.permute(0, 2, 1, 3).contiguous()                                                              # x becomes (batch, time, channel, freq)
        x = x.view((-1, x.size(1), x.size(2) * x.size(3)))                                                  # x becomes (batch, time, embedding_size)
        if self.dropout > 0: x = F.dropout(x, p = self.dropout, training = self.training)
        for _ in range(self.n_trans_layers):
            x = self.enc_layer(x)
        #for enc_layer in self.layer_stack:
        #    x = enc_layer(x)
        if self.dropout > 0: 
            x_hat = F.dropout(x, p = self.dropout, training = self.training)
        else:
            x_hat = x
#         print(x_hat.shape)
        frame_prob = torch.sigmoid(self.fc_prob(x_hat))                                                             # shape of frame_prob: (batch, time, output_size)
        frame_prob = torch.clamp(frame_prob, 1e-7, 1 - 1e-7)
        if self.pooling == 'max':
            global_prob, _ = frame_prob.max(dim = 1)
#             return global_prob, frame_prob
            return global_prob
        elif self.pooling == 'ave':
            global_prob = frame_prob.mean(dim = 1)
#             return global_prob, frame_prob
            return global_prob
        elif self.pooling == 'lin':
            global_prob = (frame_prob * frame_prob).sum(dim = 1) / frame_prob.sum(dim = 1)
#             return global_prob, frame_prob
            return global_prob
        elif self.pooling == 'exp':
            global_prob = (frame_prob * frame_prob.exp()).sum(dim = 1) / frame_prob.exp().sum(dim = 1)
#             return global_prob, frame_prob
            return global_prob
        elif self.pooling == 'att':
            frame_att = F.softmax(self.fc_att(x_hat), dim = 1)
            global_prob = (frame_prob * frame_att).sum(dim = 1)
#             return global_prob, x, frame_prob, frame_att
            return global_prob


    def predict(self, x, verbose = True, batch_size = 100):
        # Predict in batches. Both input and output are numpy arrays.
        # If verbose == True, return all of global_prob, frame_prob and att
        # If verbose == False, only return global_prob
        result = []
        for i in range(0, len(x), batch_size):
            with torch.no_grad():
                input = Variable(torch.from_numpy(x[i : i + batch_size])).cuda()
                output = self.forward(input)
                # att = output[2].cpu().numpy()
                # np.save('TALtransatt_515.npy', att)
                # frame = output[1].cpu().numpy()
                # np.save('TALtransframe_515.npy', frame)
                # exit(0)
                if not verbose: output = output[:2]
                result.append([var.data.cpu().numpy() for var in output])
        result = tuple(numpy.concatenate(items) for items in zip(*result))
        if verbose:
            return result 
        return result[0], result[1]

class Transformer(nn.Module):
    """Encoder of Transformer including self-attention and feed forward.
    """

    def __init__(self, args):
        super(Transformer, self).__init__()
        self.__dict__.update(args.__dict__) 
        self.input_n_freq_bins = n_freq_bins = args.n_mels
        self.output_size = 527
        
        # use linear transformation with layer norm to replace input embedding
        self.linear_in = nn.Linear(self.input_n_freq_bins, self.embedding_size)
        self.layer_norm_in = nn.LayerNorm(self.embedding_size)
        #self.positional_encoding = PositionalEncoding(64, max_len=400)

        self.layer_stack = nn.ModuleList([
            EncoderLayer(self.embedding_size, self.embedding_size*2, dropout=self.dropout)
            for _ in range(self.n_trans_layers)])
        self.fc_prob = nn.Linear(self.embedding_size, self.output_size)
        if self.pooling == 'att':
            self.fc_att = nn.Linear(self.embedding_size, self.output_size)
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        nn.init.xavier_uniform(self.fc_prob.weight); nn.init.constant(self.fc_prob.bias, 0)
        if self.pooling == 'att':
            nn.init.xavier_uniform(self.fc_att.weight); nn.init.constant(self.fc_att.bias, 0)

    def forward(self, x):
        """
        Args:
            padded_input: N x T x D
            input_lengths: N
        Returns:
            enc_output: N x T x H
        """
        # Forward
                                                                                                             # x becomes (batch, time, channel, freq)
        x = self.layer_norm_in(self.linear_in(x))                                                            # x becomes (batch, time, embedding_size)
        if self.dropout > 0: x = F.dropout(x, p = self.dropout, training = self.training)

        for enc_layer in self.layer_stack:
            x = enc_layer(x)
        if self.dropout > 0: x = F.dropout(x, p = self.dropout, training = self.training)
        x = x.permute(0, 2, 1)
        x = F.max_pool1d(x, 4)
        x = x.permute(0, 2, 1)
        frame_prob = torch.sigmoid(self.fc_prob(x))                                                             # shape of frame_prob: (batch, time, output_size)
        frame_prob = torch.clamp(frame_prob, 1e-7, 1 - 1e-7)
        if self.pooling == 'max':
            global_prob, _ = frame_prob.max(dim = 1)
            return global_prob, frame_prob
        elif self.pooling == 'ave':
            global_prob = frame_prob.mean(dim = 1)
            return global_prob, frame_prob
        elif self.pooling == 'lin':
            global_prob = (frame_prob * frame_prob).sum(dim = 1) / frame_prob.sum(dim = 1)
            return global_prob, frame_prob
        elif self.pooling == 'exp':
            global_prob = (frame_prob * frame_prob.exp()).sum(dim = 1) / frame_prob.exp().sum(dim = 1)
            return global_prob, frame_prob
        elif self.pooling == 'att':
            frame_att = F.softmax(self.fc_att(x), dim = 1)
            global_prob = (frame_prob * frame_att).sum(dim = 1)
            return global_prob, frame_prob, frame_att


    def predict(self, x, verbose = True, batch_size = 100):
        # Predict in batches. Both input and output are numpy arrays.
        # If verbose == True, return all of global_prob, frame_prob and att
        # If verbose == False, only return global_prob
        result = []
        for i in range(0, len(x), batch_size):
            with torch.no_grad():
                input = Variable(torch.from_numpy(x[i : i + batch_size])).cuda()
                output = self.forward(input)
                if not verbose: output = output[:1]
                result.append([var.data.cpu().numpy() for var in output])
        result = tuple(numpy.concatenate(items) for items in zip(*result))
        return result if verbose else result[0]
    
class MMTEncoder(nn.Module):
    """Encoder of Transformer including self-attention and feed forward.
    """

    def __init__(self, args):
        super(MMTEncoder, self).__init__()
        self.__dict__.update(args.__dict__) 
        assert self.n_conv_layers % self.n_pool_layers == 0
        #self.fusion_module = fusion_module
        self.input_n_freq_bins = n_freq_bins = args.n_mels
        self.output_size = 527
        self.conv = nn.ModuleList()
        pool_interval = int(self.n_conv_layers / self.n_pool_layers)
        n_input = 1
        for i in range(self.n_conv_layers):
            if (i + 1) % pool_interval == 0:        # this layer has pooling
                n_freq_bins = int(n_freq_bins/2)
                n_output = int(self.embedding_size / n_freq_bins)
                pool_stride = (2, 2) if i < pool_interval * 2 else (1, 2)
            else:
                n_output = int(self.embedding_size * 2 / n_freq_bins)
                pool_stride = None
            layer = ConvBlock(n_input, n_output, self.kernel_size, batch_norm = self.batch_norm, pool_stride = pool_stride)
            self.conv.append(layer)
            self.__setattr__('conv' + str(i + 1), layer)
            n_input = n_output
        # use linear transformation with layer norm to replace input embedding
        #self.linear_in = nn.Linear(self.input_n_freq_bins, self.embedding_size)
        #self.layer_norm_in = nn.LayerNorm(self.embedding_size)
        if self.addpos:
            self.positional_encoding = PositionalEncoding(args.n_mels, max_len=args.target_length)
#         self.enc_layer = EncoderLayer(self.embedding_size, self.embedding_size, dropout=self.transformer_dropout)
        if self.fusion_module == 0:
            self.proj0 = nn.Linear(9216, self.embedding_size)
        if self.fusion_module == 1:
            self.proj1 = nn.Linear(12288, self.embedding_size)
        self.enc_layer = EncoderLayer(self.embedding_size, self.embedding_size, dropout=self.transformer_dropout)
        
        self.fc_prob = nn.Linear(self.embedding_size, self.output_size)
        if self.pooling == 'att':
            self.fc_att = nn.Linear(self.embedding_size, self.output_size)
#         self.fc_prob = nn.Linear(self.embedding_size, self.output_size)
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        nn.init.xavier_uniform(self.fc_prob.weight); nn.init.constant(self.fc_prob.bias, 0)
        if self.pooling == 'att':
            nn.init.xavier_uniform(self.fc_att.weight); nn.init.constant(self.fc_att.bias, 0)

    def forward(self, x1, x2):
        """
        Args:
            padded_input: N x T x D   
            input_lengths: N
            x1: N T1 D1    400 64
            x2: N T2 D2    10  2048 
        Returns:
            enc_output: N x T x H
        """
        # Forward
        if self.addpos:
            x1 = x1*8 + self.positional_encoding(x1)                           
        if self.fusion_module == 0:# direct fusion
            N, T1, D1 = x1.shape
            N, T2, D2 = x2.shape
            try: 
#                 x1 = x1.reshape((N, 1, T2, -1))                                                   # x becomes (batch, channel, time, freq)
#                 x2 = x2.reshape((N, 1, T2, -1))
                x1 = x1.reshape((N, 1, 80, -1))                                                   # x becomes (batch, channel, time, freq)
                x2 = x2.reshape((N, 1, 80, -1))
            except:
                print('x1:', x1.shape)
                print('x2:', x2.shape)
            #print('x1:', x1.shape)
            #print('x2:', x2.shape)
            x = torch.cat((x1, x2), dim = 3)
            #x = F.relu(self.proj0(x))
        else:
            N, T1, D1 = x1.shape
            x = x1.view((N, 1, T1, D1))   
        for i in range(len(self.conv)):
            if self.dropout > 0: x = F.dropout(x, p = self.dropout, training = self.training)
            x = self.conv[i](x)                                                                             # x becomes (batch, channel, time, freq)
        #print (x.shape)
        if self.fusion_module == 1: # fuse after conv before transform
            N, C, T1, D1 = x.shape
            N, T2, D2 = x2.shape 
            if T1 % T2 != 0: # padding
                temp = T1 % T2
                x = F.pad(input=x, pad=(0, 0, torch.floor(temp / 2), temp -  torch.floor(temp / 2)), mode='constant', value=0)
#             x = x.permute
#             x = x.view
#             x = x.permute
            x = x.permute(0, 2, 1, 3).contiguous() 
            x1 = x.view((N, 1, T2, -1)) # need to check if view work as thought                                               # x becomes (batch, channel, time, freq)
            x2 = x2.view((N, 1, T2, D2))
            x = torch.cat((x1, x2), dim = 3)                                                            # x becomes (batch, channel, time, freq)
            x = F.relu(self.proj1(x))
        
        x = x.permute(0, 2, 1, 3).contiguous()                                                              # x becomes (batch, time, channel, freq)
        x = x.view((-1, x.size(1), x.size(2) * x.size(3)))                                                  # x becomes (batch, time, embedding_size)
        if self.fusion_module == 0:
            x = F.relu(self.proj0(x))
        if self.dropout > 0: x = F.dropout(x, p = self.dropout, training = self.training)
        for _ in range(self.n_trans_layers):
            x = self.enc_layer(x)
        #for enc_layer in self.layer_stack:
        #    x = enc_layer(x)
        
        if self.fusion_module == 2: # fuse after transform before fully connected
            N, T1, D1 = x.shape
            N, T2, D2 = x2.shape 
            if T1 % T2 != 0: # padding
                temp = T1 % T2
                x = F.pad(input=x, pad=(0, 0, torch.floor(temp / 2), temp -  torch.floor(temp / 2)), mode='constant', value=0)
            x1 = x.view((N, T2, -1)) # need to check if view work as thought                                               
            x2 = x2.view((N, T2, D2))
            x = torch.cat((x1, x2), dim = 2)
            
        
        if self.dropout > 0: x = F.dropout(x, p = self.dropout, training = self.training)
        frame_prob = torch.sigmoid(self.fc_prob(x))                                                             # shape of frame_prob: (batch, time, output_size)
        frame_prob = torch.clamp(frame_prob, 1e-7, 1 - 1e-7)
        if self.pooling == 'max':
            global_prob, _ = frame_prob.max(dim = 1)
            return global_prob, frame_prob
        elif self.pooling == 'ave':
            global_prob = frame_prob.mean(dim = 1)
            return global_prob, frame_prob
        elif self.pooling == 'lin':
            global_prob = (frame_prob * frame_prob).sum(dim = 1) / frame_prob.sum(dim = 1)
            return global_prob, frame_prob
        elif self.pooling == 'exp':
            global_prob = (frame_prob * frame_prob.exp()).sum(dim = 1) / frame_prob.exp().sum(dim = 1)
            return global_prob, frame_prob
        elif self.pooling == 'att':
            frame_att = F.softmax(self.fc_att(x), dim = 1)
            global_prob = (frame_prob * frame_att).sum(dim = 1)
            return global_prob, frame_prob, frame_att


    def predict(self, x1, x2, verbose = True, batch_size = 100):
        # Predict in batches. Both input and output are numpy arrays.
        # If verbose == True, return all of global_prob, frame_prob and att
        # If verbose == False, only return global_prob
        result = []
        for i in range(0, len(x1), batch_size):
            with torch.no_grad():
                input1 = Variable(torch.from_numpy(x1[i : i + batch_size])).cuda()
                input2 = Variable(torch.from_numpy(x2[i : i + batch_size])).cuda()
                output = self.forward(input1, input2)
                # att = output[2].cpu().numpy()
                # np.save('TALtransatt_515.npy', att)
                # frame = output[1].cpu().numpy()
                # np.save('TALtransframe_515.npy', frame)
                # exit(0)
                if not verbose: output = output[:2]
                result.append([var.data.cpu().numpy() for var in output])
        result = tuple(numpy.concatenate(items) for items in zip(*result))
        if verbose:
            return result 
        return result[0], result[1] 
    

#TODO
class LateFusion(nn.Module):
    """late fusion model.
    """

    def __init__(self, args):
        super(LateFusion, self).__init__()
        self.__dict__.update(args.__dict__) 
        
        # I think that we should copy the whole branch model instead of using extracted feature
        
        
        self.output_size = 527
        self.embedding_size=1024
        self.n_feature=2048
        
        self.alpha = nn.Linear(self.embedding_size, self.embedding_size)
        self.beta = nn.Linear(self.embedding_size, self.embedding_size)
#         self.beta = nn.Linear(self.n_feature, self.embedding_size)
        
        self.fc_prob = nn.Linear(self.embedding_size, self.output_size)
        if self.pooling == 'att':
#             self.fc_att = nn.Linear(self.embedding_size, self.output_size)
            self.fc_att = nn.Linear(self.embedding_size, self.output_size)
#         for p in self.parameters():
#             if p.dim() > 1:
#                 nn.init.xavier_uniform_(p)
        nn.init.xavier_uniform(self.fc_prob.weight); nn.init.constant(self.fc_prob.bias, 0)
        if self.pooling == 'att':
            nn.init.xavier_uniform(self.fc_att.weight); nn.init.constant(self.fc_att.bias, 0)
            
        self.branch1 = TransformerEncoder(args)
        self.branch2 = videoModel(args)
    #     if not args.from_scratch:
    #         self.branch1.load_state_dict(torch.load('/home/kaixinm/kaixinm/workspace/audioset/TAL-trans-embed1024-10C5P-kernel3x3-bn-drop0.0-att-batch100-ckpt2500-adam-lr4e-04-pat3-fac0.8-seed15213-Trans2-weight-decay0.00000000-betas0.900-0.999repro/model/checkpoint19.pt')['model'])
    # #         self.branch2.load_state_dict(torch.load('/home/billyli/data_folder/workspace/audioset/VM-embed1024-10C5P-kernel3x3-bn-drop0.0-att-batch100-ckpt2500-adam-lr4e-04-pat3-fac0.8-seed15213-Trans2-weight-decay0.00000000-betas0.900-0.999late_fusion_branch2/model/checkpoint29.pt')['model'])
    #         self.branch2.load_state_dict(torch.load('/home/kaixinm/kaixinm/workspace/audioset/VM-embed1024-10C5P-kernel3x3-bn-drop0.0-att-batch100-ckpt2500-adam-lr4e-04-pat3-fac0.8-seed15213-Trans2-weight-decay0.00000000-betas0.900-0.999baseline/model/checkpoint30.pt')['model'])
    #         #self.branch1.train(False)  
    #         #self.branch2.train(False)  

    def forward(self, x1, x2):        
        # B*E and B*E, E should be 1024 for both
        x1 = self.branch1(x1)[1]
        x2 = self.branch2(x2)[1]
        x1 = x1.permute(0, 2, 1)
        x1 = F.avg_pool1d(x1, 10)
        x1 = x1.permute(0, 2, 1) 
#         print(x1.shape, x2.shape)
        x = self.alpha(x1) + self.beta(x2)
        if self.dropout > 0: 
            x_hat = F.dropout(x, p = self.dropout, training = self.training)
        else:
            x_hat = x
        frame_prob = torch.sigmoid(self.fc_prob(x_hat))                                   # shape of frame_prob: (batch, time, output_size)
        frame_prob = torch.clamp(frame_prob, 1e-7, 1 - 1e-7)
        if self.pooling == 'max':
            global_prob, _ = frame_prob.max(dim = 1)
            return global_prob, frame_prob
        elif self.pooling == 'ave':
            global_prob = frame_prob.mean(dim = 1)
            return global_prob, frame_prob
        elif self.pooling == 'lin':
            global_prob = (frame_prob * frame_prob).sum(dim = 1) / frame_prob.sum(dim = 1)
            return global_prob, frame_prob
        elif self.pooling == 'exp':
            global_prob = (frame_prob * frame_prob.exp()).sum(dim = 1) / frame_prob.exp().sum(dim = 1)
            return global_prob, frame_prob
        elif self.pooling == 'att':
            frame_att = F.softmax(self.fc_att(x_hat), dim = 1)
            global_prob = (frame_prob * frame_att).sum(dim = 1)
            return global_prob, x, frame_prob, frame_att

    def predict(self, x1, x2, verbose = True, batch_size = 100):
        # Predict in batches. Both input and output are numpy arrays.
        # If verbose == True, return all of global_prob, frame_prob and att
        # If verbose == False, only return global_prob
        result = []
        for i in range(0, len(x1), batch_size):
            with torch.no_grad():
                input1 = Variable(torch.from_numpy(x1[i : i + batch_size])).cuda()
                input2 = Variable(torch.from_numpy(x2[i : i + batch_size])).cuda()
                output = self.forward(input1, input2)
                # att = output[2].cpu().numpy()
                # np.save('TALtransatt_515.npy', att)
                # frame = output[1].cpu().numpy()
                # np.save('TALtransframe_515.npy', frame)
                # exit(0)
                if not verbose: output = output[:2]
                result.append([var.data.cpu().numpy() for var in output])
        result = tuple(numpy.concatenate(items) for items in zip(*result))
        if verbose:
            return result 
        return result[0], result[1]
    
class SuperLateFusion(nn.Module):
    """late fusion model.
    """

    def __init__(self, args):
        super(SuperLateFusion, self).__init__()
        self.__dict__.update(args.__dict__) 
        
        # I think that we should copy the whole branch model instead of using extracted feature
        
        
        self.output_size = 527
        self.embedding_size=1024
        self.n_feature=2048
        
        #self.alpha = nn.Linear(self.embedding_size, self.embedding_size)
        #self.beta = nn.Linear(self.embedding_size, self.embedding_size)
#         self.beta = nn.Linear(self.n_feature, self.embedding_size)
        
        #self.fc_prob = nn.Linear(self.embedding_size, self.output_size)
        #if self.pooling == 'att':
#             self.fc_att = nn.Linear(self.embedding_size, self.output_size)
        #    self.fc_att = nn.Linear(self.embedding_size, self.output_size)
#         for p in self.parameters():
#             if p.dim() > 1:
#                 nn.init.xavier_uniform_(p)
        #nn.init.xavier_uniform(self.fc_prob.weight); nn.init.constant(self.fc_prob.bias, 0)
        #if self.pooling == 'att':
        #    nn.init.xavier_uniform(self.fc_att.weight); nn.init.constant(self.fc_att.bias, 0)
            
        self.branch1 = TransformerEncoder(args)
        self.branch2 = videoModel(args)
        if not args.from_scratch:
            print('here!!!!!!')
            self.branch1.load_state_dict(torch.load('/home/kaixinm/kaixinm/workspace/audioset/TAL-trans-embed1024-10C5P-kernel3x3-bn-drop0.0-att-batch100-ckpt2500-adam-lr4e-04-pat3-fac0.8-seed15213-Trans2-weight-decay0.00000000-betas0.900-0.999shorter/model/checkpoint20.pt')['model'])
    #         self.branch2.load_state_dict(torch.load('/home/billyli/data_folder/workspace/audioset/VM-embed1024-10C5P-kernel3x3-bn-drop0.0-att-batch100-ckpt2500-adam-lr4e-04-pat3-fac0.8-seed15213-Trans2-weight-decay0.00000000-betas0.900-0.999late_fusion_branch2/model/checkpoint29.pt')['model'])
            self.branch2.load_state_dict(torch.load('/home/kaixinm/kaixinm/workspace/audioset/VM-embed1024-10C5P-kernel3x3-bn-drop0.0-att-batch100-ckpt2500-adam-lr4e-04-pat3-fac0.8-seed15213-Trans2-weight-decay0.00000000-betas0.900-0.999dropout0.5/model/checkpoint18.pt')['model'])
            #self.branch1.train(False)  
            #self.branch2.train(False)  

    def forward(self, x1, x2):        
        # B*E and B*E, E should be 1024 for both
        pred1, x1, _, _ = self.branch1(x1)
        pred2, x2, _, _ = self.branch2(x2)
        final_pred = pred1*0.6 + pred2*0.4
        return (final_pred, )

    def predict(self, x1, x2, verbose = True, batch_size = 100):
        # Predict in batches. Both input and output are numpy arrays.
        # If verbose == True, return all of global_prob, frame_prob and att
        # If verbose == False, only return global_prob
        result = []
        for i in range(0, len(x1), batch_size):
            with torch.no_grad():
                input1 = Variable(torch.from_numpy(x1[i : i + batch_size])).cuda()
                input2 = Variable(torch.from_numpy(x2[i : i + batch_size])).cuda()
                output = self.forward(input1, input2)
                # att = output[2].cpu().numpy()
                # np.save('TALtransatt_515.npy', att)
                # frame = output[1].cpu().numpy()
                # np.save('TALtransframe_515.npy', frame)
                # exit(0)
                #if not verbose: output = output[:2]
                result.append([var.data.cpu().numpy() for var in output])
        result = tuple(numpy.concatenate(items) for items in zip(*result))
        #if verbose:
        #    return result 
        return result[0], None
    
class videoModel(nn.Module):
    """Encoder of Transformer including self-attention and feed forward.
    """

    def __init__(self, args):
        super(videoModel, self).__init__()
        self.__dict__.update(args.__dict__) 
        
        self.output_size = 527
        self.embedding_size=1024
        self.n_feature=2048

        self.enc_layer = EncoderLayer(self.n_feature, self.embedding_size, dropout=self.transformer_dropout)
        self.hidden1 = nn.Linear(self.n_feature, self.embedding_size)
        self.fc_prob = nn.Linear(self.embedding_size, self.output_size)
        if self.pooling == 'att':
            self.fc_att = nn.Linear(self.embedding_size, self.output_size)
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        nn.init.xavier_uniform(self.fc_prob.weight); nn.init.constant(self.fc_prob.bias, 0)
        if self.pooling == 'att':
            nn.init.xavier_uniform(self.fc_att.weight); nn.init.constant(self.fc_att.bias, 0)

    def forward(self, x):     
        # x: BxTxE
        b,t,e = x.shape
        for _ in range(self.n_trans_layers):
            x = self.enc_layer(x)
        x = F.relu(self.hidden1(x))
        if self.dropout > 0: 
            x_hat = F.dropout(x, p = self.dropout, training = self.training)
        else:
            x_hat = x
        frame_prob = torch.sigmoid(self.fc_prob(x_hat))                                   # shape of frame_prob: (batch, time, output_size)
        frame_prob = torch.clamp(frame_prob, 1e-7, 1 - 1e-7)
        if self.pooling == 'max':
            global_prob, _ = frame_prob.max(dim = 1)
            return global_prob, frame_prob
        elif self.pooling == 'ave':
            global_prob = frame_prob.mean(dim = 1)
            return global_prob, frame_prob
        elif self.pooling == 'lin':
            global_prob = (frame_prob * frame_prob).sum(dim = 1) / frame_prob.sum(dim = 1)
            return global_prob, frame_prob
        elif self.pooling == 'exp':
            global_prob = (frame_prob * frame_prob.exp()).sum(dim = 1) / frame_prob.exp().sum(dim = 1)
            return global_prob, frame_prob
        elif self.pooling == 'att':
            frame_att = F.softmax(self.fc_att(x_hat), dim = 1)
            global_prob = (frame_prob * frame_att).sum(dim = 1)
            return global_prob, x, frame_prob, frame_att

    
    def predict(self, x1, verbose = True, batch_size = 100):
        # Predict in batches. Both input and output are numpy arrays.
        # If verbose == True, return all of global_prob, frame_prob and att
        # If verbose == False, only return global_prob
        result = []
        for i in range(0, len(x1), batch_size):
            with torch.no_grad():
                input = Variable(torch.from_numpy(x1[i : i + batch_size])).cuda()
                output = self.forward(input)
                # att = output[2].cpu().numpy()
                # np.save('TALtransatt_515.npy', att)
                # frame = output[1].cpu().numpy()
                # np.save('TALtransframe_515.npy', frame)
                # exit(0)
                if not verbose: output = output[:2]
                result.append([var.data.cpu().numpy() for var in output])
        result = tuple(numpy.concatenate(items) for items in zip(*result))
        if verbose:
            return result 
        return result[0], result[1]
