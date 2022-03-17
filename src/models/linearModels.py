import torch.nn as nn
import torch
import torch.nn.functional as F
from .HigherModels import *
from efficientnet_pytorch import EfficientNet
import torchvision

class LinearModel(nn.Module):
    def __init__(self, n_layers=3, input_dim=64, hidden_dim=128, label_dim=527):
        self.n_layers = n_layers
        self.hidden_dim = hidden_dim
        self.pooling = 'att'
        self.dropout = 0.1
        self.linear = nn.ModuleList()
        self.linear.append(nn.Linear(in_features=input_dim, out_features=hidden_dim, bias=True))
        for i in range(self.n_layers):
            self.linear.append(nn.LayerNorm(self.hidden_dim))
            self.linear.append(nn.Linear(self.hidden_dim, self.hidden_dim))
            
            
        self.fc_prob = nn.Linear(self.hidden_dim, self.label_dim)
        if self.pooling == 'att':
            self.fc_att = nn.Linear(self.hidden_dim, self.label_dim)

    def forward(self, x):
         for i in range(len(self.linear)):
            if self.dropout > 0: x = F.dropout(x, p = self.dropout, training = self.training)
            x = self.linear[i](x)          
        frame_prob = torch.sigmoid(self.fc_prob(x))
        frame_prob = torch.clamp(frame_prob, 1e-7, 1 - 1e-7)
        frame_att = F.softmax(self.fc_att(x), dim = 1)
        global_prob = (frame_prob * frame_att).sum(dim = 1)
#             return global_prob, frame_prob, frame_att
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
                #frame = output[1].cpu().numpy()
                #np.save('TALframe_516.npy', frame)
                if not verbose: output = output[:2]
                result.append([var.data.cpu().numpy() for var in output])
        result = tuple(numpy.concatenate(items) for items in zip(*result))
        #return result if verbose else result[0]
        if verbose:
            return result 
        return result[0], result[1]
    
# class ResNetAttention(nn.Module):
#     def __init__(self, label_dim=527, pretrain=True):
#         super(ResNetAttention, self).__init__()

#         self.model = torchvision.models.resnet50(pretrained=False)

#         if pretrain == False:
#             print('ResNet50 Model Trained from Scratch (ImageNet Pretraining NOT Used).')
#         else:
#             print('Now Use ImageNet Pretrained ResNet50 Model.')

#         self.model.conv1 = torch.nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)

#         # remove the original ImageNet classification layers to save space.
#         self.model.fc = torch.nn.Identity()
#         self.model.avgpool = torch.nn.Identity()

#         # attention pooling module
#         self.attention = Attention(
#             2048,
#             label_dim,
#             att_activation='sigmoid',
#             cla_activation='sigmoid')
#         self.avgpool = nn.AvgPool2d((4, 1))

#     def forward(self, x):
#         # expect input x = (batch_size, time_frame_num, frequency_bins), e.g., (12, 1024, 128)
#         x = x.unsqueeze(1)
#         x = x.transpose(2, 3)

#         batch_size = x.shape[0]
#         x = self.model(x)
#         x = x.reshape([batch_size, 2048, 4, 32])
#         x = self.avgpool(x)
#         x = x.transpose(2,3)
#         out, norm_att = self.attention(x)
#         return out

# class MBNet(nn.Module):
#     def __init__(self, label_dim=527, pretrain=True):
#         super(MBNet, self).__init__()

#         self.model = torchvision.models.mobilenet_v2(pretrained=pretrain)

#         self.model.features[0][0] = torch.nn.Conv2d(1, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
#         self.model.classifier = torch.nn.Linear(in_features=1280, out_features=label_dim, bias=True)

#     def forward(self, x, nframes):
#         # expect input x = (batch_size, time_frame_num, frequency_bins), e.g., (12, 1024, 128)
#         x = x.unsqueeze(1)
#         x = x.transpose(2, 3)

#         out = torch.sigmoid(self.model(x))
#         return out


# class EffNetAttention(nn.Module):
#     def __init__(self, label_dim=527, b=0, pretrain=True, head_num=4):
#         super(EffNetAttention, self).__init__()
#         self.middim = [1280, 1280, 1408, 1536, 1792, 2048, 2304, 2560]
#         if pretrain == False:
#             print('EfficientNet Model Trained from Scratch (ImageNet Pretraining NOT Used).')
#             self.effnet = EfficientNet.from_name('efficientnet-b'+str(b), in_channels=1)
#         else:
#             print('Now Use ImageNet Pretrained EfficientNet-B{:d} Model.'.format(b))
#             self.effnet = EfficientNet.from_pretrained('efficientnet-b'+str(b), in_channels=1)
#         # multi-head attention pooling
#         if head_num > 1:
#             print('Model with {:d} attention heads'.format(head_num))
#             self.attention = MHeadAttention(
#                 self.middim[b],
#                 label_dim,
#                 att_activation='sigmoid',
#                 cla_activation='sigmoid')
#         # single-head attention pooling
#         elif head_num == 1:
#             print('Model with single attention heads')
#             self.attention = Attention(
#                 self.middim[b],
#                 label_dim,
#                 att_activation='sigmoid',
#                 cla_activation='sigmoid')
#         # mean pooling (no attention)
#         elif head_num == 0:
#             print('Model with mean pooling (NO Attention Heads)')
#             self.attention = MeanPooling(
#                 self.middim[b],
#                 label_dim,
#                 att_activation='sigmoid',
#                 cla_activation='sigmoid')
#         else:
#             raise ValueError('Attention head must be integer >= 0, 0=mean pooling, 1=single-head attention, >1=multi-head attention.')

#         self.avgpool = nn.AvgPool2d((4, 1))
#         #remove the original ImageNet classification layers to save space.
#         self.effnet._fc = nn.Identity()

#     def forward(self, x, nframes=1056):
#         # expect input x = (batch_size, time_frame_num, frequency_bins), e.g., (12, 1024, 128)
#         x = x.unsqueeze(1)
#         x = x.transpose(2, 3)

#         x = self.effnet.extract_features(x)
#         x = self.avgpool(x)
#         x = x.transpose(2,3)
#         out, norm_att = self.attention(x)
#         return out

# if __name__ == '__main__':
#     input_tdim = 1056
#     #ast_mdl = ResNetNewFullAttention(pretrain=False)
#     psla_mdl = EffNetFullAttention(pretrain=False, b=0, head_num=0)
#     # input a batch of 10 spectrogram, each with 100 time frames and 128 frequency bins
#     test_input = torch.rand([10, input_tdim, 128])
#     test_output = psla_mdl(test_input)
#     # output should be in shape [10, 527], i.e., 10 samples, each with prediction of 527 classes.
#     print(test_output.shape)