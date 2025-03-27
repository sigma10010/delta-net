import torch
import torch.nn as nn 
import numpy as np
import math
import copy
from models.resnet import resnet18
from models.itracker import FaceGridModel

def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

class TransformerEncoder(nn.Module):

    def __init__(self, encoder_layer, num_layers, norm=None):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, src, pos):
        output = src
        for layer in self.layers:
            output = layer(output, pos)

        if self.norm is not None:
            output = self.norm(output)

        return output


class TransformerEncoderLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=512, dropout=0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = nn.ReLU(inplace=True)

    def pos_embed(self, src, pos):
        batch_pos = pos.unsqueeze(1).repeat(1, src.size(1), 1)
        return src + batch_pos
        

    def forward(self, src, pos):
                # src_mask: Optional[Tensor] = None,
                # src_key_padding_mask: Optional[Tensor] = None):
                # pos: Optional[Tensor] = None):

        q = k = self.pos_embed(src, pos)
        src2 = self.self_attn(q, k, value=src)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)

        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src

class GazeTR(nn.Module):
    def __init__(self):
        super(GazeTR, self).__init__()
        maps = 32
        nhead = 8
        dim_feature = 7*7
        dim_feedforward=512
        dropout = 0.1
        num_layers=6

        self.base_model = resnet18(pretrained=False, maps=maps)
        self.gridModel = FaceGridModel()
        
        # Joining everything
        self.featureFC = nn.Sequential(
            #nn.Linear(128+64+128, 128),
            nn.Linear(32*3+16, 16),
            nn.ReLU(inplace=True),
            )

        # d_model: dim of Q, K, V 
        # nhead: seq num
        # dim_feedforward: dim of hidden linear layers
        # dropout: prob

        encoder_layer = TransformerEncoderLayer(
                  maps, 
                  nhead, 
                  dim_feedforward, 
                  dropout)

        encoder_norm = nn.LayerNorm(maps) 
        # num_encoder_layer: deeps of layers 

        self.encoder = TransformerEncoder(encoder_layer, num_layers, encoder_norm)

        self.cls_token = nn.Parameter(torch.randn(1, 1, maps))

        self.pos_embedding = nn.Embedding(dim_feature+1, maps)

        self.feed = nn.Linear(maps, 2)
        self.classifier = nn.Linear(maps, 4)
            
        self.loss_op = nn.L1Loss()

    def get_feature(self, im):
        feature = self.base_model(im)
        #print('base model feature',feature.shape)
        batch_size = feature.size(0)
        feature = feature.flatten(2)
        feature = feature.permute(2, 0, 1)
        
        cls = self.cls_token.repeat( (1, batch_size, 1))
        feature = torch.cat([cls, feature], 0)
        
        position = torch.from_numpy(np.arange(0, 50)).cuda()

        pos_feature = self.pos_embedding(position)

        # feature is [HW, batch, channel]
        feature = self.encoder(feature, pos_feature)
  
        feature = feature.permute(1, 2, 0)

        feature = feature[:,:,0] #(batch, channel) 32
        return feature
        
    def forward(self, query, calib):
        faces = query[0]
        eyesLeft = query[1]
        eyesRight = query[2]
        faceGrids = query[3]
        
        xEyeL = self.get_feature(eyesLeft)
        xEyeR = self.get_feature(eyesRight)

        # Face net
        xFace = self.get_feature(faces)
        xGrid = self.gridModel(faceGrids)

        # Cat all
        x = torch.cat((xEyeL, xEyeR, xFace, xGrid), 1)
        x_query = self.featureFC(x)
        
        # calib feature
        faces = calib[0]
        eyesLeft = calib[1]
        eyesRight = calib[2]
        faceGrids = calib[3]
        xEyeL = self.get_feature(eyesLeft)
        xEyeR = self.get_feature(eyesRight)

        # Face net
        xFace = self.get_feature(faces)
        xGrid = self.gridModel(faceGrids)

        # Cat all
        x = torch.cat((xEyeL, xEyeR, xFace, xGrid), 1)
        x_calib = self.featureFC(x)
        
        x = torch.cat((x_query, x_calib), 1)

        delta = self.feed(x)
        ori = self.classifier(x)
        
        return {'delta':delta, 'ori':ori, 'x_query':x_query, 'x_calib':x_calib}

    def loss(self, x_in, label):
        gaze = self.forward(x_in)
        loss = self.loss_op(gaze, label) 
        return loss

if __name__ == "__main__":
    print('test model')
#     model = GazeTR()
#     im =torch.randn(1,3,224,224)
#     query = (im, im, im, im)
#     model(query, query)
#     print('DONE')
