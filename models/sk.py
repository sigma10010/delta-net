# Selective Kernel Networks

import torch
from torch import nn


class SKConv(nn.Module):
    def __init__(self, in_channel,out_channel, WH=None, M=2, G=1, r=4, stride=1 ,L=32):
        """ Constructor 
        Args:
            in_channel: input channel dimensionality.
            out_channel: output channel dimensionality.
            WH: input spatial dimensionality, used for GAP kernel size.
            M: the number of branchs.
            G: num of convolution groups.
            r: the radio for compute d, the length of z.
            stride: stride, default 1.
            L: the minimum dim of the vector z in paper, default 32.
        """
        super(SKConv, self).__init__()
        # d = max(int(out_channel/r), L)
        d = int(out_channel/r)
        self.M = M
        # self.features = features
        self.convs = nn.ModuleList([])
        for i in range(M):
            self.convs.append(nn.Sequential(
                nn.Conv2d(in_channel, out_channel, kernel_size=3+i*2, stride=stride, padding=1+i, groups=G),
                nn.BatchNorm2d(out_channel),
                nn.ReLU(inplace=False)
            ))
        # self.gap = nn.AvgPool2d(int(WH/stride))
        self.fc = nn.Linear(out_channel, d)
        self.fcs = nn.ModuleList([])
        for i in range(M):
            self.fcs.append(
                nn.Linear(d, out_channel)
            )
        self.softmax = nn.Softmax(dim=1)
        
    def forward(self, x):
        for i, conv in enumerate(self.convs):
            fea = conv(x).unsqueeze(dim=1)
            if i == 0:
                feas = fea
            else:
                feas = torch.cat([feas, fea], dim=1) # [b,M,features,h,w]
        fea_U = torch.sum(feas, dim=1) # [b,features,h,w]
        # fea_s = self.gap(fea_U).squeeze_()
        fea_s = fea_U.mean(-1).mean(-1) # [b,features]
        fea_z = self.fc(fea_s) # [b,d]
        for i, fc in enumerate(self.fcs):
            vector = fc(fea_z).unsqueeze_(dim=1)
            if i == 0:
                attention_vectors = vector
            else:
                attention_vectors = torch.cat([attention_vectors, vector], dim=1)
        attention_vectors = self.softmax(attention_vectors) # [b,M,features]
        attention_vectors = attention_vectors.unsqueeze(-1).unsqueeze(-1) # [b,M,features,h,w]
        fea_v = (feas * attention_vectors).sum(dim=1) # [b,features,h,w], M attention
        return fea_v


class SKUnit(nn.Module):
    def __init__(self, in_features, out_features, WH=None, M=2, G=1, r=4, stride=1, L=32, is_shortcut=True):
        super(SKUnit, self).__init__()
        self.feas = nn.Sequential(
            SKConv(in_features, out_features, WH=WH, M=M, G=G, r=r, stride=stride, L=L)
        )
        if in_features == out_features: # Identity Shortcut
            self.shortcut = nn.Sequential()
        else: # Projection Shortcut
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_features, out_features, 1, stride=stride),
                nn.BatchNorm2d(out_features)
            )
        self.is_shortcut = is_shortcut
    
    def forward(self, x):
        fea = self.feas(x)
        return fea + self.shortcut(x) if self.is_shortcut else fea


class SKNet(nn.Module):
    def __init__(self, class_num):
        super(SKNet, self).__init__()
        self.basic_conv = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.BatchNorm2d(64)
        ) # 32x32
        self.stage_1 = nn.Sequential(
            SKUnit(64, 256, 32, 2, 8, 2, stride=2),
            nn.ReLU(),
            SKUnit(256, 256, 32, 2, 8, 2),
            nn.ReLU(),
            SKUnit(256, 256, 32, 2, 8, 2),
            nn.ReLU()
        ) # 32x32
        self.stage_2 = nn.Sequential(
            SKUnit(256, 512, 32, 2, 8, 2, stride=2),
            nn.ReLU(),
            SKUnit(512, 512, 32, 2, 8, 2),
            nn.ReLU(),
            SKUnit(512, 512, 32, 2, 8, 2),
            nn.ReLU()
        ) # 16x16
        self.stage_3 = nn.Sequential(
            SKUnit(512, 1024, 32, 2, 8, 2, stride=2),
            nn.ReLU(),
            SKUnit(1024, 1024, 32, 2, 8, 2),
            nn.ReLU(),
            SKUnit(1024, 1024, 32, 2, 8, 2),
            nn.ReLU()
        ) # 8x8
        self.pool = nn.AvgPool2d(8)
        self.classifier = nn.Sequential(
            nn.Linear(1024, class_num),
            # nn.Softmax(dim=1)
        )

    def forward(self, x):
        fea = self.basic_conv(x)
        fea = self.stage_1(fea)
        fea = self.stage_2(fea)
        fea = self.stage_3(fea)
        fea = self.pool(fea)
        fea = torch.squeeze(fea)
        fea = self.classifier(fea)
        return fea


if __name__=='__main__':
    x = torch.rand(8, 32, 112, 112)
    conv = SKConv(32,30)
    out = conv(x)
    print('out shape : {}'.format(out.shape))