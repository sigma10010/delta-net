import torch
import torch.nn as nn
import torch.nn.functional as F

# from models.safs import SAFS, SAFF, SAFS_X
from models.itracker import FaceGridModel

# Basic building blocks

class MLP(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.dropout(self.fc2(self.act(self.fc1(x))))

class WindowAttention(nn.Module):
    def __init__(self, dim, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.scale = (dim // num_heads) ** -0.5
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        return self.proj(x)

class SwinBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., dropout=0.):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = WindowAttention(dim, num_heads)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLP(dim, int(dim * mlp_ratio), dropout)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x

class PatchEmbed(nn.Module):
    def __init__(self, patch_size=4, in_channels=3, embed_dim=64):
        super().__init__()
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.proj(x)  # (B, C, H/patch, W/patch)
        _, _, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)  # (B, N, C)
        return x, H, W

class BasicLayer(nn.Module):
    def __init__(self, dim, depth, num_heads):
        super().__init__()
        self.blocks = nn.Sequential(*[SwinBlock(dim, num_heads) for _ in range(depth)])

    def forward(self, x):
        return self.blocks(x)

class PatchMerging(nn.Module):
    def __init__(self, input_resolution, dim):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = nn.LayerNorm(4 * dim)

    def forward(self, x):
        B, L, C = x.shape
        H = W = int(L ** 0.5)
        assert H * W == L, "Input feature length is not a perfect square"
        x = x.view(B, H, W, C)

        x0 = x[:, 0::2, 0::2, :]
        x1 = x[:, 1::2, 0::2, :]
        x2 = x[:, 0::2, 1::2, :]
        x3 = x[:, 1::2, 1::2, :]
        x = torch.cat([x0, x1, x2, x3], -1)
        x = x.view(B, -1, 4 * C)
        x = self.norm(x)
        x = self.reduction(x)
        return x

class SwinEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.stage_dims = [16, 32, 64, 128]
        self.num_heads = [1, 2, 4, 8]
        self.depths = [1, 1, 1, 1]

        self.patch_embed = PatchEmbed(patch_size=4, in_channels=3, embed_dim=self.stage_dims[0])
        self.pos_drop = nn.Dropout(0.)

        self.stage1 = BasicLayer(self.stage_dims[0], self.depths[0], self.num_heads[0])
        self.merge1 = PatchMerging((56, 56), self.stage_dims[0])

        self.stage2 = BasicLayer(self.stage_dims[1], self.depths[1], self.num_heads[1])
        self.merge2 = PatchMerging((28, 28), self.stage_dims[1])

        self.stage3 = BasicLayer(self.stage_dims[2], self.depths[2], self.num_heads[2])
        self.merge3 = PatchMerging((14, 14), self.stage_dims[2])

        self.stage4 = BasicLayer(self.stage_dims[3], self.depths[3], self.num_heads[3])

    def forward(self, x):
        B = x.size(0)
        x, H, W = self.patch_embed(x)
        x = self.pos_drop(x)

        x1 = self.stage1(x)
        x2 = self.stage2(self.merge1(x1))
        x3 = self.stage3(self.merge2(x2))
        x4 = self.stage4(self.merge3(x3))

        feats = []
        for feat, c in zip([x1, x2, x3, x4], self.stage_dims):
            B, N, _ = feat.shape
            S = int(N ** 0.5)
            assert S * S == N, f"Feature length {N} is not a perfect square"
            f = feat.transpose(1, 2).reshape(B, c, S, S)
            feats.append(f)


        return feats

class SwinGaze(nn.Module):
    def __init__(self, network_depth=4, is_scale_adaptive=True, n_scales=2):
        super(SwinGaze, self).__init__()
        self.swin = SwinEncoder()

        # self.is_scale_adaptive = is_scale_adaptive
        # self.n_scales = n_scales # [1, network_depth-1]
        # self.featureBlocks = nn.ModuleList([])
        # size0 = 7
        # for i in range(network_depth):
        #     size = size0*2**(network_depth-1-i)
        #     self.featureBlocks.append(nn.Linear(size*size*32*2**i, 128))
        # # self.ss = SAFS_X(M=2, ch_in=128, r=8)

        self.fc = nn.Sequential(
            nn.Linear(7*7*128, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 32),
            nn.ReLU(inplace=True),
            )

        self.gridModel = FaceGridModel()
        # Joining everything
        self.featureFC = nn.Sequential(
            nn.Linear(32*3+16, 16),
            nn.ReLU(inplace=True),
            )
        self.feed = nn.Linear(16, 2)

    def get_feature(self, x):
        x = self.swin(x)[-1]
        # print(f"Feature shape: {x.shape}")
        x = x.reshape(x.size(0), -1)
        x = self.fc(x)
        return x

    def forward(self, query):
        faces = query[0]
        eyesLeft = query[1]
        eyesRight = query[2]
        faceGrids = query[3]

        xEyeL = self.get_feature(eyesLeft)
        del eyesLeft
        torch.cuda.empty_cache()

        xEyeR = self.get_feature(eyesRight)
        del eyesRight
        torch.cuda.empty_cache()

        xFace = self.get_feature(faces)
        del faces
        torch.cuda.empty_cache()

        xGrid = self.gridModel(faceGrids)

        x = torch.cat((xEyeL, xEyeR, xFace, xGrid), 1)
        x_query = self.featureFC(x)
        gaze = self.feed(x_query)
        return gaze


# Test
if __name__ == "__main__":
    model = SwinEncoder()
    x = torch.randn(1, 3, 224, 224)
    feats = model(x)
    for i, f in enumerate(feats):
        print(f"Stage {i}: shape = {f.shape}")
