import torch
import torchvision.models as models
import torchvision.models.swin_transformer as st
#import torchvision.models.swin_transformer.SwinTransformerBlockV2 as Swin_T
#from torchvision.models.swin_transformer import SwinTransformerBlockV2 as Swin_T
swin_t = st.SwinTransformerBlockV2(dim=3,num_heads=8,window_size=[8,8],shift_size=[4,4])
#swin_t = Swin_T(dim=3,num_heads=8,window_size=[8,8],shift_size=[4,4])
x = torch.randn(1, 3, 128, 128)
out = swin_t(x)
print(out.shape)
