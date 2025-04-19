# ------------------------------------------------------------------------
# Copyright (c) 2022 megvii-model. All Rights Reserved.
# ------------------------------------------------------------------------

'''
Simple Baselines for Image Restoration

@article{chen2022simple,
  title={Simple Baselines for Image Restoration},
  author={Chen, Liangyu and Chu, Xiaojie and Zhang, Xiangyu and Sun, Jian},
  journal={arXiv preprint arXiv:2204.04676},
  year={2022}
}
'''
import os
import sys
sys.path.append('/data1/chenxiao/NAFNet')
import torch
import torch.nn as nn
import torch.nn.functional as F
from basicsr.models.archs.arch_util import LayerNorm2d
from basicsr.models.archs.local_arch import Local_Base, Detect_Base
from timm.models.vision_transformer import PatchEmbed, Block, Mlp
from einops import rearrange
from torch import nn, einsum
from collections import OrderedDict

class SimpleGate(nn.Module):
    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2

class NAFBlock(nn.Module):
    def __init__(self, c, DW_Expand=2, FFN_Expand=2, drop_out_rate=0.):
        super().__init__()
        dw_channel = c * DW_Expand
        self.conv1 = nn.Conv2d(in_channels=c, out_channels=dw_channel, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        self.conv2 = nn.Conv2d(in_channels=dw_channel, out_channels=dw_channel, kernel_size=3, padding=1, stride=1, groups=dw_channel,
                               bias=True)
        self.conv3 = nn.Conv2d(in_channels=dw_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        
        # Simplified Channel Attention
        self.sca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels=dw_channel // 2, out_channels=dw_channel // 2, kernel_size=1, padding=0, stride=1,
                      groups=1, bias=True),
        )

        # SimpleGate
        self.sg = SimpleGate()

        ffn_channel = FFN_Expand * c
        self.conv4 = nn.Conv2d(in_channels=c, out_channels=ffn_channel, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        self.conv5 = nn.Conv2d(in_channels=ffn_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1, groups=1, bias=True)

        self.norm1 = LayerNorm2d(c)
        self.norm2 = LayerNorm2d(c)

        self.dropout1 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
        self.dropout2 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()

        self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)

    def forward(self, inp):
        x = inp

        x = self.norm1(x)

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.sg(x)
        x = x * self.sca(x)
        x = self.conv3(x)

        x = self.dropout1(x)

        y = inp + x * self.beta

        x = self.conv4(self.norm2(y))
        x = self.sg(x)
        x = self.conv5(x)

        x = self.dropout2(x)

        return y + x * self.gamma


class PreNorm(nn.Module):
    def __init__(self, dim, fn, context_dim = None):
        super().__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(dim)
        self.norm_context = nn.LayerNorm(context_dim) if exists(context_dim) else None

    def forward(self, x, **kwargs):
        x = self.norm(x)

        if exists(self.norm_context):
            context = kwargs['context']
            normed_context = self.norm_context(context)
            kwargs.update(context = normed_context)

        return self.fn(x, **kwargs)

class GEGLU(nn.Module):
    def forward(self, x):
        x, gates = x.chunk(2, dim = -1)
        return x * F.gelu(gates)

class FeedForward(nn.Module):
    def __init__(self, dim, mult = 4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim * mult * 2),
            GEGLU(),
            nn.Linear(dim * mult, dim)
        )

    def forward(self, x):
        return self.net(x)

def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

class Attention(nn.Module):
    def __init__(self, query_dim, context_dim = None, heads = 8, dim_head = 64):
        super().__init__()
        inner_dim = dim_head * heads
        context_dim = default(context_dim, query_dim)
        self.scale = dim_head ** -0.5
        self.heads = heads

        self.to_q = nn.Linear(query_dim, inner_dim, bias = False)
        self.to_kv = nn.Linear(context_dim, inner_dim * 2, bias = False)
        self.to_out = nn.Linear(inner_dim, query_dim)

    def forward(self, x, context = None, mask = None):
        h = self.heads

        q = self.to_q(x)
        context = default(context, x)
        k, v = self.to_kv(context).chunk(2, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h = h), (q, k, v))

        sim = einsum('b i d, b j d -> b i j', q, k) * self.scale

        if exists(mask):
            mask = rearrange(mask, 'b ... -> b (...)')
            max_neg_value = -torch.finfo(sim.dtype).max
            mask = repeat(mask, 'b j -> (b h) () j', h = h)
            sim.masked_fill_(~mask, max_neg_value)

        # attention, what we cannot get enough of
        attn = sim.softmax(dim = -1)

        out = einsum('b i j, b j d -> b i d', attn, v)
        out = rearrange(out, '(b h) n d -> b n (h d)', h = h)
        return self.to_out(out)

class ChannelWiseCrossAttention(nn.Module):
    def __init__(self, in_channels):
        super(ChannelWiseCrossAttention, self).__init__()
        self.query_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))  # Learnable scalar to scale the attention output
    
    def forward(self, x1, x2):
        # Compute query, key, and value features
        query = self.query_conv(x1)  # [B, C, H, W]
        key = self.key_conv(x2)      # [B, C, H, W]
        value = self.value_conv(x2)  # [B, C, H, W]

        # Channel-wise cross attention (reshape to [B, C, H*W])
        B, C, H, W = query.size()
        query = query.view(B, C, -1)  # [B, C, H*W]
        key = key.view(B, C, -1)      # [B, C, H*W]
        value = value.view(B, C, -1)  # [B, C, H*W]

        # Compute attention scores
        attention = torch.bmm(query.permute(0, 2, 1), key)  # [B, H*W, H*W]
        attention = F.softmax(attention, dim=-1)  # Normalize attention scores

        # Apply attention to value
        out = torch.bmm(value, attention.permute(0, 2, 1))  # [B, C, H*W]
        out = out.view(B, C, H, W)  # Reshape back to [B, C, H, W]

        # Combine with input feature map
        out = self.gamma * out + x1  # Weighted residual connection
        return out

class CrossAttentionWithFFN(nn.Module):
    def __init__(self, embed_dim, heads=8, ffn_dim=256, dropout=0.1):
        super(CrossAttentionWithFFN, self).__init__()
        self.embed_dim = embed_dim
        self.head_dim = embed_dim // heads
        self.heads = heads
        self.scale = embed_dim ** -0.5  

        self.query_proj = nn.Linear(embed_dim, embed_dim)
        self.key_proj = nn.Linear(embed_dim, embed_dim)
        self.value_proj = nn.Linear(embed_dim, embed_dim)

        self.attention_dropout = nn.Dropout(dropout)
        self.output_proj = nn.Linear(embed_dim, embed_dim)

        self.norm1 = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, ffn_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, embed_dim),
        )
        self.norm2 = nn.LayerNorm(embed_dim)
        self.ffn_dropout = nn.Dropout(dropout)

    def forward(self, feature_map, mask):
        B, C, H, W = feature_map.shape

        features = feature_map.flatten(2).permute(0, 2, 1)  # (B, H*W, C)
        mask = mask.flatten(2).permute(0, 2, 1)  # (B, H*W, 1)

        anchor = features * mask
        queries = self.query_proj(anchor).view(B, -1, self.heads, self.head_dim).permute(0, 2, 1, 3)  
        keys = self.key_proj(features).view(B, -1, self.heads, self.head_dim).permute(0, 2, 1, 3)  
        values = self.value_proj(features).view(B, -1, self.heads, self.head_dim).permute(0, 2, 1, 3)  

        # Compute attention scores
        attention_scores = torch.matmul(queries, keys.transpose(-2, -1))  
        attention_scores = attention_scores * self.scale
        attention_weights = F.softmax(attention_scores, dim=-1)  
        attention_weights = self.attention_dropout(attention_weights)

        # Apply attention weights to values
        attended_features = torch.matmul(attention_weights, values)  

        attended_features = attended_features.permute(0, 2, 1, 3).contiguous().view(B, -1, self.embed_dim) 
        attended_features = self.output_proj(attended_features)  
        attention_output = self.norm1(attended_features + features)  

        ffn_input = attention_output 
        ffn_output = self.ffn(ffn_input)  
        ffn_output = self.ffn_dropout(ffn_output)
        ffn_output = ffn_output + ffn_input  
        ffn_output = self.norm2(ffn_output)  
      
        final_output = ffn_output.permute(0, 2, 1).view(B, C, H, W) 

        return final_output
    
class DualStreamBlock(nn.Module):
    def __init__(self, *args):
        super(DualStreamBlock, self).__init__()
        self.seq = nn.Sequential()

        if len(args) == 1 and isinstance(args[0], OrderedDict):
            for key, module in args[0].items():
                self.seq.add_module(key, module)
        else:
            for idx, module in enumerate(args):
                self.seq.add_module(str(idx), module)

    def forward(self, x, y):
        return self.seq(x), self.seq(y)

class CABlock(nn.Module):
    def __init__(self, channels):
        super(CABlock, self).__init__()
        self.ca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels, 1)
        )

    def forward(self, x):
        return x * self.ca(x)


class DualStreamGate(nn.Module):
    def forward(self, x, y):
        x1, x2 = x.chunk(2, dim=1)
        y1, y2 = y.chunk(2, dim=1)
        return x1 * y2, y1 * x2


class DualStreamSeq(nn.Sequential):
    def forward(self, x, y=None):
        y = y if y is not None else x
        for module in self:
            x, y = module(x, y)
        return x, y
    
class MuGIBlock(nn.Module):
    def __init__(self, c, shared_b=False):
        super().__init__()
        self.block1 = DualStreamSeq(
            DualStreamBlock(
                LayerNorm2d(c),
                nn.Conv2d(c, c * 2, 1),
                nn.Conv2d(c * 2, c * 2, 3, padding=1, groups=c * 2)
            ),
            DualStreamGate(),
            DualStreamBlock(CABlock(c)),
            DualStreamBlock(nn.Conv2d(c, c, 1))
        )

        self.a_l = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        self.a_r = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)

        self.block2 = DualStreamSeq(
            DualStreamBlock(
                LayerNorm2d(c),
                nn.Conv2d(c, c * 2, 1)
            ),
            DualStreamGate(),
            DualStreamBlock(
                nn.Conv2d(c, c, 1)
            )

        )

        self.shared_b = shared_b
        if shared_b:
            self.b = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        else:
            self.b_l = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
            self.b_r = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
    
    def forward(self, inp_l, inp_r):
        x, y = self.block1(inp_l, inp_r)
        x_skip, y_skip = inp_l + x * self.a_l, inp_r + y * self.a_r
        x, y = self.block2(x_skip, y_skip)
        if self.shared_b:
            out_l, out_r = x_skip + x * self.b, y_skip + y * self.b
        else:
            out_l, out_r = x_skip + x * self.b_l, y_skip + y * self.b_r
        return out_l, out_r

class NAFNet(nn.Module):

    def __init__(self, img_channel=3, out_channel=3, width=16, middle_blk_num=1, enc_blk_nums=[], dec_blk_nums=[]):
        super().__init__()

        self.intro = nn.Conv2d(in_channels=img_channel, out_channels=width, kernel_size=3, padding=1, stride=1, groups=1,
                              bias=True)
        
        
        self.ending = nn.Conv2d(in_channels=width, out_channels=out_channel, kernel_size=3, padding=1, stride=1, groups=1,
                              bias=True)
        # self.ending1 = nn.Conv2d(in_channels=width, out_channels=out_channel, kernel_size=3, padding=1, stride=1, groups=1,
        #                       bias=True)
    
        self.encoders = nn.ModuleList()
        self.decoders_trans = nn.ModuleList()
        # self.decoders_reflect = nn.ModuleList()
        self.middle_blks = nn.ModuleList()
        # self.middle_blks1 = nn.ModuleList()
        # 14*14
        self.crossattn = CrossAttentionWithFFN(embed_dim=1024, heads=8, ffn_dim=256)
        self.crossattn1 = CrossAttentionWithFFN(embed_dim=1024, heads=8, ffn_dim=256)
    
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()
        self.out_channel = out_channel
        self.middle_avgpool = nn.AvgPool2d(2)
        chan = width
        for num in enc_blk_nums:
            self.encoders.append(
                nn.Sequential(
                    *[NAFBlock(chan) for _ in range(num)]
                )
            )
            self.downs.append(
                nn.Conv2d(chan, 2*chan, 2, 2)
            )
            chan = chan * 2

        self.middle_blks = \
            nn.Sequential(
                *[NAFBlock(chan) for _ in range(middle_blk_num)]
            )
        # self.middle_blks1 = \
        #     nn.Sequential(
        #         *[NAFBlock(chan) for _ in range(middle_blk_num)]
        #     )

        for num in dec_blk_nums:
            self.ups.append(
                nn.Sequential(
                    nn.Conv2d(chan, chan * 2, 1, bias=False),
                    nn.PixelShuffle(2)
                )
            )
            chan = chan // 2
            self.decoders_trans.append(
                nn.Sequential(
                    *[NAFBlock(chan) for _ in range(num)]
                )
            )
            # self.decoders_reflect.append(
            #     nn.Sequential(
            #         *[NAFBlock(chan) for _ in range(num)]
            #     )
            # )

        # self.ending = DualStreamBlock(nn.Conv2d(64, 3, 3, padding=1))
        self.padder_size = 2 ** len(self.encoders)
        
            
    def forward(self, inp, Train=True):
        B, C, H, W = inp.shape
        inp = self.check_image_size(inp)
        mask = inp[:, 3:, :, :]
        x = self.intro(inp)

        encs = []

        for encoder, down in zip(self.encoders, self.downs):
            x = encoder(x)
            encs.append(x)
            x = down(x)

        mask = F.interpolate(mask, size=x.shape[2:], mode='bilinear', align_corners=False)
        ref_mask = (mask == 1).float()
        trans_mask = ((mask != 1) & (mask != 0)).float()
        T_feat = self.crossattn(x, trans_mask)
        T_feat = self.middle_blks(T_feat)
        R_feat = self.crossattn1(x, ref_mask)
        R_feat = self.middle_blks(R_feat)

        for decoder, up, enc_skip in zip(self.decoders_trans, self.ups, encs[::-1]):
            T_feat = up(T_feat)
            T_feat = T_feat + enc_skip
            T_feat = decoder(T_feat)
         
            R_feat = up(R_feat)
            R_feat = R_feat + enc_skip
            R_feat = decoder(R_feat)

        # T = self.ending(T_feat) + inp[:, :3, :, :]
        # R = self.ending1(R_feat)
        # for decoder, up, enc_skip in zip(self.decoders, self.ups, encs[::-1]):
        #     T_feat, R_feat = up(T_feat, R_feat)
        #     T_feat, R_feat = T_feat + enc_skip, R_feat + enc_skip
        #     T_feat, R_feat = decoder(T_feat, R_feat)

        T = self.ending(T_feat)
        R = self.ending(R_feat)

        return T, R
            
    def check_image_size(self, x):
        _, _, h, w = x.size()
        mod_pad_h = (self.padder_size - h % self.padder_size) % self.padder_size
        mod_pad_w = (self.padder_size - w % self.padder_size) % self.padder_size
        x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h))
        return x

class NAFNet_detect(nn.Module):

    def __init__(self, img_channel=3, out_channel=3, width=16, middle_blk_num=1, enc_blk_nums=[], dec_blk_nums=[]):
        super().__init__()

        self.intro = nn.Conv2d(in_channels=img_channel, out_channels=width, kernel_size=3, padding=1, stride=1, groups=1,
                              bias=True)
        self.ending = nn.Conv2d(in_channels=width, out_channels=out_channel, kernel_size=3, padding=1, stride=1, groups=1,
                              bias=True)

        self.encoders = nn.ModuleList()
        self.decoders = nn.ModuleList()
        self.middle_blks = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()
        self.out_channel = out_channel
        
        chan = width
        for num in enc_blk_nums:
            self.encoders.append(
                nn.Sequential(
                    *[NAFBlock(chan) for _ in range(num)]
                )
            )
            self.downs.append(
                nn.Conv2d(chan, 2*chan, 2, 2)
            )
            chan = chan * 2

        self.middle_blks = \
            nn.Sequential(
                *[NAFBlock(chan) for _ in range(middle_blk_num)]
            )

        for num in dec_blk_nums:
            self.ups.append(
                nn.Sequential(
                    nn.Conv2d(chan, chan * 2, 1, bias=False),
                    nn.PixelShuffle(2)
                )
            )
            chan = chan // 2
            self.decoders.append(
                nn.Sequential(
                    *[NAFBlock(chan) for _ in range(num)]
                )
            )

        self.padder_size = 2 ** len(self.encoders)

    def forward(self, inp):
        B, C, H, W = inp.shape
        inp = self.check_image_size(inp)
        
        x = self.intro(inp)

        encs = []

        for encoder, down in zip(self.encoders, self.downs):
            x = encoder(x)
            encs.append(x)
            x = down(x)

        x = self.middle_blks(x)

        for decoder, up, enc_skip in zip(self.decoders, self.ups, encs[::-1]):
            x = up(x)
            x = x + enc_skip
            x = decoder(x)

        y = self.ending(x)
        return y

    def check_image_size(self, x):
        _, _, h, w = x.size()
        mod_pad_h = (self.padder_size - h % self.padder_size) % self.padder_size
        mod_pad_w = (self.padder_size - w % self.padder_size) % self.padder_size
        x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h))
        return x


class NAFNetLocal(Local_Base, NAFNet):
    def __init__(self, *args, train_size=(1, 4, 224, 224), fast_imp=False, **kwargs):
        Local_Base.__init__(self)
        NAFNet.__init__(self, *args, **kwargs)

        N, C, H, W = train_size
        base_size = (int(H * 1.5), int(W * 1.5))

        self.eval()
        with torch.no_grad():
            self.convert(base_size=base_size, train_size=train_size, fast_imp=fast_imp)

from basicsr.models.archs.efficientNet import EfficientInterpolationNet
class EfficientNetLocal(Local_Base, EfficientInterpolationNet):
    def __init__(self, *args, train_size=(1, 3, 256, 256), fast_imp=False, **kwargs):
        Local_Base.__init__(self)
        EfficientInterpolationNet.__init__(self, *args, **kwargs)

        N, C, H, W = train_size
        base_size = (int(H * 1.5), int(W * 1.5))

        self.eval()
        with torch.no_grad():
            self.convert(base_size=base_size, train_size=train_size, fast_imp=fast_imp)

# from basicsr.models.archs.cirs import CRIS
# class CRISNet(Detect_Base, CRIS):
#     def __init__(self, *args, train_size=(1, 3, 256, 256), fast_imp=False, **kwargs):
#         Detect_Base.__init__(self)
#         CRIS.__init__(self, *args, **kwargs)

#         N, C, H, W = train_size
#         base_size = (int(H * 1.5), int(W * 1.5))

#         self.eval()
#         with torch.no_grad():
#             self.convert(base_size=base_size, train_size=train_size, fast_imp=fast_imp)
                        
class NAFNetLocal_detect(Local_Base, NAFNet_detect):
    def __init__(self, *args, train_size=(1, 3, 256, 256), fast_imp=False, **kwargs):
        Local_Base.__init__(self)
        NAFNet_detect.__init__(self, *args, **kwargs)

        N, C, H, W = train_size
        base_size = (int(H * 1.5), int(W * 1.5))

        self.eval()
        with torch.no_grad():
            self.convert(base_size=base_size, train_size=train_size, fast_imp=fast_imp)
            
class NAFNetLocal_6in(Local_Base, NAFNet):
    def __init__(self, *args, train_size=(1, 6, 256, 256), fast_imp=False, **kwargs):
        Local_Base.__init__(self)
        NAFNet.__init__(self, *args, **kwargs)

        N, C, H, W = train_size
        base_size = (int(H * 1.5), int(W * 1.5))

        self.eval()
        with torch.no_grad():
            self.convert(base_size=base_size, train_size=train_size, fast_imp=fast_imp)

class NAFNetLocal_4in(Local_Base, NAFNet):
    def __init__(self, *args, train_size=(1, 4, 256, 256), fast_imp=False, **kwargs):
        Local_Base.__init__(self)
        NAFNet.__init__(self, *args, **kwargs)

        N, C, H, W = train_size
        base_size = (int(H * 1.5), int(W * 1.5))

        self.eval()
        with torch.no_grad():
            self.convert(base_size=base_size, train_size=train_size, fast_imp=fast_imp)

if __name__ == '__main__':
    import time
    img_channel = 3
    width = 64

    enc_blks = [1, 1, 1, 28]
    middle_blk_num = 1
    dec_blks = [1, 1, 1, 4]
    
    net = NAFNet(img_channel=img_channel, width=width, middle_blk_num=middle_blk_num,
                      enc_blk_nums=enc_blks, dec_blk_nums=dec_blks).to('cuda:2')  
    
    # total_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
    # print(f"Total trainable parameters: {total_params}")
    
    num_runs = 2  # 总共运行的次数
    start_time = time.time()
    input_tensor = torch.randn(1, 3, 224, 224).to('cuda:2')  

    with torch.no_grad():
        for _ in range(num_runs):
            _ = net(input_tensor)

    end_time = time.time()
    total_time = end_time - start_time
    throughput = (num_runs) / total_time
    print(f"Total time: {total_time:.2f}s")
    print(f"Throughput: {throughput:.2f} img/s")

    # from thop import profile, clever_format

    # flops, params = profile(net, inputs=(input,))
    # flops, params = clever_format([flops, params], "%.3f")
    # print(flops, params)

    # from ptflops import get_model_complexity_info
    # macs, params = get_model_complexity_info(net, (4, 224, 224), as_strings=True, print_per_layer_stat=False)
    # print(f"MACs: {macs}")
    # print(f"Params: {params}") 
