import torch
import torch.nn.functional as F
from torch import nn
from torchvision.models import resnet34
from torch.nn import Linear, Conv2d, BatchNorm2d, PReLU, Flatten, Sequential, Module, PixelShuffle, UpsamplingBilinear2d
import math
from gghead.models.encoders.helpers import get_blocks, bottleneck_IR, bottleneck_IR_SE
from timm.layers import DropPath, to_2tuple, trunc_normal_

class PatchEmbedding(nn.Module):
    def __init__(self, patch_size: int = 16, embedding_dim: int = 768):
        super(PatchEmbedding, self).__init__()
        self.patch_size = patch_size
        self.embedding_dim = embedding_dim
        self.cov2d = nn.Conv2d(3, embedding_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        """
        :param x: (B, C, H, W)
        :return: (B, seq_len, C)
        """
        o = self.cov2d(x)
        o = o.reshape(o.shape[0], self.embedding_dim, -1)
        o = o.permute(0, 2, 1)
        return o


class Encoder(nn.Module):
    def __init__(self, num_heads: int = 12, embedding_dim: int = 768, mlp_dim: int = 3072,
                 dropout: float = 0, attention_dropout: float = 0):
        super(Encoder, self).__init__()
        self.num_heads = num_heads

        # MSA
        self.norm1 = nn.LayerNorm(embedding_dim)
        self.attention = nn.MultiheadAttention(embedding_dim, num_heads,
                                               dropout=attention_dropout, batch_first=True)
        self.dropout = nn.Dropout(dropout)

        # MLP
        self.norm2 = nn.LayerNorm(embedding_dim)
        self.linear1 = nn.Linear(embedding_dim, mlp_dim)
        self.gelu = nn.GELU()
        self.dropout1 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(mlp_dim, embedding_dim)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x):
        """
        :param x: (B, seq_len, C)
        :return: (B, seq_len, C)
        """
        o = self.norm1(x)
        o, _ = self.attention(o, o, o, need_weights=False)
        o = self.dropout(o)
        o = o + x
        y = self.norm2(o)
        y = self.linear1(y)
        y = self.gelu(y)
        y = self.dropout1(y)
        y = self.linear2(y)
        y = self.dropout2(y)
        return y + o

class ViT(nn.Module):
    def __init__(self, image_size: int, num_classes: int, pretrain_dim: int,
                 patch_size: int = 16, num_layers: int = 12, num_heads: int = 12,
                 embedding_dim: int = 768, mlp_dim: int = 3072, dropout: float = 0.0,
                 attention_dropout: float = 0.0):
        super(ViT, self).__init__()
        self.embedding_dim = embedding_dim
        self.num_classes = num_classes

        # patch embedding
        self.patch_embedding = PatchEmbedding(patch_size, embedding_dim)

        # learnable class token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embedding_dim))

        # stand learnable 1-d position embedding
        seq_length = (image_size // patch_size) ** 2 + 1
        self.pos_embedding = nn.Parameter(torch.empty(1, seq_length, embedding_dim))

        # encoders
        self.encoders = nn.Sequential()
        for _ in range(num_layers):
            self.encoders.append(Encoder(num_heads, embedding_dim, mlp_dim, dropout, attention_dropout))

        # pretrain head
        self.head = nn.Sequential(nn.Linear(embedding_dim, pretrain_dim),
                                  nn.Tanh(),
                                  nn.Linear(pretrain_dim, num_classes))

    def forward(self, x):
        o = self.patch_embedding(x)
        batch_cls_token = self.cls_token.expand(o.shape[0], -1, -1)
        o = torch.cat([batch_cls_token, o], dim=1)
        o = o + self.pos_embedding
        o = self.encoders(o)
        o = o[:, 0]
        o = self.head(o)
        return o

    def pretrain(self, pretrain_dim):
        self.head = nn.Sequential(nn.Linear(self.embedding_dim, pretrain_dim),
                                  nn.Tanh(),
                                  nn.Linear(pretrain_dim, self.num_classes))

    def finetune(self):
        self.head = nn.Linear(self.embedding_dim, self.num_classes)

class LayerNorm(nn.LayerNorm):
    def forward(self, x):
        if x.ndim == 4:
            B, C, H, W = x.shape
            x = x.view(B, C, -1).transpose(1, 2)
            x = super().forward(x)
            x = x.transpose(1, 2).view(B, C, H, W)
        else:
            x = super().forward(x)
        return x

class OverlapPatchEmbed(nn.Module):
    """Image to Patch Embedding"""

    def __init__(self, img_size=224, patch_size=7, stride=4, in_chans=3, embed_dim=768):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)

        self.img_size = img_size
        self.patch_size = patch_size
        self.H, self.W = img_size[0] // patch_size[0], img_size[1] // patch_size[1]
        self.num_patches = self.H * self.W
        self.proj = nn.Conv2d(
            in_chans,
            embed_dim,
            kernel_size=patch_size,
            stride=stride,
            padding=(patch_size[0] // 2, patch_size[1] // 2),
        )
        self.norm = LayerNorm(embed_dim)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):
        x = self.proj(x)
        x = self.norm(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., sr_ratio=1):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        #self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        #self.proj_drop = nn.Dropout(proj_drop)

        self.sr_ratio = sr_ratio
        if sr_ratio > 1:
            self.sr = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
            self.norm = nn.LayerNorm(dim)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, H, W):
        B, N, C = x.shape
        q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        if self.sr_ratio > 1:
            x_ = x.permute(0, 2, 1).reshape(B, C, H, W)
            x_ = self.sr(x_).reshape(B, C, -1).permute(0, 2, 1)
            x_ = self.norm(x_)
            kv = self.kv(x_).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        else:
            kv = self.kv(x).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        #attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        #x = self.proj_drop(x)

        return x

class DWConv(nn.Module):
    def __init__(self, dim=768):
        super(DWConv, self).__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)

    def forward(self, x, H, W):
        B, N, C = x.shape
        x = x.transpose(1, 2).view(B, C, H, W)
        x = self.dwconv(x)
        x = x.flatten(2).transpose(1, 2)

        return x

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.dwconv = DWConv(hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        #self.drop = nn.Dropout(drop)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, H, W):
        x = self.fc1(x)
        x = self.dwconv(x, H, W)
        x = self.act(x)
        #x = self.drop(x)
        x = self.fc2(x)
        #x = self.drop(x)
        return x


class TransformerBlock(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, sr_ratio=1):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=drop, sr_ratio=sr_ratio)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, H, W):
        x = x + self.drop_path(self.attn(self.norm1(x), H, W))
        x = x + self.drop_path(self.mlp(self.norm2(x), H, W))

        return x

class Conv_1(nn.Module):
    def __init__(self, image_size: int, embed_dim: int,num_heads: int):
        super(Conv_1, self).__init__()
        self.net = nn.Sequential(
            OverlapPatchEmbed(img_size=64,stride=2,in_chans=256,embed_dim=1024),
            TransformerBlock(dim=1024,num_heads=4,mlp_ratio=2,sr_ratio=1),
            TransformerBlock(dim=1024, num_heads=4, mlp_ratio=2, sr_ratio=1),
            TransformerBlock(dim=1024, num_heads=4, mlp_ratio=2, sr_ratio=1),
            TransformerBlock(dim=1024, num_heads=4, mlp_ratio=2, sr_ratio=1),
            TransformerBlock(dim=1024, num_heads=4, mlp_ratio=2, sr_ratio=1),
            nn.PixelShuffle(upscale_factor=2),
            nn.UpsamplingBilinear2d(scale_factor=2),
            nn.Conv2d(256,128,kernel_size=(3,3),stride=(1,1),padding=(1,1)),
            nn.ReLU(),
            nn.UpsamplingBilinear2d(scale_factor=2),
            nn.Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(),
            nn.Conv2d(128, 96, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        )


    def forward(self, x):
        return self.net(x)

class Conv_2(nn.Module):
    def __init__(self, image_size: int, embed_dim: int,num_heads: int):
        self.net_1 = nn.Sequential(
            nn.Conv2d(192, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Conv2d(256, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.LeakyReLU(negative_slope=0.01),
            OverlapPatchEmbed(img_size=256, stride=2, in_chans=128, embed_dim=1024),
            TransformerBlock(dim=1024, num_heads=2, mlp_ratio=2, sr_ratio=2),
            nn.PixelShuffle(upscale_factor=2),
        )
        self.net_2 = nn.Sequential(
            nn.Conv2d(352, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Conv2d(256, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Conv2d(128, 96, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
        )
    def forward(self, x1,x2):
        x1 = self.net_1(x1)
        x1 = torch.cat([x1,x2], dim=1)
        x1 = self.net_2(x1)
        return x1


class DoubleConv(nn.Module):

    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.double_conv = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.PReLU(out_channels),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.PReLU(out_channels),
            nn.PReLU(out_channels)
        )

    def forward(self, x):
        return self.double_conv(x)


class Up(nn.Module):

    def __init__(self, in_channels, out_channels, upscale_factor=2, use_pixelshuffle=True):
        super().__init__()

        if use_pixelshuffle:
            self.up = nn.PixelShuffle(upscale_factor=upscale_factor)
        else:
            self.up = nn.Upsample(scale_factor=upscale_factor)
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)

        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class TriPlane_Encoder(Module):

    def __init__(self, opts):

        super(TriPlane_Encoder, self).__init__()
        self.opts = opts
        blocks = get_blocks(num_layers=50)
        unit_module = bottleneck_IR_SE

        self.input_layer = Sequential(Conv2d(9, 64, (3, 3), 2, 1, bias=False),
                                      BatchNorm2d(64),
                                      PReLU(64))

        modules = []
        for block in blocks:
            for bottleneck in block:
                modules.append(unit_module(bottleneck.in_channel,
                                           bottleneck.depth,
                                           bottleneck.stride))
        self.body = Sequential(*modules)
        # self.vit_1 = ViT(512,64,256)
        # self.vit_2 = ViT(512,64,256)
        if self.opts.use_pixelshuffle:
            self.up1 = Up(1024, 512, upscale_factor=1, use_pixelshuffle=True)
            self.up2 = Up(384, 384, use_pixelshuffle=True)
            self.up3 = Up(224, 256, use_pixelshuffle=True)
            self.up4 = Up(128, 96, use_pixelshuffle=True)
            self.up5 = nn.PixelShuffle(upscale_factor=2)
            self.final_head_two = nn.Sequential(
                nn.Conv2d(120, 96, kernel_size=3, padding=1),
                nn.PReLU(96),
                nn.Conv2d(96, 96, kernel_size=3, padding=1),
                nn.PReLU(96),
                nn.Conv2d(96, 96, kernel_size=1)
            )
        else:
            self.up1 = Up(1024, 512, upscale_factor=1, use_pixelshuffle=False)
            self.up2 = Up(768, 256, use_pixelshuffle=False)
            self.up3 = Up(384, 128, use_pixelshuffle=False)
            self.up4 = Up(192, 64, use_pixelshuffle=False)
            self.up5 = nn.Upsample(scale_factor=2)
            self.final_head_two = nn.Sequential(
                nn.Conv2d(86, 22, kernel_size=3, padding=1),
                nn.PReLU(22),
                nn.Conv2d(22, 22, kernel_size=3, padding=1),
                nn.PReLU(22),
                nn.Conv2d(22, 22, kernel_size=1)
            )

    def forward(self, x, t_planes):
        x = self.input_layer(x)
        modulelist = list(self.body._modules.values())
        for i, l in enumerate(modulelist):
            x = l(x)
            if i == 2:
                c0 = x
            if i == 6:
                c1 = x
            if i == 20:
                c2 = x
            elif i == 21:
                c3 = x

        tri_plane = self.up1(x, c3)
        tri_plane = self.up2(tri_plane, c2)
        tri_plane = self.up3(tri_plane, c1)
        tri_plane = self.up4(tri_plane, c0)
        tri_plane = self.up5(tri_plane)
        tri_plane = torch.cat([t_planes, tri_plane], dim=1)
        tri_plane2 = self.final_head_two(tri_plane)
        t_planes = t_planes + tri_plane2
        return t_planes