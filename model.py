
import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
import torch.nn.functional as F
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
import math
from torch import einsum
from torchvision.ops.misc import Conv3dNormActivation,Conv2dNormActivation
from typing import Any, Callable, Optional
from torch import nn

class LOSTembedding(nn.Module):
    def __init__(self,steps,frameNos,widthRangeOrg,lentghRangeOrg,widthRangeSl,lentghRangeSl,embeddingDim=2,depth=4,inputOrg=512):
      super().__init__()

      self.steps = steps # total steps / QP values 
      self.frameNos = frameNos # number of frames in the clip/video
      self.widthRangeOrg = widthRangeOrg #maximum width resolution for the clip/video
      self.lentghRangeOrg = lentghRangeOrg #maximum height resolution for the clip/video
      self.widthRangeSl = widthRangeSl #resolution for width input 
      self.lentghRangeSl = lentghRangeSl #resolution for height input 
      self.embDim = embeddingDim #latent dimension considered fo each feature
      self.depth = depth

      self.stepEmbedding = nn.Embedding((steps//3)+1, self.embDim, max_norm=True)
      self.frameNoEmbedding = nn.Embedding((frameNos//2)+1, self.embDim, max_norm=True)
      self.widthEmbeddingOrg = nn.Embedding(widthRangeOrg, self.embDim, max_norm=True)
      self.lentghEmbeddingOrg = nn.Embedding(lentghRangeOrg, self.embDim, max_norm=True)
      self.widthEmbeddingSl = nn.Embedding(widthRangeSl, self.embDim, max_norm=True)
      self.lentghEmbeddingSl = nn.Embedding(lentghRangeSl, self.embDim, max_norm=True)

      self.proj = nn.Sequential(nn.Linear(6*embeddingDim, 16),nn.SiLU())

    def forward(self, loc,input_size):
      input_size  =int(math.sqrt(input_size))
      batchsize = loc.shape[1]
      step,frameNo,widthSP,lentghSP,widthSl,lentghSl = loc #SP -> Starting Point
      stepEmb = self.stepEmbedding(step).view(batchsize,self.embDim,1)
      frameNoEmb = self.frameNoEmbedding(frameNo).view(batchsize,self.embDim,1)
      widthSPEmb = self.widthEmbeddingOrg(widthSP).view(batchsize,self.embDim,1)
      lentghSPEmb = self.lentghEmbeddingOrg(lentghSP).view(batchsize,self.embDim,1)
      widthSlEmb = self.widthEmbeddingSl(widthSl).view(batchsize,self.embDim,1)
      lentghSlEmb = self.lentghEmbeddingSl(lentghSl).view(batchsize,self.embDim,1)
      lost  =torch.concat([stepEmb,frameNoEmb,widthSPEmb,lentghSPEmb,widthSlEmb,lentghSlEmb],dim=-1).view(batchsize,-1)
      lost = self.proj(lost)

      b,ll = lost.shape
      l = int(math.sqrt(ll))
      lost = rearrange(lost,'b (l1 l2) -> b l1 l2',l1=l,l2=l)
      lost=repeat(lost, 'b l1 l2-> b (k1 l1) (k2 l2)', k1=input_size//l, k2=input_size//l)

      return lost.view(b,-1,1)

class TemporalSeparableConv(nn.Sequential):
    def __init__(
        self,
        in_planes: int,
        out_planes: int,
        kernel_size: int,
        stride: int,
        padding: int,
        norm_layer: Optional[Callable[..., nn.Module]],
        activation_layer: Optional[Callable[..., nn.Module]]
    ):
        super().__init__(
            Conv3dNormActivation(
                in_planes,
                out_planes,
                kernel_size=(1, kernel_size, kernel_size),
                stride=(1, stride, stride),
                padding=(0, padding, padding),
                bias=True,
                norm_layer=norm_layer,
                activation_layer =activation_layer

            ),
            Conv3dNormActivation(
                out_planes,
                out_planes,
                kernel_size=(kernel_size, 1, 1),
                stride=(stride, 1, 1),
                padding=(padding, 0, 0),
                bias=True,
                norm_layer=norm_layer,
                activation_layer =activation_layer
            ),
        )
class LookAround(nn.Module):
    def __init__(self,levels,channels,kernel_size,padding,stride,norm_layer,activation_layer):
      super().__init__()
      self.channels = channels
      self.encoder  = nn.ModuleList([
          nn.Sequential(
                    TemporalSeparableConv(self.channels[i] ,self.channels[i+1],kernel_size,stride if i == 0 else stride+1,padding,None if i == 0 else norm_layer,activation_layer),
                    TemporalSeparableConv(self.channels[i+1],self.channels[i+1],kernel_size,stride,padding,norm_layer,activation_layer),
                    TemporalSeparableConv(self.channels[i+1],self.channels[i+1],kernel_size,stride,padding,norm_layer,activation_layer),
      ) for i in range(levels)])

    def forward(self, x,level):

      for layer in self.encoder[:level]:

        x = layer(x)

      #B, C ,F, H, W = x.shape
      x = rearrange(x, ' b c f h w -> b (h w) (c f) ')
      return x

class SpatialSeparableConv(nn.Sequential):
    def __init__(
        self,
        in_planes: int,
        out_planes: int,
        kernel_size: int,
        stride: int,
        padding: int,
        norm_layer: Optional[Callable[..., nn.Module]],
        activation_layer: Optional[Callable[..., nn.Module]]
    ):
        super().__init__(
            Conv2dNormActivation(
                in_planes,
                in_planes,
                kernel_size=(kernel_size, kernel_size),
                stride=(stride, stride),
                padding=(padding, padding),
                bias=True,
                norm_layer=norm_layer,
                activation_layer =activation_layer

            ),
            Conv2dNormActivation(
                in_planes,
                out_planes,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=True,
                norm_layer=norm_layer,
                activation_layer =activation_layer
            ),
        )

class LookAhead(nn.Module):
    def __init__(self,levels,channels,kernel_size,padding,stride,norm_layer,activation_layer):
      super().__init__()
      self.channels = channels
      self.bPicNet  = nn.ModuleList([
          nn.Sequential(
                    TemporalSeparableConv(self.channels[i],self.channels[i+1],kernel_size,stride+1,padding,None if i == 0 else norm_layer,activation_layer),
                    TemporalSeparableConv(self.channels[i+1],self.channels[i+1],kernel_size,stride,padding,norm_layer,activation_layer),
                    TemporalSeparableConv(self.channels[i+1],self.channels[i+1],kernel_size,stride,padding,norm_layer,activation_layer),
      ) for i in range(levels)])

      self.sPicNet  = nn.ModuleList([
          nn.Sequential(
                    TemporalSeparableConv(self.channels[i],self.channels[i+1],kernel_size,stride+1,padding,None if i == 0 else norm_layer,activation_layer),
                    TemporalSeparableConv(self.channels[i+1],self.channels[i+1],kernel_size,stride,padding,norm_layer,activation_layer),
                    TemporalSeparableConv(self.channels[i+1],self.channels[i+1],kernel_size,stride,padding,norm_layer,activation_layer),
      ) for i in range(levels)])

    def forward(self, x1,x2,level,weightDecay):

      for i,layer in enumerate(self.bPicNet[:level]):
        if i == 0:
        #   print(layer(x1).shape)
          y1 = layer(x1)*weightDecay
        else:
          y1 = layer(y1)*weightDecay

      for i,layer in enumerate(self.sPicNet[:level]):
        if i == 0:
          y2 = layer(x2)*weightDecay
        else:
          y2 = layer(y2)*weightDecay

      y = torch.concat([y1,y2],dim=1)
      y = y.squeeze(2)
      y = rearrange(y, ' b c h w -> b (h w) c ')

      return y


class eca_layer_1d(nn.Module):

    def __init__(self, channel, k_size=3):
        super(eca_layer_1d, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()
        self.channel = channel
        self.k_size =k_size

    def forward(self, x):
        y = self.avg_pool(x.transpose(-1, -2))
        y = self.conv(y.transpose(-1, -2))
        y = self.sigmoid(y)
        return x * y.expand_as(x)


class SepConv3d(torch.nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=0,
                 dilation=1,act_layer=nn.ReLU):
        super(SepConv3d, self).__init__()
        self.depthwise = torch.nn.Conv3d(in_channels,
                                         in_channels,
                                         kernel_size=kernel_size,
                                         stride=stride,
                                         padding=padding,
                                         dilation=dilation,
                                         groups=in_channels)
        self.pointwise = torch.nn.Conv3d(in_channels, out_channels, kernel_size=1)
        self.act_layer = act_layer() if act_layer is not None else nn.Identity()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride

    def forward(self, x):
        x = self.depthwise(x)
        x = self.act_layer(x)
        x = self.pointwise(x)
        return x

class ConvProjection(nn.Module):
    def __init__(self, dim,frameNos=3, heads = 8, dim_head = 64, kernel_size=3, q_stride=1, k_stride=1, v_stride=1, dropout = 0.,
                 last_stage=False,bias=True):

        super().__init__()

        inner_dim = dim_head *  heads
        self.heads = heads
        self.frameNos = frameNos
        pad = (kernel_size - q_stride)//2
        self.to_q = SepConv3d(dim, inner_dim, kernel_size, q_stride, pad, bias)
        self.to_k = SepConv3d(dim, inner_dim, kernel_size, k_stride, pad, bias)
        self.to_v = SepConv3d(dim, inner_dim, kernel_size, v_stride, pad, bias)

    def forward(self, x, attn_kv=None):
        b, n, cfn, h = *x.shape, self.heads
        l = int(math.sqrt(n))
        w = int(math.sqrt(n))
        attn_kv = x if attn_kv is None else attn_kv
        x = rearrange(x, 'b (l w) (c f) -> b c f l w', l=l, w=w, c=cfn//self.frameNos , f =self.frameNos)
        attn_kv = rearrange(attn_kv, 'b (l w) (c f) -> b c f l w', l=l, w=w, c=cfn//self.frameNos , f =self.frameNos)
        q = self.to_q(x)
        q = rearrange(q, 'b (h d) f l w -> b h (l w) (d f)', h=h)

        k = self.to_k(attn_kv)
        v = self.to_v(attn_kv)
        k = rearrange(k, 'b (h d) f l w -> b h (l w) (d f)', h=h)
        v = rearrange(v, 'b (h d) f l w -> b h (l w) (d f)', h=h)
        return q,k,v


class LinearProjection(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0., bias=True):
        super().__init__()
        inner_dim = dim_head *  heads
        self.heads = heads
        self.to_q = nn.Linear(dim, inner_dim, bias = bias)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias = bias)
        self.dim = dim
        self.inner_dim = inner_dim

    def forward(self, x, attn_kv=None):
        B_, N, C = x.shape
        if attn_kv is not None:
            attn_kv = attn_kv.unsqueeze(0).repeat(B_,1,1)
        else:
            attn_kv = x
        N_kv = attn_kv.size(1)
        q = self.to_q(x).reshape(B_, N, 1, self.heads, C // self.heads).permute(2, 0, 3, 1, 4)
        kv = self.to_kv(attn_kv).reshape(B_, N_kv, 2, self.heads, C // self.heads).permute(2, 0, 3, 1, 4)
        q = q[0]
        k, v = kv[0], kv[1]
        return q,k,v

class WindowAttention(nn.Module):
    def __init__(self, dim, win_size,num_heads, token_projection='linear', qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.):

        super().__init__()
        self.dim = dim
        self.win_size = win_size 
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * win_size[0] - 1) * (2 * win_size[1] - 1), num_heads))  

        coords_h = torch.arange(self.win_size[0])
        coords_w = torch.arange(self.win_size[1]) 
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))
        coords_flatten = torch.flatten(coords, 1)  
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :] 
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += self.win_size[0] - 1 
        relative_coords[:, :, 1] += self.win_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.win_size[1] - 1
        relative_position_index = relative_coords.sum(-1) 
        self.register_buffer("relative_position_index", relative_position_index)
        trunc_normal_(self.relative_position_bias_table, std=.02)

        if token_projection =='conv':
            self.qkv = ConvProjection(dim,num_heads,dim//num_heads,bias=qkv_bias)
        elif token_projection =='linear':
            self.qkv = LinearProjection(dim,num_heads,dim//num_heads,bias=qkv_bias)
        else:
            raise Exception("Projection error!")

        self.token_projection = token_projection
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, attn_kv=None, mask=None):
        B_, N, C = x.shape
        q, k, v = self.qkv(x,attn_kv)
        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.win_size[0] * self.win_size[1], self.win_size[0] * self.win_size[1], -1)
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
        ratio = attn.size(-1)//relative_position_bias.size(-1)
        relative_position_bias = repeat(relative_position_bias, 'nH l c -> nH l (c d)', d = ratio)

        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            mask = repeat(mask, 'nW m n -> nW m (n d)',d = ratio)
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N*ratio) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N*ratio)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

    def extra_repr(self) -> str:
        return f'dim={self.dim}, win_size={self.win_size}, num_heads={self.num_heads}'


########### self-attention #############
class Attention(nn.Module):
    def __init__(self, dim,num_heads, token_projection='linear', qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.):

        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = LinearProjection(dim,num_heads,dim//num_heads,bias=qkv_bias)

        self.token_projection = token_projection
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, attn_kv=None, mask=None):
        B_, N, C = x.shape
        q, k, v = self.qkv(x,attn_kv)
        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))
        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

    def extra_repr(self) -> str:
        return f'dim={self.dim}, num_heads={self.num_heads}'

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)
        self.in_features = in_features
        self.hidden_features = hidden_features
        self.out_features = out_features

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class STeFF(nn.Module):
    def __init__(self, dim=32 ,hidden_dim=128, act_layer=nn.GELU,drop = 0., use_eca=False,frameNos=3):
        super().__init__()
        self.linear1 = nn.Sequential(nn.Linear(dim, hidden_dim),
                                act_layer())
        self.dwconv = nn.Sequential(nn.Conv3d(hidden_dim//frameNos,hidden_dim//frameNos,groups=hidden_dim//frameNos,kernel_size=3,stride=1,padding=1),
                        act_layer())
        self.linear2 = nn.Sequential(nn.Linear(hidden_dim, dim))
        self.dim = dim
        self.frameNos = frameNos
        self.hidden_dim = hidden_dim
        self.eca = eca_layer_1d(dim) if use_eca else nn.Identity()

    def forward(self, x):
        bs, hw, cfn = x.size()
        hh = int(math.sqrt(hw))
        x = self.linear1(x)
        x = rearrange(x, ' b (h w) (c f) -> b c f h w ', h = hh, w = hh, c=self.hidden_dim//self.frameNos , f =self.frameNos)
        x = self.dwconv(x)
        x = rearrange(x, ' b c f h w -> b (h w) (c f)')
        x = self.linear2(x)
        x = self.eca(x)

        return x

def window_partition(x, win_size, dilation_rate=1):
    B, H, W, CFn = x.shape
    if dilation_rate !=1:
        x = x.permute(0,3,1,2)
        assert type(dilation_rate) is int, 'dilation_rate should be a int'
        x = F.unfold(x, kernel_size=win_size,dilation=dilation_rate,padding=4*(dilation_rate-1),stride=win_size)
        windows = x.permute(0,2,1).contiguous().view(-1, CFn, win_size, win_size)
        windows = windows.permute(0,2,3,1).contiguous()
    else:
        x = x.view(B, H // win_size, win_size, W // win_size, win_size, CFn)
        windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, win_size, win_size, CFn)
    return windows

def window_reverse(windows, win_size, H, W, dilation_rate=1):
    B = int(windows.shape[0] / (H * W / win_size / win_size))
    x = windows.view(B, H // win_size, W // win_size, win_size, win_size, -1)
    if dilation_rate !=1:
        x = windows.permute(0,5,3,4,1,2).contiguous()
        x = F.fold(x, (H, W), kernel_size=win_size, dilation=dilation_rate, padding=4*(dilation_rate-1),stride=win_size)
    else:
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x

class Downsample(nn.Module):
    def __init__(self, in_channel, out_channel,frameNos=3):
        super(Downsample, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(in_channel//frameNos, out_channel//frameNos ,kernel_size=(3,4,4), stride=(1,2,2), padding=1)

        )
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.frameNos = frameNos

    def forward(self, x):
        B, L, Cfn = x.shape
        C = Cfn // self.frameNos
        H = int(math.sqrt(L))
        W = int(math.sqrt(L))
        x = rearrange(x, 'b (h w) (c f) -> b c f h w', h = H, w = W, c=C , f =self.frameNos)
        x = self.conv(x) 
        x = rearrange(x, 'b c f h w ->  b (h w) (c f)')
        return x


class Upsample(nn.Module):
    def __init__(self, in_channel, out_channel,frameNos=3):
        super(Upsample, self).__init__()
        self.deconv = nn.Sequential(
            nn.ConvTranspose3d(in_channel//frameNos, out_channel//frameNos, kernel_size=2,stride=(1,2,2))
            ,nn.Conv3d(out_channel//frameNos, out_channel//frameNos, kernel_size=(2,1,1), stride=1,padding=0)
        )
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.frameNos = frameNos

    def forward(self, x):
        B, L, Cfn = x.shape
        C = Cfn // self.frameNos
        H = int(math.sqrt(L))
        W = int(math.sqrt(L))
        x = rearrange(x, 'b (h w) (c f) -> b c f h w', h = H, w = W, c=C , f =self.frameNos)
        x = self.deconv(x)
        x = rearrange(x, 'b c f h w ->  b (h w) (c f)')
        return x

class InputProj(nn.Module):
    def __init__(self, in_channel=3, out_channel=64, kernel_size=3, stride=1, norm_layer=None,act_layer=nn.LeakyReLU):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Conv3d(in_channel, out_channel, kernel_size=3, stride=stride, padding=kernel_size//2),
            act_layer(inplace=True)
        )
        if norm_layer is not None:
            self.norm = norm_layer(out_channel)
        else:
            self.norm = None
        self.in_channel = in_channel
        self.out_channel = out_channel

    def forward(self, x):
        B, C ,F, H, W = x.shape
        x = self.proj(x)
        x=rearrange(x, 'b c f h w -> b (h w) (c f)')
        if self.norm is not None:
            x = self.norm(x)
        return x

class OutputProj(nn.Module):
    def __init__(self, in_channel=64, out_channel=3, kernel_size=3,frameNos=3, stride=1, norm_layer=None,act_layer=None):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Conv3d(in_channel, out_channel, kernel_size=3, stride=stride, padding=kernel_size//2),
        )
        if act_layer is not None:
            self.proj.add_module(act_layer(inplace=True))
        if norm_layer is not None:
            self.norm = norm_layer(out_channel)
        else:
            self.norm = None
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.frameNos = frameNos

    def forward(self, x):
        B, L, Cfn = x.shape
        C = Cfn // self.frameNos
        H = int(math.sqrt(L))
        W = int(math.sqrt(L))
        x = rearrange(x, 'b (h w) (c f) -> b c f h w', h = H, w = W, c=C , f =self.frameNos)
        x = self.proj(x)
        if self.norm is not None:
            x = self.norm(x)
        return x


class TransformerBlock(nn.Module):
    def __init__(self, dim, input_resolution, num_heads, win_size=8, shift_size=0,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm,token_projection='linear',token_mlp='steff',
                 guidnace=None):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.win_size = win_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        self.token_mlp = token_mlp
        if min(self.input_resolution) <= self.win_size:
            self.shift_size = 0
            self.win_size = min(self.input_resolution)
        assert 0 <= self.shift_size < self.win_size, "shift_size must in 0-win_size"
        self.guidance = guidnace

        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(
            dim, win_size=to_2tuple(self.win_size), num_heads=num_heads,
            qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop,
            token_projection=token_projection)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        if token_mlp in ['ffn','mlp']:
            self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim,act_layer=act_layer, drop=drop)
        elif token_mlp=='steff':
            self.mlp =  STeFF(dim,mlp_hidden_dim,act_layer=act_layer, drop=drop)
        else:
            raise Exception("FFN error!")


    def with_pos_embed(self, tensor, pos):
        return tensor if pos is None else tensor + pos

    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution={self.input_resolution}, num_heads={self.num_heads}, " \
               f"win_size={self.win_size}, shift_size={self.shift_size}, mlp_ratio={self.mlp_ratio}"

    def forward(self, x, mask=None):
        if self.guidance is not None:
            x = torch.cat([x,self.guidance],dim=-1)
        B, L, CFn = x.shape
        #print(x.shape,'1')
        H = int(math.sqrt(L))
        W = int(math.sqrt(L))

        ## input mask
        if mask is not None:
            # print(mask.shape)
            # print(mask)
            input_mask = F.interpolate(mask, size=(H,W)).permute(0,2,3,1)
            input_mask_windows = window_partition(input_mask, self.win_size)
            attn_mask = input_mask_windows.view(-1, self.win_size * self.win_size) 
            attn_mask = attn_mask.unsqueeze(2)*attn_mask.unsqueeze(1) 
            attn_mask = attn_mask.masked_fill(attn_mask!=0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        else:
            attn_mask = None

        ## shift mask
        if self.shift_size > 0:
            # calculate attention mask for SW-MSA
            shift_mask = torch.zeros((1, H, W, 1)).type_as(x)
            h_slices = (slice(0, -self.win_size),
                        slice(-self.win_size, -self.shift_size),
                        slice(-self.shift_size, None))
            w_slices = (slice(0, -self.win_size),
                        slice(-self.win_size, -self.shift_size),
                        slice(-self.shift_size, None))
            cnt = 0
            for h in h_slices:
                for w in w_slices:
                    shift_mask[:, h, w, :] = cnt
                    cnt += 1
            shift_mask_windows = window_partition(shift_mask, self.win_size) 
            shift_mask_windows = shift_mask_windows.view(-1, self.win_size * self.win_size)
            shift_attn_mask = shift_mask_windows.unsqueeze(1) - shift_mask_windows.unsqueeze(2) 
            shift_attn_mask = shift_attn_mask.masked_fill(shift_attn_mask != 0, float(-100.0)).masked_fill(shift_attn_mask == 0, float(0.0))
            attn_mask = attn_mask + shift_attn_mask if attn_mask is not None else shift_attn_mask

        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, CFn)

        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x

        x_windows = window_partition(shifted_x, self.win_size)
        x_windows = x_windows.view(-1, self.win_size * self.win_size, CFn)
        wmsa_in = x_windows
        attn_windows = self.attn(wmsa_in, mask=attn_mask) 

        attn_windows = attn_windows.view(-1, self.win_size, self.win_size, CFn)
        shifted_x = window_reverse(attn_windows, self.win_size, H, W)

        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x
        x = x.view(B, H * W, CFn)

        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        del attn_mask
        return x

class BasicLayer(nn.Module):
    def __init__(self, dim, output_dim, input_resolution, depth, num_heads, win_size,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, use_checkpoint=False,
                 token_projection='linear',token_mlp='ffn', shift_flag=True,
                 ):

        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.use_checkpoint = use_checkpoint
        # build blocks
        if shift_flag:
            self.blocks = nn.ModuleList([
                TransformerBlock(dim=dim, input_resolution=input_resolution,
                                    num_heads=num_heads, win_size=win_size,
                                    shift_size=0 if (i % 2 == 0) else win_size // 2,
                                    mlp_ratio=mlp_ratio,
                                    qkv_bias=qkv_bias, qk_scale=qk_scale,
                                    drop=drop, attn_drop=attn_drop,
                                    drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                                    norm_layer=norm_layer,token_projection=token_projection,token_mlp=token_mlp,
                                    )
                for i in range(depth)])
        else:
            self.blocks = nn.ModuleList([
                TransformerBlock(dim=dim, input_resolution=input_resolution,
                                    num_heads=num_heads, win_size=win_size,
                                    shift_size=0,
                                    mlp_ratio=mlp_ratio,
                                    qkv_bias=qkv_bias, qk_scale=qk_scale,
                                    drop=drop, attn_drop=attn_drop,
                                    drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                                    norm_layer=norm_layer,token_projection=token_projection,token_mlp=token_mlp,
                                    )
            for i in range(depth)])

    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution={self.input_resolution}, depth={self.depth}"

    def forward(self, x, mask=None):
        for blk in self.blocks:
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x)
            else:
                x = blk(x,mask)
        return x

class DiQP(nn.Module):
    def __init__(self, crop_size=512, in_chans=3, dd_in=3,
                 embed_dim=32, depths=[2, 2, 2, 2, 2, 2, 2, 2, 2], num_heads=[1, 2, 4, 8, 16, 16, 8, 4, 2],
                 win_size=8, mlp_ratio=4., qkv_bias=True, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm, patch_norm=True,
                 use_checkpoint=False, token_projection='linear', token_mlp='steff',
                 dowsample=Downsample, upsample=Upsample, shift_flag=True,inputFrames=3,steps=256,frameNos=300,widthRangeOrg=8192,lentghRangeOrg=4320,widthRange=8192,lentghRange=4320, **kwargs):
        super().__init__()

        self.num_enc_layers = len(depths)//2
        self.num_dec_layers = len(depths)//2
        self.embed_dim = embed_dim * inputFrames
        self.patch_norm = patch_norm
        self.mlp_ratio = mlp_ratio
        self.token_projection = token_projection
        self.mlp = token_mlp
        self.win_size =win_size #window size for window mechanism (attention)
        self.reso = crop_size #input resolution
        self.pos_drop = nn.Dropout(p=drop_rate)
        self.dd_in = dd_in

        self.inputFrames =  #number of frames in the input
        self.steps = steps # total number of steps/qualities/QP values
        self.frameNos =frameNos # total number of frames in the whole clip/video
        self.widthRangeOrg = widthRangeOrg #maximum resolution for width in the clip/video
        self.lentghRangeOrg =lentghRangeOrg #maximum resolution for height in the clip/video

        # stochastic depth
        enc_dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths[:self.num_enc_layers]))]
        conv_dpr = [drop_path_rate]*depths[4]
        dec_dpr = enc_dpr[::-1]

        # build layers

        # Input/Output
        self.input_proj = InputProj(in_channel=dd_in, out_channel=self.embed_dim // inputFrames, kernel_size=3, stride=1, act_layer=nn.LeakyReLU)
        self.output_proj = OutputProj(in_channel=2*self.embed_dim // inputFrames, out_channel=in_chans, kernel_size=3, stride=1)


        self.lost = LOSTembedding(self.steps,self.frameNos,self.widthRangeOrg,self.lentghRangeOrg, self.reso, self.reso,embeddingDim=4)

        self.encoderGuide = LookAround(self.num_enc_layers,[3, 7, 17, 23, 29],3,1,1,nn.InstanceNorm3d,nn.SiLU)
        self.decoderGuide = LookAhead(self.num_enc_layers,[3, 7, 10, 13, 16],3,1,1,nn.InstanceNorm3d,nn.SiLU)

        # Encoder
        self.encoderlayer_0 = BasicLayer(dim=self.embed_dim +1-1,
                            output_dim=self.embed_dim,
                            input_resolution=(crop_size,
                                                crop_size),
                            depth=depths[0],
                            num_heads=num_heads[0],
                            win_size=win_size,
                            mlp_ratio=self.mlp_ratio,
                            qkv_bias=qkv_bias, qk_scale=qk_scale,
                            drop=drop_rate, attn_drop=attn_drop_rate,
                            drop_path=enc_dpr[sum(depths[:0]):sum(depths[:1])],
                            norm_layer=norm_layer,
                            use_checkpoint=use_checkpoint,
                            token_projection=token_projection,token_mlp=token_mlp,shift_flag=shift_flag)
        
        self.dowsample_0 = dowsample(self.embed_dim + 2*self.encoderGuide.channels[1]+1, self.embed_dim*2)

        self.encoderlayer_1 = BasicLayer(dim=self.embed_dim*2 ,
                            output_dim=self.embed_dim*2,
                            input_resolution=(crop_size // 2,
                                                crop_size // 2),
                            depth=depths[1],
                            num_heads=num_heads[1],
                            win_size=win_size,
                            mlp_ratio=self.mlp_ratio,
                            qkv_bias=qkv_bias, qk_scale=qk_scale,
                            drop=drop_rate, attn_drop=attn_drop_rate,
                            drop_path=enc_dpr[sum(depths[:1]):sum(depths[:2])],
                            norm_layer=norm_layer,
                            use_checkpoint=use_checkpoint,
                            token_projection=token_projection,token_mlp=token_mlp,shift_flag=shift_flag)

        self.dowsample_1 = dowsample(self.embed_dim*2 + self.encoderGuide.channels[2] +1, self.embed_dim*4)

        self.encoderlayer_2 = BasicLayer(dim=self.embed_dim*4 ,
                            output_dim=self.embed_dim*4,
                            input_resolution=(crop_size // (2 ** 2),
                                                crop_size // (2 ** 2)),
                            depth=depths[2],
                            num_heads=num_heads[2],
                            win_size=win_size,
                            mlp_ratio=self.mlp_ratio,
                            qkv_bias=qkv_bias, qk_scale=qk_scale,
                            drop=drop_rate, attn_drop=attn_drop_rate,
                            drop_path=enc_dpr[sum(depths[:2]):sum(depths[:3])],
                            norm_layer=norm_layer,
                            use_checkpoint=use_checkpoint,
                            token_projection=token_projection,token_mlp=token_mlp,shift_flag=shift_flag)

        self.dowsample_2 = dowsample(self.embed_dim*4 + self.encoderGuide.channels[3] +1, self.embed_dim*8)

        self.encoderlayer_3 = BasicLayer(dim=self.embed_dim*8 ,
                            output_dim=self.embed_dim*8,
                            input_resolution=(crop_size // (2 ** 3),
                                                crop_size // (2 ** 3)),
                            depth=depths[3],
                            num_heads=num_heads[3],
                            win_size=win_size,
                            mlp_ratio=self.mlp_ratio,
                            qkv_bias=qkv_bias, qk_scale=qk_scale,
                            drop=drop_rate, attn_drop=attn_drop_rate,
                            drop_path=enc_dpr[sum(depths[:3]):sum(depths[:4])],
                            norm_layer=norm_layer,
                            use_checkpoint=use_checkpoint,
                            token_projection=token_projection,token_mlp=token_mlp,shift_flag=shift_flag)

        self.dowsample_3 = dowsample(self.embed_dim*8 + self.encoderGuide.channels[4] +1, self.embed_dim*16)

        # Bottleneck
        self.conv = BasicLayer(dim=self.embed_dim*16,
                            output_dim=self.embed_dim*16,
                            input_resolution=(crop_size // (2 ** 4),
                                                crop_size // (2 ** 4)),
                            depth=depths[4],
                            num_heads=num_heads[4],
                            win_size=win_size,
                            mlp_ratio=self.mlp_ratio,
                            qkv_bias=qkv_bias, qk_scale=qk_scale,
                            drop=drop_rate, attn_drop=attn_drop_rate,
                            drop_path=conv_dpr,
                            norm_layer=norm_layer,
                            use_checkpoint=use_checkpoint,
                            token_projection=token_projection,token_mlp=token_mlp,shift_flag=shift_flag)

        # Decoder
        self.upsample_0 = upsample(self.embed_dim*16 + 2*self.decoderGuide.channels[4] +1, self.embed_dim*8)

        self.decoderlayer_0 = BasicLayer(dim=self.embed_dim*16 ,
                            output_dim=self.embed_dim*16,
                            input_resolution=(crop_size // (2 ** 3),
                                                crop_size // (2 ** 3)),
                            depth=depths[5],
                            num_heads=num_heads[5],
                            win_size=win_size,
                            mlp_ratio=self.mlp_ratio,
                            qkv_bias=qkv_bias, qk_scale=qk_scale,
                            drop=drop_rate, attn_drop=attn_drop_rate,
                            drop_path=dec_dpr[:depths[5]],
                            norm_layer=norm_layer,
                            use_checkpoint=use_checkpoint,
                            token_projection=token_projection,token_mlp=token_mlp,shift_flag=shift_flag,
                            )
        self.upsample_1 = upsample(self.embed_dim*16 +  2*self.decoderGuide.channels[3] +1, self.embed_dim*4)

        self.decoderlayer_1 = BasicLayer(dim=self.embed_dim*8 ,
                            output_dim=self.embed_dim*8,
                            input_resolution=(crop_size // (2 ** 2),
                                                crop_size // (2 ** 2)),
                            depth=depths[6],
                            num_heads=num_heads[6],
                            win_size=win_size,
                            mlp_ratio=self.mlp_ratio,
                            qkv_bias=qkv_bias, qk_scale=qk_scale,
                            drop=drop_rate, attn_drop=attn_drop_rate,
                            drop_path=dec_dpr[sum(depths[5:6]):sum(depths[5:7])],
                            norm_layer=norm_layer,
                            use_checkpoint=use_checkpoint,
                            token_projection=token_projection,token_mlp=token_mlp,shift_flag=shift_flag,
                            )
        self.upsample_2 = upsample(self.embed_dim*8 +  2*self.decoderGuide.channels[2] +1, self.embed_dim*2)

        self.decoderlayer_2 = BasicLayer(dim=self.embed_dim*4 ,
                            output_dim=self.embed_dim*4,
                            input_resolution=(crop_size // 2,
                                                crop_size // 2),
                            depth=depths[7],
                            num_heads=num_heads[7],
                            win_size=win_size,
                            mlp_ratio=self.mlp_ratio,
                            qkv_bias=qkv_bias, qk_scale=qk_scale,
                            drop=drop_rate, attn_drop=attn_drop_rate,
                            drop_path=dec_dpr[sum(depths[5:7]):sum(depths[5:8])],
                            norm_layer=norm_layer,
                            use_checkpoint=use_checkpoint,
                            token_projection=token_projection,token_mlp=token_mlp,shift_flag=shift_flag,
                            )
        self.upsample_3 = upsample(self.embed_dim*4 +  2*self.decoderGuide.channels[1] +1, self.embed_dim)

        self.decoderlayer_3 = BasicLayer(dim=self.embed_dim*2 ,
                            output_dim=self.embed_dim*2,
                            input_resolution=(crop_size,
                                                crop_size),
                            depth=depths[8],
                            num_heads=num_heads[8],
                            win_size=win_size,
                            mlp_ratio=self.mlp_ratio,
                            qkv_bias=qkv_bias, qk_scale=qk_scale,
                            drop=drop_rate, attn_drop=attn_drop_rate,
                            drop_path=dec_dpr[sum(depths[5:8]):sum(depths[5:9])],
                            norm_layer=norm_layer,
                            use_checkpoint=use_checkpoint,
                            token_projection=token_projection,token_mlp=token_mlp,shift_flag=shift_flag,
                            )

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'absolute_pos_embed'}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {'relative_position_bias_table'}

    def extra_repr(self) -> str:
        return f"self.embed_dim={self.embed_dim}, token_projection={self.token_projection}, token_mlp={self.mlp},win_size={self.win_size}"

    def forward(self, x,around,aheadCropped,aheadScaled,loc,weightDecay, mask=None):
        y = self.input_proj(x)
        y = self.pos_drop(y)
        conv0 = self.encoderlayer_0(y,mask=mask)
        _,L,_ = conv0.shape
        pool0 = self.dowsample_0(torch.cat([conv0,self.encoderGuide(around,1),self.lost(loc,L)],-1))
        conv1 = self.encoderlayer_1(pool0,mask=mask)
        _,L,_ = conv1.shape
        pool1 = self.dowsample_1(torch.cat([conv1,self.encoderGuide(around,2),self.lost(loc,L)],-1))
        conv2 = self.encoderlayer_2(pool1,mask=mask)
        _,L,_ = conv2.shape
        pool2 = self.dowsample_2(torch.cat([conv2,self.encoderGuide(around,3),self.lost(loc,L)],-1))
        _,L,_ = pool2.shape
        conv3 = self.encoderlayer_3(pool2,mask=mask)
        _,L,_ = conv3.shape
        pool3 = self.dowsample_3(torch.cat([conv3,self.encoderGuide(around,4),self.lost(loc,L)],-1))
        _,L,_ = pool2.shape
        conv4 = self.conv(pool3, mask=mask)
        _,L,_ = conv4.shape
        conv4  = torch.cat([conv4,self.decoderGuide(aheadCropped,aheadScaled,4,weightDecay),self.lost(loc,L)],-1)
        up0 = self.upsample_0(conv4)
        deconv0 = torch.cat([up0,conv3],-1)
        deconv0 = self.decoderlayer_0(deconv0,mask=mask)
        _,L,_ = deconv0.shape
        deconv0  = torch.cat([deconv0,self.decoderGuide(aheadCropped,aheadScaled,3,weightDecay),self.lost(loc,L)],-1)
        up1 = self.upsample_1(deconv0)
        deconv1 = torch.cat([up1,conv2],-1)
        deconv1 = self.decoderlayer_1(deconv1,mask=mask)
        _,L,_ = deconv1.shape
        deconv1  = torch.cat([deconv1,self.decoderGuide(aheadCropped,aheadScaled,2,weightDecay),self.lost(loc,L)],-1)
        up2 = self.upsample_2(deconv1)
        deconv2 = torch.cat([up2,conv1],-1)
        deconv2 = self.decoderlayer_2(deconv2,mask=mask)
        _,L,_ = deconv2.shape
        deconv2  = torch.cat([deconv2,self.decoderGuide(aheadCropped,aheadScaled,1,weightDecay),self.lost(loc,L)],-1)
        up3 = self.upsample_3(deconv2)
        deconv3 = torch.cat([up3,conv0],-1)
        _,L,_ = deconv0.shape
        deconv3 = self.decoderlayer_3(deconv3,mask=mask)
        y = self.output_proj(deconv3)
        return x + y if self.dd_in ==3 else y

