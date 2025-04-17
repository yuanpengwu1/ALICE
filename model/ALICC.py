
import numbers
import torch
# import pytorch_lightning as pl
from einops import rearrange
from torch import nn
from torch import nn, Tensor
from torch.nn import functional as F
from torchvision.ops import DeformConv2d
from timm.models.layers import DropPath
from ptflops import get_model_complexity_info

# ======================================================================================================================
class units:    #CA modules
    class ChannelAttention(nn.Module):
        def __init__(self, k_size=7):
            super().__init__()
            self.avgpool = nn.AdaptiveAvgPool2d(1)
            self.maxpool = nn.AdaptiveAvgPool2d(1)
            self.conv1d = nn.Conv1d(2, 1, kernel_size=k_size, padding=(k_size // 2), bias=False)
            self.sigmoid = nn.Sigmoid()

        def forward(self, x):
            y_avg = self.avgpool(x)
            y_max = self.avgpool(x)
            y = torch.cat([y_avg, y_max], dim=2)
            y = self.conv1d(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
            y = self.sigmoid(y.float())
            return x * y.expand_as(x)

    # # ================================================================add it myself
    class h_sigmoid(nn.Module):
        def __init__(self, inplace=True):
            super(units.h_sigmoid, self).__init__()
            self.relu = nn.ReLU6(inplace=inplace)

        def forward(self, x):
            return self.relu(x + 3) / 6

    class h_swish(nn.Module):
        def __init__(self, inplace=True):
            super(units.h_swish, self).__init__()
            self.sigmoid = units.h_sigmoid(inplace=inplace)

        def forward(self, x):
            return x * self.sigmoid(x)

    class CoordAtt(nn.Module):
        def __init__(self, inp, oup, reduction=32):
            super(units.CoordAtt, self).__init__()
            self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
            self.pool_w = nn.AdaptiveAvgPool2d((1, None))

            mip = max(8, inp // reduction)

            self.conv1 = nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0)
            self.ln1 =LayerNorm2d(mip)
            self.act = units.h_swish()

            self.conv_h = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
            self.conv_w = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)

        def forward(self, x):
            identity = x

            n, c, h, w = x.size()
            x_h = self.pool_h(x)
            x_w = self.pool_w(x).permute(0, 1, 3, 2)

            y = torch.cat([x_h, x_w], dim=2)
            y = self.conv1(y)
            y = self.ln1(y)
            y = self.act(y)

            x_h, x_w = torch.split(y, [h, w], dim=2)
            x_w = x_w.permute(0, 1, 3, 2)

            a_h = self.conv_h(x_h).sigmoid()
            a_w = self.conv_w(x_w).sigmoid()

            out = identity * a_w * a_h

            return out
    # =============================================================================================================
    #=============================================================================================================
    class ResBlock(nn.Module):
        def __init__(self, channels):
            super().__init__()
            self.res = nn.Sequential(
                nn.Conv2d(channels, channels // 2, kernel_size=3, padding=1, bias=False),
                nn.LeakyReLU(inplace=True),
                nn.Conv2d(channels // 2, channels, kernel_size=3, padding=1, bias=False),
            )

        def forward(self, x):
            return x + self.res(x)

    @staticmethod
    def Squeeze(channels):
        return nn.Conv2d(channels * 2, channels, 1, bias=False)

    @staticmethod
    def UpDownSample(ch_in, ch_out, scale):
        return nn.Sequential(nn.Upsample(scale_factor=scale, mode='bicubic', align_corners=False),
                             nn.Conv2d(ch_in, ch_out, kernel_size=1, bias=False))

    def to_3d(x):
        return rearrange(x, 'b c h w -> b (h w) c')

    def to_4d(x, h, w):
        return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)

    class BiasFree_LayerNorm(nn.Module):
        def __init__(self, normalized_shape):
            super().__init__()
            if isinstance(normalized_shape, numbers.Integral):
                normalized_shape = (normalized_shape,)
            normalized_shape = torch.Size(normalized_shape)

            assert len(normalized_shape) == 1

            self.weight = nn.Parameter(torch.ones(normalized_shape))
            self.normalized_shape = normalized_shape

        def forward(self, x):
            sigma = x.var(-1, keepdim=True, unbiased=False)
            return x / torch.sqrt(sigma + 1e-5) * self.weight

    class WithBias_LayerNorm(nn.Module):
        def __init__(self, normalized_shape):
            super().__init__()
            if isinstance(normalized_shape, numbers.Integral):
                normalized_shape = (normalized_shape,)
            normalized_shape = torch.Size(normalized_shape)

            assert len(normalized_shape) == 1

            self.weight = nn.Parameter(torch.ones(normalized_shape))
            self.bias = nn.Parameter(torch.zeros(normalized_shape))
            self.normalized_shape = normalized_shape

        def forward(self, x):
            mu = x.mean(-1, keepdim=True)
            sigma = x.var(-1, keepdim=True, unbiased=False)
            return (x - mu) / torch.sqrt(sigma + 1e-5) * self.weight + self.bias

    class LayerNorm(nn.Module):
        def __init__(self, dim, LayerNorm_type='WithBias'):
            super().__init__()
            if LayerNorm_type == 'BiasFree':
                self.body = units.BiasFree_LayerNorm(dim)
            else:
                self.body = units.WithBias_LayerNorm(dim)

        def forward(self, x):
            h, w = x.shape[-2:]
            return units.to_4d(self.body(units.to_3d(x)), h, w)

# ----------------------------------------------------------------------------------------------------------------------
class LayerNormFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, weight, bias, eps):
        ctx.eps = eps
        N, C, H, W = x.size()
        mu = x.mean(1, keepdim=True)
        var = (x - mu).pow(2).mean(1, keepdim=True)
        y = (x - mu) / (var + eps).sqrt()
        ctx.save_for_backward(y, var, weight)
        y = weight.view(1, C, 1, 1) * y + bias.view(1, C, 1, 1)
        return y

    @staticmethod
    def backward(ctx, grad_output):
        eps = ctx.eps

        N, C, H, W = grad_output.size()
        y, var, weight = ctx.saved_variables
        g = grad_output * weight.view(1, C, 1, 1)
        mean_g = g.mean(dim=1, keepdim=True)

        mean_gy = (g * y).mean(dim=1, keepdim=True)
        gx = 1. / torch.sqrt(var + eps) * (g - y * mean_gy - mean_g)
        return gx, (grad_output * y).sum(dim=3).sum(dim=2).sum(dim=0), grad_output.sum(dim=3).sum(dim=2).sum(
            dim=0), None

class LayerNorm2d(nn.Module):

    def __init__(self, channels, eps=1e-6):
        super(LayerNorm2d, self).__init__()
        self.register_parameter('weight', nn.Parameter(torch.ones(channels)))
        self.register_parameter('bias', nn.Parameter(torch.zeros(channels)))
        self.eps = eps

    def forward(self, x):
        return LayerNormFunction.apply(x, self.weight, self.bias, self.eps)
#-----------------------------------------------------DWT trans---------------------------------------------------------
def dwt_init(x):

    x01 = x[:, :, 0::2, :] / 2
    x02 = x[:, :, 1::2, :] / 2
    x1 = x01[:, :, :, 0::2]
    x2 = x02[:, :, :, 0::2]
    x3 = x01[:, :, :, 1::2]
    x4 = x02[:, :, :, 1::2]
    x_LL = x1 + x2 + x3 + x4
    x_HL = -x1 - x2 + x3 + x4
    x_LH = -x1 + x2 - x3 + x4
    x_HH = x1 - x2 - x3 + x4

    return x_LL, x_HL, x_LH, x_HH


def iwt_init(x):
    r = 2
    in_batch, in_channel, in_height, in_width = x.size()
    out_batch, out_channel, out_height, out_width = in_batch,int(in_channel/(r**2)), r * in_height, r * in_width
    x1 = x[:, :out_channel, :, :] / 2
    x2 = x[:,out_channel:out_channel * 2, :, :] / 2
    x3 = x[:,out_channel * 2:out_channel * 3, :, :] / 2
    x4 = x[:,out_channel * 3:out_channel * 4, :, :] / 2

    h = torch.zeros([out_batch, out_channel, out_height,
                     out_width]).float().to(x.device)

    h[:, :, 0::2, 0::2] = x1 - x2 - x3 + x4
    h[:, :, 1::2, 0::2] = x1 - x2 + x3 - x4
    h[:, :, 0::2, 1::2] = x1 + x2 - x3 - x4
    h[:, :, 1::2, 1::2] = x1 + x2 + x3 + x4

    return h


class DWT(nn.Module):
    def __init__(self):
        super(DWT, self).__init__()
        self.requires_grad = False

    def forward(self, x):
        return dwt_init(x)


class IWT(nn.Module):
    def __init__(self):
        super(IWT, self).__init__()
        self.requires_grad = False

    def forward(self, x):
        return iwt_init(x)

#=======================adaptive feature fusion module==================================================================
class AFFM(nn.Module):
    def __init__(self, in_channels, height=3, reduction=8, bias=False):
        super(AFFM, self).__init__()

        self.height = height
        d = max(int(in_channels / reduction), 4)

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_du = nn.Sequential(nn.Conv2d(in_channels, d, 1, padding=0, bias=bias), nn.PReLU())

        self.fcs = nn.ModuleList([])
        for i in range(self.height):
            self.fcs.append(nn.Conv2d(d, in_channels, kernel_size=1, stride=1, bias=bias))

        self.softmax = nn.Softmax(dim=1)
        self.res = units.ResBlock(in_channels)
    def forward(self, inp_feats):
        batch_size = inp_feats[0].shape[0]
        n_feats = inp_feats[0].shape[1]

        inp_feats = torch.cat(inp_feats, dim=1)
        inp_feats = inp_feats.view(batch_size, self.height, n_feats, inp_feats.shape[2], inp_feats.shape[3])

        feats_U = torch.sum(inp_feats, dim=1)
        feats_S = self.avg_pool(feats_U)
        feats_Z = self.conv_du(feats_S)

        attention_vectors = [fc(feats_Z) for fc in self.fcs]
        attention_vectors = torch.cat(attention_vectors, dim=1)
        attention_vectors = attention_vectors.view(batch_size, self.height, n_feats, 1, 1)
        # stx()
        attention_vectors = self.softmax(attention_vectors)

        feats_V = torch.sum(inp_feats * attention_vectors, dim=1)
        fusion_out = self.res(feats_V)
        return fusion_out

#------------------------------------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------------------------------
#Wavelet-Based Perturbation Elimination Network
class CoordAttBlock(nn.Module):
    def __init__(self, channels, k_size, groups):
        super().__init__()
        self.res = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=k_size, padding=(k_size // 2), bias=False, groups=groups),
            nn.Conv2d(channels, channels, kernel_size=1, bias=False),
            units.CoordAtt(channels, channels),  #add it myself
            nn.ReLU(True)
        )

    def forward(self, x):
        return x + self.res(x)

class WPENet(nn.Module):
    def __init__(self, ch_img=3, channels=32, groups=1):
        super().__init__()
        self.channels = channels
        self.img_in = nn.Conv2d(ch_img, channels, kernel_size=1, bias=False)

        self.p_down1 = nn.Sequential(
            nn.PixelUnshuffle(2),
            nn.Conv2d((2**2) * ch_img, channels, 1, 1, 0)
        )
        self.p_down2 = nn.Sequential(
            nn.PixelUnshuffle(4),
            nn.Conv2d((4**2) * ch_img, channels, 1, 1, 0)
        )
        self.p_down3 = nn.Sequential(
            nn.PixelUnshuffle(8),
            nn.Conv2d((8**2) * ch_img, channels, 1, 1, 0)
        )

        # encoder of UNet-64
        self.enc = nn.ModuleList([
            self.DownFRG(channels, groups=1),
            self.DownFRG(channels, groups=1),
            self.DownFRG(channels, groups=1)
        ])

        # decoder of UNet-64
        self.dec = nn.ModuleList([
            self.upFRG(channels, groups=1),
            self.upFRG(channels, groups=1),
            self.upFRG(channels, groups=1)
        ])

        self.img_out = nn.Conv2d(channels, ch_img, kernel_size=1, groups=1, bias=False)

    class DownFRG(nn.Module):
        def __init__(self, dim,groups=1):
            super().__init__()
            self.dwt = DWT()
            self.l_conv = nn.Conv2d(dim * 2, dim, 3, 1, 1)
            self.l_blk = RectificationConvModule(dim, dim,k_size=3, mlp_ratio=4, groups=1, REF=False, drop_out_rate=0.)

            self.h_fusion = AFFM(dim, height=3, reduction=8)
            self.h_blk = RectificationConvModule(dim, dim,k_size=3, mlp_ratio=4, groups=1, REF=False, drop_out_rate=0.)

        def forward(self, x, x_d):
            x_LL, x_HL, x_LH, x_HH = self.dwt(x)
            x_LL = self.l_conv(torch.cat([x_LL, x_d], dim=1))
            x_LL = self.l_blk(x_LL)

            x_h = self.h_fusion([x_HL, x_LH, x_HH])
            x_h = self.h_blk(x_h)
            return x_LL, x_h

    class upFRG(nn.Module):
        def __init__(self, dim,groups=1):
            super().__init__()
            self.iwt = IWT()
            self.l_blk = RectificationConvModule(dim, dim,k_size=3, mlp_ratio=4, groups=1, REF=False, drop_out_rate=0.)

            self.h_out_conv = nn.Conv2d(dim, dim * 3, 3, 1, 1)
            self.h_blk = RectificationConvModule(dim, dim, k_size=3, mlp_ratio=4, groups=1, REF=False, drop_out_rate=0.)

        def forward(self, x_l, x_h):
            x_l = self.l_blk(x_l)
            x_h = self.l_blk(x_h)
            x_h = self.h_out_conv(x_h)
            x_l = self.iwt(torch.cat([x_l, x_h], dim=1))

            return x_l

    def forward(self, x):

        img = x
        img_down1 = self.p_down1(x)
        img_down2 = self.p_down2(x)
        img_down3 = self.p_down3(x)

        ##### shallow conv #####
        x1 = self.img_in(img)

        # Down-path (Encoder)
        x_l, x_H1 = self.enc[0](x1, img_down1)
        # print(x_l.shape)
        # print(x_H1.shape)
        x_l, x_H2 = self.enc[1](x_l, img_down2)
        x_l, x_H3 = self.enc[2](x_l, img_down3)

        # Up-path (Decoder)
        x_l = self.dec[0](x_l, x_H3)
        x_l = self.dec[1](x_l, x_H2)
        x_l = self.dec[2](x_l, x_H1)

        noise = self.img_out(x_l)

        return noise

#==================================Illumination Matching Transformation Module (IMTM)===============================
def batched_index_select(input, dim, index):
    for ii in range(1, len(input.shape)):
        if ii != dim:
            index = index.unsqueeze(ii)
    expanse = list(input.shape)
    expanse[0] = -1
    expanse[dim] = -1
    index = index.expand(expanse)
    return torch.gather(input, dim, index)

def neirest_neighbores(input_maps, candidate_maps, distances, num_matches):
    batch_size = input_maps.size(0)

    if num_matches is None or num_matches == -1:
        num_matches = input_maps.size(1)

    topk_values, topk_indices = distances.topk(k=1, largest=False)


    topk_values = topk_values.squeeze(-1)
    topk_indices = topk_indices.squeeze(-1)

    sorted_values, sorted_values_indices = torch.sort(topk_values, dim=1)

    sorted_indices, sorted_indices_indices = torch.sort(sorted_values_indices, dim=1)

    mask = torch.stack(
        [
            torch.where(sorted_indices_indices[i] < num_matches, True, False)
            for i in range(batch_size)
        ]
    )
    topk_indices_selected = topk_indices.masked_select(mask)
    topk_indices_selected = topk_indices_selected.reshape(batch_size, num_matches)
    # indices = (
    #     torch.arange(0, topk_values.size(1))
    #     .unsqueeze(0)
    #     .repeat(batch_size, 1)
    #     .to(topk_values.device)
    # )
    # indices_selected = indices.masked_select(mask)
    # indices_selected = indices_selected.reshape(batch_size, num_matches)
    # filtered_input_maps = batched_index_select(input_maps, 1, indices_selected)
    filtered_candidate_maps = batched_index_select(
        candidate_maps, 1, topk_indices_selected
    )

    # return filtered_input_maps, filtered_candidate_maps
    return filtered_candidate_maps


def neirest_neighbores_on_l2(input_maps, candidate_maps, num_matches):
    """
    input_maps: (B, H * W, C)
    candidate_maps: (B, H * W, C)
    """
    distances = torch.cdist(input_maps, candidate_maps)
    return neirest_neighbores(input_maps, candidate_maps, distances, num_matches)

class Matching(nn.Module):
    def __init__(self, dim=32, match_factor=1):
        super(Matching, self).__init__()
        self.num_matching = int(dim/match_factor)
    def forward(self, x, Ym):
        b, c, h, w = x.size()
        x = x.flatten(2, 3)
        Ym = Ym.flatten(2, 3)
        # print('x, perception1', x.size(), perception.size())
        filtered_candidate_maps = neirest_neighbores_on_l2(x, Ym, self.num_matching)
        # filtered_input_maps = filtered_input_maps.reshape(b, self.num_matching, h, w)
        filtered_candidate_maps = filtered_candidate_maps.reshape(b, self.num_matching, h, w)
        return filtered_candidate_maps


class Matching_transformation(nn.Module):
    def __init__(self, dim=32, match_factor=1, ffn_expansion_factor=2, bias=True):
        super(Matching_transformation, self).__init__()
        self.num_matching = int(dim / match_factor)
        self.channel = dim
        hidden_features = int(self.channel * ffn_expansion_factor)
        self.matching = Matching(dim=dim, match_factor=match_factor)

        self.dwconv = nn.Sequential(nn.Conv2d(2 * self.num_matching, hidden_features, 1, bias=bias),
                                    nn.Conv2d(hidden_features, hidden_features, kernel_size=3, stride=1, padding=1,
                                              groups=hidden_features, bias=bias), nn.GELU(),
                                    nn.Conv2d(hidden_features, 2 * self.num_matching, 1, bias=bias))
        self.conv12 = nn.Conv2d(2 * self.num_matching, self.channel, 1, bias=bias)

    def forward(self, x, Ym):    #perception当成Mono。
        filtered_candidate_maps = self.matching(x, Ym)
        concat = torch.cat([x, filtered_candidate_maps], dim=1)
        # conv11 = self.conv11(concat)
        dwconv = self.dwconv(concat)
        Trans_out = self.conv12(dwconv * concat)

        return Trans_out
#======================================================DRCM============================================================
class RectificationConvModule(nn.Module):
    def __init__(self, channels, ch_z, k_size=3, mlp_ratio=4, groups=1, REF=True, CODE=True, drop_out_rate=0.):
        super().__init__()
        ###############attention########################
        self.res = nn.Sequential(
            #nn.Conv2d(channels, channels, kernel_size=k_size, padding=(k_size // 2), bias=False, groups=groups),
            #nn.Conv2d(channels, channels, kernel_size=1, bias=False),
            nn.Conv2d(channels, channels, kernel_size=1, padding=0, stride=1, groups=1,
                      bias=True),  # add 1x1conv
            nn.Conv2d(channels, channels, 3, padding=1,stride=1, groups=channels,bias=True),  #### add dwconv
            nn.GELU(),#### add
            #units.ChannelAttention(),
            units.CoordAtt(channels, channels),  # add it myself
            #nn.LeakyReLU(inplace=True)
            nn.Conv2d(channels, channels, kernel_size=1, padding=0, stride=1, groups=1,bias=True),
        )

        #self.norm1 = nn.BatchNorm2d(channels)
        #self.norm2 = nn.BatchNorm2d(channels)
        self.norm1 = LayerNorm2d(channels)                 ######add
        self.norm2 = LayerNorm2d(channels)                 ######add

        self.dropout1 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
        self.dropout2 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()

        #self.mlp = self.MLP(channels=channels, mlp_ratio=mlp_ratio,bias=True) #################add_DGFN
        self.mlp = self.MLP(channels=channels, mlp_ratio=mlp_ratio)
        #self.mlp = self.MLP(in_features=channels, hidden_features=int(channels * mlp_ratio))  # add_CONVGLU

        if REF:
            self.afm = self.alignment(channels=channels)

    class alignment(nn.Module):
        def __init__(self, channels, memory=False, stride=1, type='group_conv'):
            super().__init__()

            act = nn.GELU()
            bias = False

            kernel_size = 3
            padding = kernel_size // 2
            deform_groups = 8
            out_channels = deform_groups * 3 * (kernel_size ** 2)

            self.offset_conv = nn.Conv2d(channels, out_channels, kernel_size, stride=1, padding=padding, bias=bias)
            self.deform = DeformConv2d(channels, channels, kernel_size, padding=2, groups=deform_groups, dilation=2)
            self.back_projection = self.ref_back_projection(channels, stride=1)

            self.bottleneck = nn.Sequential(nn.Conv2d(channels * 2, channels, kernel_size=3, padding=1, bias=bias), act)

        def offset_gen(self, x):
            o1, o2, mask = torch.chunk(x, 3, dim=1)
            offset = torch.cat((o1, o2), dim=1)
            mask = torch.sigmoid(mask)

            return offset, mask

        def forward(self, Ym, Yc):
            ref = Ym
            offset_feat = self.bottleneck(torch.cat([ref, Yc], dim=1))
            # 是将参考帧 ref 和输入帧 x 沿着通道维度进行拼接，
            # 这样做的目的是将两个帧的特征信息合并在一起，使得模型能够同时考虑参考帧和输入帧的特征。

            offset, mask = self.offset_gen(self.offset_conv(offset_feat))
            aligned_feat = self.deform(Yc, offset, mask)
            Y = self.back_projection(Ym,aligned_feat)

            return Y
        #########################################################Reference Based Feature Enrichmen####################
        class ref_back_projection(nn.Module):
            def __init__(self, channels, stride):
                super().__init__()

                bias = False
                self.feat_fusion = nn.Sequential(nn.Conv2d(channels * 2, channels, 3, stride=1, padding=1), nn.GELU())
                # self.encoder1 = nn.Sequential(*[
                #     BFA(dim=channels * 2, num_heads=1, stride=stride, ffn_expansion_factor=2.66, bias=bias,
                #         LayerNorm_type='WithBias') for i in range(1)])

                self.encoder1 = nn.Sequential(
                nn.Conv2d(channels* 2, channels* 2, kernel_size=1, padding=0, stride=1, groups=1,
                          bias=True),  # add 1x1conv
                nn.Conv2d(channels* 2, channels* 2, 3, padding=1, stride=1, groups=channels* 2, bias=True),  #### add dwconv
                nn.GELU(),
                units.CoordAtt(channels* 2, channels* 2),
                nn.Conv2d(channels* 2, channels* 2, kernel_size=1, padding=0, stride=1, groups=1, bias=True),
                )

                self.feat_expand = nn.Sequential(nn.Conv2d(channels, channels * 2, 3, stride=1, padding=1), nn.GELU())
                self.diff_fusion = nn.Sequential(nn.Conv2d(channels * 2, channels, 3, stride=1, padding=1), nn.GELU())

            def forward(self, Ym, Yc):
                # feat = self.encoder1(x)
                #首先，分别得到对齐后特征burst features和参考帧特征base feature，将后者重复f次后和前者concatenate，
                #然后再通过Burst Feature Fusion (BFF)模块融合并恢复到通道数。
                #对齐后特征burst features为aligned_feat= self.deform(Yc, offset, mask)
                #参考帧特征base feature为Ym。

                ref = Ym
                feat = self.encoder1(torch.cat([ref, Yc], dim=1))

                fused_feat = self.feat_fusion(feat)
                exp_feat = self.feat_expand(fused_feat)

                residual = exp_feat - feat
                residual = self.diff_fusion(residual)
                fused_feat = fused_feat + residual

                return fused_feat

    class MLP(nn.Module):
        def __init__(self, channels, mlp_ratio=2):
            super().__init__()
            self.fc1 = nn.Conv2d(channels, int(channels * mlp_ratio), kernel_size=1, bias=False)
            self.act = nn.GELU()   ##修改成GELU
            self.fc2 = nn.Conv2d(int(channels * mlp_ratio), channels, kernel_size=1, bias=False)

        def forward(self, x):
            return self.fc2(self.act(self.fc1(x)))

    def forward(self, x, Y=None):
        if Y is not None:
            Y_align = self.afm(x, Y)
            x = x + Y_align

        x = x + self.dropout1(self.res(self.norm1(x)))
        x = x + self.dropout2(self.mlp(self.norm2(x)))

        if Y is not None:
            return x, Y_align
        else:
            return x
#========================illumination Matching Transformation Transformer Block=========================================
class IMTTB(nn.Module):
    def __init__(self, dim=32, ch_Z=64, num_heads=1, match_factor=2, ffn_expansion_factor=2 , bias=True,
                 LayerNorm_type='WithBias', mode='Y',REF=True):
        super(IMTTB, self).__init__()
        self.dim =dim
        self.norm1 = units.LayerNorm(dim, LayerNorm_type)
        self.norm2 = units.LayerNorm(dim, LayerNorm_type)
        self.EDAFB = self.AlignFusionModule(channels=dim)
        if mode == 'Y' and REF:
            self.attn1 = self.Attention(dim=dim,
                                  num_heads=num_heads,
                                  match_factor=match_factor,
                                  ffn_expansion_factor=ffn_expansion_factor,
                                  bias=bias,
                                  attention_matching=REF)
            self.ffn1 = self.FeedForward(dim=dim,
                                   match_factor=match_factor,
                                   ffn_expansion_factor=ffn_expansion_factor,
                                   bias=bias,
                                   ffn_matching=REF)

        else:
            self.attn2 = self.Attention(dim=dim,
                                  num_heads=num_heads,
                                  match_factor=match_factor,
                                  ffn_expansion_factor=ffn_expansion_factor,
                                  bias=bias,
                                  attention_matching=False)
            self.ffn2 = self.FeedForward(dim=dim,
                                   match_factor=match_factor,
                                   ffn_expansion_factor=ffn_expansion_factor,
                                   bias=bias,
                                   ffn_matching=False)
###################################################################alignment############################################
    class AlignFusionModule(nn.Module):
        def __init__(self, channels, ch_hidden=32, k_size=3):
            super().__init__()
            self.grid = nn.Sequential(
                # units.ResBlock(channels),
                nn.Conv2d(channels, ch_hidden, kernel_size=k_size, padding=k_size // 2, bias=False),
                nn.ReLU(inplace=True),
                nn.Conv2d(ch_hidden, 2, kernel_size=k_size, padding=k_size // 2, bias=False),
                # nn.ReLU(inplace=True),
            )

            self.sigmoid = nn.Sigmoid()
            self.conv = nn.Conv2d(channels * 2, channels, kernel_size=1, padding=0, bias=False)

        def forward(self, Yc, Ym):
            grid = self.grid(Yc).permute(0, 2, 3, 1).contiguous()
            align_Ym = F.grid_sample(Ym, grid, mode='nearest', align_corners=True)
            z = self.sigmoid(self.conv(torch.cat([Yc, align_Ym], dim=1)))
            Y = Yc * z + align_Ym * (1 - z)
            return Y
#################################################=========IMT===========================================================
    class FeedForward(nn.Module):
        def __init__(self, dim=32, match_factor=2, ffn_expansion_factor=2, bias=True,
                     ffn_matching=True):
            super().__init__()
            self.num_matching = int(dim / match_factor)
            self.channel = dim
            self.matching = ffn_matching
            hidden_features = int(self.channel * ffn_expansion_factor)

            self.project_in = nn.Sequential(
                nn.Conv2d(self.channel, hidden_features, 1, bias=bias),
                nn.Conv2d(hidden_features, self.channel, kernel_size=3, stride=1, padding=1, groups=self.channel,
                          bias=bias)
            )
            if self.matching is True:

                self.matching_transformation = Matching_transformation(dim=dim,
                                                                       match_factor=match_factor,
                                                                       ffn_expansion_factor=ffn_expansion_factor,
                                                                       bias=bias)

            self.project_out = nn.Sequential(
                nn.Conv2d(self.channel, hidden_features, kernel_size=3, stride=1, padding=1, groups=self.channel,
                          bias=bias),
                # nn.GELU(),
                nn.Conv2d(hidden_features, self.channel, 1, bias=bias))

        def forward(self, x, Ym):
            project_in = self.project_in(x)
            if self.matching is True:
                    out = self.matching_transformation(project_in, Ym)
            else:
                out = project_in
            project_out = self.project_out(out)
            return project_out


    class Attention(nn.Module):
        def __init__(self, dim, num_heads, match_factor=2, ffn_expansion_factor=2, bias=True,
                     attention_matching=True):
            super().__init__()
            self.num_heads = num_heads
            self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

            self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)
            self.qkv_dwconv = nn.Conv2d(dim * 3, dim * 3, kernel_size=3, stride=1, padding=1, groups=dim * 3,
                                        bias=bias)
            self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
            self.matching = attention_matching
            if self.matching is True:
                self.matching_transformation = Matching_transformation(dim=dim,
                                                                       match_factor=match_factor,
                                                                       ffn_expansion_factor=ffn_expansion_factor,
                                                                       bias=bias)

        def forward(self, x, Ym):
            b, c, h, w = x.shape

            qkv = self.qkv_dwconv(self.qkv(x))
            q, k, v = qkv.chunk(3, dim=1)

            # perception = self.LayerNorm(perception)
            if self.matching is True:
                q = self.matching_transformation(q, Ym)
            else:
                q = q
            q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
            k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
            v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

            q = torch.nn.functional.normalize(q, dim=-1)
            k = torch.nn.functional.normalize(k, dim=-1)

            attn = (q @ k.transpose(-2, -1)) * self.temperature
            attn = attn.softmax(dim=-1)

            out = (attn @ v)

            out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

            out = self.project_out(out)
            return out
#=======================================================================================================================
    def forward(self, x, Y=None):
        if Y is not None:
            Y_align = self.EDAFB(x, Y)
            x = x + Y_align
            x = x + self.attn1(self.norm1(x), Y)
            x = x + self.ffn1(self.norm2(x), Y)
        else:
            x = x + self.attn2(self.norm1(x),Y)
            x = x + self.ffn2(self.norm2(x),Y)

        if Y is not None:
            return x, Y_align
        else:
            return x

#=======================================================================================================================
class RefRectifyNet(nn.Module):
    def __init__(self, ch_img=3, ch_z=64, ch_level=32, ch_add=16, k_size=3,
                 mlp_ratio=4, groups=1, REF=True, CODE=True):
        super().__init__()
        # Configs
        self.num_head = 8
        self.split_size = 4
        self.ch_level = ch_level
        self.ch_level1 = ch_level + ch_add
        self.ch_level2 = ch_level + ch_add * 2
        self.ch_level3 = self.ch_level2 * 2
        self.REF = REF

        # In_feats
        self.feat = nn.ModuleList([
            nn.ModuleList([
                nn.Conv2d(2, ch_level, kernel_size=k_size, padding=(k_size // 2), bias=False, groups=groups),
            ]),
            nn.ModuleList([
                nn.Conv2d(1, ch_level, kernel_size=k_size, padding=(k_size // 2), bias=False, groups=groups),
            ])
        ])
        if self.REF:
            self.feat.append(
                nn.Conv2d(1, ch_level, kernel_size=k_size, padding=(k_size // 2), bias=False, groups=groups)
            )

        # UpDownSample & Squeeze
        self.UVdown = self.UpDown_List(mode='Down')
        self.Ydown = self.UpDown_List(mode='Down')
        self.enc_UVdown = self.UpDown_List(mode='Down')
        self.enc_Ydown = self.UpDown_List(mode='Down')
        self.dec_Yup = self.UpDown_List(mode='Up')
        self.dec_UVup = self.UpDown_List(mode='Up')
        self.squ = nn.Conv2d(ch_level * 2, ch_level, kernel_size=1, bias=False)

        # Encoder Y/UV & Decoder
        self.enc_UV = self.Module_List( mlp_ratio, groups, mode='UV')
        self.enc_Y = self.Module_List(mlp_ratio, groups, mode='Y')
        self.dec_UV = self.Module_List( mlp_ratio, groups, mode='UV')
        self.dec_Y = self.Module_List(mlp_ratio, groups, mode='Y')

        # Pixel Detail Enhance & Out
        self.drm = self.DetailRefineModule(ch_level, 7, ch_level)
        self.out = nn.Conv2d(ch_level, ch_img, 1, groups=1, bias=False)

        self.AFFM = AFFM(ch_level, height=4, reduction=8)

    def Module_List(self, mlp_ratio, groups, mode):
        if mode == 'UV':
            return nn.ModuleList([
                nn.ModuleList([
                    RectificationConvModule(self.ch_level, self.ch_level, 7, mlp_ratio, groups, False),
                    RectificationConvModule(self.ch_level, self.ch_level, 7, mlp_ratio, groups, False)
                ]),
                nn.ModuleList([
                    RectificationConvModule(self.ch_level1, self.ch_level1, 5, mlp_ratio, groups, False),
                    RectificationConvModule(self.ch_level1, self.ch_level1, 5, mlp_ratio, groups, False)
                ]),
                nn.ModuleList([
                    RectificationConvModule(self.ch_level2, self.ch_level2, 3, mlp_ratio, groups, False),
                    RectificationConvModule(self.ch_level2, self.ch_level2 ,3, mlp_ratio, groups, False)
                ]),
                nn.ModuleList([
                    IMTTB(self.ch_level3, self.ch_level3, self.num_head*2, 1, 2, True,
                                         mode=mode, REF=False),
                    IMTTB(self.ch_level3, self.ch_level3, self.num_head*2, 1, 2, True,
                                         mode=mode, REF=False),
                ])
            ])
        elif mode == 'Y':
            return nn.ModuleList([
                nn.ModuleList([
                    RectificationConvModule(self.ch_level, self.ch_level, 7, mlp_ratio, groups, self.REF),
                    RectificationConvModule(self.ch_level, self.ch_level, 7, mlp_ratio, groups, self.REF)
                ]),
                nn.ModuleList([
                    RectificationConvModule(self.ch_level1, self.ch_level1, 5, mlp_ratio, groups, self.REF),
                    RectificationConvModule(self.ch_level1, self.ch_level1, 5, mlp_ratio, groups, self.REF)
                ]),
                nn.ModuleList([

                    RectificationConvModule(self.ch_level2, self.ch_level2, 3, mlp_ratio, groups, self.REF),
                    RectificationConvModule(self.ch_level2, self.ch_level2, 3, mlp_ratio, groups, self.REF)
                ]),
                nn.ModuleList([
                    IMTTB(self.ch_level3, self.ch_level3, self.num_head*2, 1, 2, True,
                                         mode=mode, REF=self.REF),
                    IMTTB(self.ch_level3, self.ch_level3, self.num_head*2, 1, 2, True,
                                         mode=mode, REF=self.REF),
                ])
            ])

    def UpDown_List(self, mode='Down'):
        if mode == 'Down':
            return nn.ModuleList([
                units.UpDownSample(self.ch_level, self.ch_level1, 0.5),
                units.UpDownSample(self.ch_level1, self.ch_level2, 0.5),
                units.UpDownSample(self.ch_level2, self.ch_level3, 0.5)
            ])
        elif mode == 'Up':
            return nn.ModuleList([
                units.UpDownSample(self.ch_level1, self.ch_level, 2),
                units.UpDownSample(self.ch_level2, self.ch_level1, 2),
                units.UpDownSample(self.ch_level3, self.ch_level2, 2)
            ])

    class DetailRefineModule(nn.Module):
        def __init__(self, channels, k_size, groups):
            super().__init__()
            self.res = nn.Sequential(
                self.DetailBlock(channels, k_size, groups),
                self.DetailBlock(channels, k_size, groups),
                self.DetailBlock(channels, k_size, groups),
            )

        class DetailBlock(nn.Module):
            def __init__(self, channels, k_size, groups):
                super().__init__()
                self.pdwconv = nn.Sequential(
                    nn.Conv2d(channels, channels, kernel_size=k_size, padding=(k_size // 2), bias=False, groups=groups),
                    nn.Conv2d(channels, channels, kernel_size=1, bias=False)
                )
                self.ca = units.ChannelAttention()
                self.act = nn.LeakyReLU(inplace=True)

            def forward(self, x):
                x = self.pdwconv(x)
                x = x + self.ca(x)
                x = self.act(x)
                return x

        def forward(self, x):
            return self.res(x)

    def UV_coding(self, UV):
        UV1 = self.UVdown[0](UV)
        UV2 = self.UVdown[1](UV1)
        UV3 = self.UVdown[2](UV2)

        enc_UV0 = UV
        for blk in self.enc_UV[0]:
            enc_UV0 = blk(enc_UV0)

        enc_UV1 = self.enc_UVdown[0](enc_UV0)
        enc_UV1 = enc_UV1 + UV1
        for blk in self.enc_UV[1]:
            enc_UV1 = blk(enc_UV1)

        enc_UV2 = self.enc_UVdown[1](enc_UV1)
        enc_UV2 = enc_UV2 + UV2
        for blk in self.enc_UV[2]:
            enc_UV2 = blk(enc_UV2)

        enc_UV3 = self.enc_UVdown[2](enc_UV2)
        enc_UV3 = enc_UV3 + UV3
        for blk in self.enc_UV[3]:
            enc_UV3 = blk(enc_UV3)

        for blk in self.dec_UV[3]:
            dec3 = blk(enc_UV3)

        dec2 = enc_UV2 + self.dec_UVup[2](dec3)
        for blk in self.dec_UV[2]:
            dec2 = blk(dec2)

        dec1 = enc_UV1 + self.dec_UVup[1](dec2)
        for blk in self.dec_UV[1]:
            dec1 = blk(dec1)

        dec0 = enc_UV0 + self.dec_UVup[0](dec1)
        for blk in self.dec_UV[0]:
            dec0 = blk(dec0)


        dec1_up = self.dec_UVup[0](dec1)
        dec2_up = self.dec_UVup[0](self.dec_UVup[1](dec2))
        dec3_up = self.dec_UVup[0](self.dec_UVup[1](self.dec_UVup[2](dec3)))
        dec_out = self.AFFM([dec0, dec1_up, dec2_up, dec3_up])

        return dec_out

    def Y_coding(self, Yc, Ym):

        if Ym is not None:
            Ym1 = self.Ydown[0](Ym)
            Ym2 = self.Ydown[1](Ym1)
            Ym3 = self.Ydown[2](Ym2)

            Ye, Ye1, Ye2, Ye3 = Ym, Ym1, Ym2, Ym3
            Yd, Yd1, Yd2, Yd3 = Ye, Ye1, Ye2, Ye3

        # Encoding
        enc_Y0 = Yc
        for blk in self.enc_Y[0]:
            if self.REF:
                enc_Y0, Ye = blk(enc_Y0, Ye)
            else:
                enc_Y0 = blk(enc_Y0)

        enc_Y1 = self.enc_Ydown[0](enc_Y0)
        for blk in self.enc_Y[1]:
            if self.REF:
                enc_Y1, Ye1 = blk(enc_Y1, Ye1)
            else:
                enc_Y1 = blk(enc_Y1)

        enc_Y2 = self.enc_Ydown[1](enc_Y1)
        for blk in self.enc_Y[2]:
            if self.REF:
                enc_Y2, Ye2 = blk(enc_Y2, Ye2)
            else:
                enc_Y2 = blk(enc_Y2)

        enc_Y3 = self.enc_Ydown[2](enc_Y2)
        for blk in self.enc_Y[3]:
            if self.REF:
                enc_Y3, Ye3 = blk(enc_Y3, Ye3)
            else:
                enc_Y3 = blk(enc_Y3)

        # Decoding
        for blk in self.dec_Y[3]:
            if self.REF:
                dec3, Yd3 = blk(enc_Y3, Yd3)
            else:
                dec3 = blk(enc_Y3)

        dec2 = enc_Y2 + self.dec_Yup[2](dec3)
        for blk in self.dec_Y[2]:
            if self.REF:
                dec2, Yd2 = blk(dec2, Yd2)
            else:
                dec2 = blk(dec2)

        dec1 = enc_Y1 + self.dec_Yup[1](dec2)
        for blk in self.dec_Y[1]:
            if self.REF:
                dec1, Yd1 = blk(dec1, Yd1)
            else:
                dec1 = blk(dec1)

        dec0 = enc_Y0 + self.dec_Yup[0](dec1)
        for blk in self.dec_Y[0]:
            if self.REF:
                dec0, Yd = blk(dec0, Yd)
            else:
                dec0 = blk(dec0)

        dec1_up = self.dec_Yup[0](dec1)
        dec2_up = self.dec_Yup[0](self.dec_Yup[1](dec2))
        dec3_up = self.dec_Yup[0](self.dec_Yup[1](self.dec_Yup[2](dec3)))
        dec_out = self.AFFM([dec0, dec1_up, dec2_up, dec3_up])

        return dec_out

    def forward(self, UVc, Yc, Ym):
        for blk in self.feat[0]:
            UVc = blk(UVc)
        for blk in self.feat[1]:
            Yc = blk(Yc)
        if Ym is not None:
            Ym = self.feat[2](Ym)

        dec_UV = self.UV_coding(UVc)
        dec_Y = self.Y_coding(Yc, Ym)
        dec = self.squ(torch.cat([dec_Y, dec_UV], dim=1))
        h_feat = self.drm(dec)
        # C->3
        h_map = self.out(h_feat)

        Y, U, V = torch.split(h_map, 1, dim=1)
        R = Y + 1.14 * V
        G = Y - 0.39 * U - 0.58 * V
        B = Y + 2.03 * U
        h_map = torch.cat([R, G, B], dim=1)
        return h_map


# ----------------------------------------------------------------------------------------------------------------------
class ALICC(nn.Module):
    def __init__(self, Ch_img=3, Channels=32, state='Test', REF=True, tests=False):
        super().__init__()
        self.state = state
        self.tests = tests
        self.REF = REF
        self.WPENet = WPENet(ch_img=Ch_img, channels=Channels)
        # self.RDRNet = RefRectifyNet(ch_img=Ch_img, ch_level=Channels, REF=REF)
        self.RDRNet = RefRectifyNet(ch_img=Ch_img, ch_level=Channels, REF=REF)
    def state_train(self, color_img, mono_img=None, gt_img=None, mode='Pretrain'):
        #AdditivityRectifyNet
        color_noise = self.WPENet(color_img)
        color = color_img - color_noise

        # RGB => YUV
        R, G, B = torch.split(color, 1, dim=1)
        Uc = -0.147 * R - 0.289 * G + 0.436 * B
        Vc = 0.615 * R - 0.515 * G - 0.100 * B
        UVc = torch.cat([Uc, Vc], dim=1)
        Yc = 0.299 * R + 0.587 * G + 0.114 * B

        if mono_img is not None:
            Ym = torch.unsqueeze(mono_img[:, 0, :, :], dim=1)
        else:
            Ym = mono_img
        # Degradation Rectify
        if mode == 'Pretrain':
            h_map = self.RDRNet(UVc, Yc, Ym)
            restored_img = color * h_map
            return restored_img, h_map, color, color_noise

    def state_test(self, color_img, mono_img=None):
        # Denoising
        color_noise = self.WPENet(color_img)
        color = color_img - color_noise

        # RGB => YCbCr
        R, G, B = torch.split(color, 1, dim=1)
        Uc = -0.147 * R - 0.289 * G + 0.436 * B
        Vc = 0.615 * R - 0.515 * G - 0.100 * B
        UVc = torch.cat([Uc, Vc], dim=1)
        Yc = 0.299 * R + 0.587 * G + 0.114 * B

        Ym = torch.unsqueeze(mono_img[:, 0, :, :], dim=1)

        h_map = self.RDRNet(UVc, Yc, Ym)
        restored_img = color * h_map
        return restored_img, h_map, color, color_noise

    def forward(self, color_img, mono_img=None, gt_img=None, mode='Pretrain'):
        if self.tests:
            if self.REF:
                mono_img = color_img
            gt_img = color_img
        if self.REF is not True: #REF判断是否有AFM的输入
            mono_img = None
        if self.state == 'Train':
            return self.state_train(color_img=color_img, mono_img=mono_img, gt_img=gt_img, mode=mode)
        else:
            return self.state_test(color_img=color_img, mono_img=mono_img)


# ======================================================================================================================
if __name__ == '__main__':
    def model_complex(model, input_shape):
        macs, params = get_model_complexity_info(model, input_shape, as_strings=True,
                                                 print_per_layer_stat=False, verbose=True)
        print(f'====> Number of Model Params: {params}')
        print(f'====> Computational complexity: {macs}')


    model = ALICC(3, state='Train', REF=True, tests=True)
    model_complex(model, (3, 256, 256))
