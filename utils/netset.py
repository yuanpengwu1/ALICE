import torch
import os
import math
from collections import OrderedDict
import pywt
from torch import Tensor
from torch.nn.functional import normalize
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from math import log
from natsort import natsorted
from glob import glob
import numpy as np



class NET:
    @staticmethod
    def freeze(model):
        for p in model.parameters():
            p.requires_grad=False

    @staticmethod
    def unfreeze(model):
        for p in model.parameters():
            p.requires_grad=True

    @staticmethod
    def is_frozen(model):
        x = [p.requires_grad for p in model.parameters()]
        return not all(x)

    @staticmethod
    def save_checkpoint(model_dir, state, session):
        epoch = state['epoch']
        model_out_path = os.path.join(model_dir,"model_epoch_{}_{}.pth".format(epoch,session))
        torch.save(state, model_out_path)

    @staticmethod
    def get_last_path(path, session):
        x = natsorted(glob(os.path.join(path, '*%s' % session)))[-1]
        return x

    @staticmethod
    def load_checkpoint(model, weights, strict=True):
        checkpoint = torch.load(weights)
        try:
            model.load_state_dict(checkpoint["state_dict"], strict=strict)
        except:
            state_dict = checkpoint["state_dict"]
            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                name = k[7:] # remove `module.`
                new_state_dict[name] = v
            model.load_state_dict(new_state_dict, strict=strict)

    @staticmethod
    def load_checkpoint_multigpu(model, weights):
        checkpoint = torch.load(weights)
        state_dict = checkpoint["state_dict"]
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:] # remove `module.`
            new_state_dict[name] = v
        model.load_state_dict(new_state_dict)

    @staticmethod
    def load_start_epoch(weights):
        checkpoint = torch.load(weights)
        epoch = checkpoint["epoch"]
        return epoch

    @staticmethod
    def load_optim(optimizer, weights):
        checkpoint = torch.load(weights)
        optimizer.load_state_dict(checkpoint['optimizer'])
        # for p in optimizer.param_groups: lr = p['lr']
        # return lr

    @staticmethod
    def print_model_parm_nums(model):
        total = sum([param.nelement() for param in model.parameters()])
        return total / 1e6


# ----------------------------------------------------------------------------------------------------------------------
class CharbonnierLoss(nn.Module):
    def __init__(self, eps=1e-3):
        super().__init__()
        self.eps = eps

    def forward(self, x, y):
        diff = x - y
        loss = torch.mean(torch.sqrt((diff * diff) + (self.eps * self.eps)))
        return loss


class EdgeLoss(nn.Module):
    def __init__(self):
        super().__init__()
        k = torch.Tensor([[.05, .25, .4, .25, .05]])
        self.kernel = torch.matmul(k.t(), k).unsqueeze(0).repeat(3, 1, 1, 1)
        if torch.cuda.is_available():
            self.kernel = self.kernel.cuda()
        self.loss = CharbonnierLoss()

    def conv_gauss(self, img):
        n_channels, _, kw, kh = self.kernel.shape
        img = F.pad(img, (kw // 2, kh // 2, kw // 2, kh // 2), mode='replicate')
        return F.conv2d(img, self.kernel, groups=n_channels)

    def laplacian_kernel(self, current):
        filtered = self.conv_gauss(current)  # filter
        down = filtered[:, :, ::2, ::2]  # downsample
        new_filter = torch.zeros_like(filtered)
        new_filter[:, :, ::2, ::2] = down * 4  # upsample
        filtered = self.conv_gauss(new_filter)  # filter
        diff = current - filtered
        return diff

    def forward(self, x, y):
        loss = self.loss(self.laplacian_kernel(x), self.laplacian_kernel(y))
        return loss


class PerceptualLoss(nn.Module):
    def __init__(self):
        super(PerceptualLoss, self).__init__()
        self.vgg19 = self.VGG19()

    class VGG19(nn.Module):
        def __init__(self):
            super().__init__()
            '''
             use vgg19 conv1_2, conv2_2, conv3_3 feature, before relu layer
            '''
            self.feature_list = [2, 7, 14]
            vgg19 = torchvision.models.vgg19(pretrained=True).cuda()
            self.model = nn.Sequential(*list(vgg19.features.children())[:self.feature_list[-1] + 1])

        def forward(self, x):
            x = (x - 0.5) / 0.5
            features = []
            for i, layer in enumerate(list(self.model)):
                x = layer(x)
                if i in self.feature_list:
                    features.append(x)

            return features

    def forward(self, x, y):
        weights = [1, 0.2, 0.04]
        features_fake = self.vgg19(x)
        features_real = self.vgg19(y)
        features_real_no_grad = [f_real.detach() for f_real in features_real]
        mse_loss = nn.MSELoss(reduction='elementwise_mean')
        loss = 0
        for i in range(len(features_real)):
            loss_i = mse_loss(features_fake[i], features_real_no_grad[i])
            loss = loss + loss_i * weights[i]
        return loss


class TVLoss(nn.Module):
    def __init__(self, TVLoss_weight=1):
        super(TVLoss, self).__init__()
        self.TVLoss_weight = TVLoss_weight

    def _tensor_size(self, t):
        return t.size()[1]*t.size()[2]*t.size()[3]

    def forward(self, x):
        batch_size = x.size()[0]
        h_x = x.size()[2]
        w_x = x.size()[3]
        count_h = self._tensor_size(x[:, :, 1:, :])
        count_w = self._tensor_size(x[:, :, :, 1:])
        h_tv = torch.pow((x[:, :, 1:, :] - x[:, :, :h_x - 1, :]), 2).sum()
        w_tv = torch.pow((x[:, :, :, 1:] - x[:, :, :, :w_x - 1]), 2).sum()
        return self.TVLoss_weight * 2 * (h_tv / count_h + w_tv / count_w) / batch_size
#-----------------------------------------------标准DWT dwtloss1---------------------------------------------------------
# def dwt_int(x):
#     """使用标准DWT对每个通道进行2D小波变换"""
#     batch_size, channels, height, width = x.shape
#     LL, LH, HL, HH = [], [], [], []
#
#     # 对每个通道单独进行DWT
#     for b in range(batch_size):
#         for c in range(channels):
#             coeffs2 = pywt.dwt2(x[b, c].detach().cpu().numpy(), 'haar')
#             ll, (lh, hl, hh) = coeffs2
#
#             # 将每个分量转换为张量并存储
#             LL.append(torch.tensor(ll).unsqueeze(0))
#             LH.append(torch.tensor(lh).unsqueeze(0))
#             HL.append(torch.tensor(hl).unsqueeze(0))
#             HH.append(torch.tensor(hh).unsqueeze(0))
#
#     # 合并回 [batch, channels/2, height/2, width/2] 的格式
#     x_LL = torch.stack(LL).view(batch_size, channels, height // 2, width // 2).to(x.device)
#     x_LH = torch.stack(LH).view(batch_size, channels, height // 2, width // 2).to(x.device)
#     x_HL = torch.stack(HL).view(batch_size, channels, height // 2, width // 2).to(x.device)
#     x_HH = torch.stack(HH).view(batch_size, channels, height // 2, width // 2).to(x.device)
#
#     return torch.cat([x_LL, x_LH, x_HL, x_HH], dim=1)
#
# class DWT(nn.Module):
#     def __init__(self):
#         super(DWT, self).__init__()
#         self.requires_grad = False
#
#     def forward(self, x):
#         return dwt_int(x)
#------------------------------------------高效近似DWT dwtloss2-----------------------------------------------------------
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

    return torch.cat([x_LL, x_LH, x_HL, x_HH], dim=1)

class DWT(nn.Module):
    def __init__(self):
        super(DWT, self).__init__()
        self.requires_grad = False

    def forward(self, x):
        return dwt_init(x)
# ----------------------------------------------------------------------------------------------------------------------
class criterion(nn.Module):
    def __init__(self, lambda1=0.1, lambda2=0.05):
        super().__init__()
        self.L1 = nn.L1Loss()
        self.L_edge = EdgeLoss()
        self.L_per = PerceptualLoss()
        self.L_tv = TVLoss()
        self.ld1 = lambda1
        self.ld2 = lambda2
        self.dwt = DWT()


    def forward(self, restored, target, color_dn, epoch):
        loss_l1 = self.L1(restored, target)
        loss_edge = self.L_edge(restored, target)
        loss_per = self.L_per(restored, target)
        loss_dwt = self.L1(self.dwt(color_dn), self.dwt(target)) #dwt loss
        loss_tvc = self.L_tv(color_dn)  #对
        loss_tvr = self.L_tv(restored)

        # 原始设置
        #return loss_l1 + self.ld1 * loss_edge + self.ld2 * loss_per + 0.1 * (loss_tvc + loss_tvr)

        # 只加上DWT损失,删去tvc_loss
        # return loss_l1 + self.ld1 * loss_edge + self.ld2 * loss_per + 0.1 * (loss_dwt + loss_tvr)

        # 在原始设置上加上了DWT损失
        return loss_l1 + self.ld1 * loss_edge + self.ld2 * loss_per + 0.1 * (loss_dwt + loss_tvc + loss_tvr )

        #消融
        # return loss_l1 + self.ld1 * loss_edge + 0.1 * (loss_dwt + loss_tvc + loss_tvr )
        # return loss_l1 + self.ld2 * loss_per + 0.1 * (loss_dwt + loss_tvc + loss_tvr )

        #return loss_char + self.lambda1 * loss_edge + self.lambda2 * loss_per + loss_tvc + loss_tvm

        #对感知损失进行消融
        # return loss_l1 + 0.1 * (loss_tvc + loss_tvr)#(0,0)
        # return loss_l1 + 0.01 * loss_per + 0.1 * (loss_tvc + loss_tvr)#(0.01,0)
        # return loss_l1 + 0.05 * loss_per + 0.1 * (loss_tvc + loss_tvr)#(0.05,0)
        # return loss_l1 + 0.1 * loss_per + 0.1 * (loss_tvc + loss_tvr)#(0.1,0)
        # return loss_l1 + 0.2 * loss_per + 0.1 * (loss_tvc + loss_tvr)#(0.2,0)

        # 对边缘损失进行消融
        # return loss_l1 + 0.05 * loss_per + 0.01 * loss_edge + 0.1 * (loss_tvc + loss_tvr)#(0.05,0.01)
        # return loss_l1 + 0.05 * loss_per + 0.05 * loss_edge + 0.1 * (loss_tvc + loss_tvr)  # (0.05,0.05)
        # return loss_l1 + 0.05 * loss_per + 0.1 * loss_edge + 0.1 * (loss_tvc + loss_tvr)  # (0.05,0.1)
        # return loss_l1 + 0.05 * loss_per + 0.2 * loss_edge + 0.1 * (loss_tvc + loss_tvr)  # (0.05,0.2)
