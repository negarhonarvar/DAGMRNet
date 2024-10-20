import torch
import torch.nn as nn
import fastmri
import torch.nn.functional as F
from fastmri.data import transforms
from typing import List
from importlib import import_module
import os
#########################################################################
# -------------------------------  DAGL ----------------------------------

def default_conv(in_channels, out_channels, kernel_size = 3, stride=1, bias=True):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size // 2), stride=stride, bias=bias)


class MeanShift(nn.Conv2d):
    def __init__(
        self, in_channels, rgb_range, sign=-1):
        super(MeanShift, self).__init__(in_channels, in_channels, kernel_size=1)
        for p in self.parameters():
            p.requires_grad = False


class ResBlock(nn.Module):
    def __init__(self, conv, n_feats, kernel_size = 3, bias=True, act=nn.PReLU(), res_scale=0.1):
        super(ResBlock, self).__init__()
        m = [conv(n_feats, n_feats, kernel_size, bias=bias)]
        m.append(act)
        m.append(conv(n_feats, n_feats, kernel_size, bias=bias))
        self.body = nn.Sequential(*m)
        self.res_scale = res_scale

    def forward(self, x):
        res = self.body(x).mul(self.res_scale)
        res += x
        return res


def same_padding(images, ksizes, strides, rates):
    assert len(images.size()) == 4
    batch_size, channel, rows, cols = images.size()
    out_rows = (rows + strides[0] - 1) // strides[0]
    out_cols = (cols + strides[1] - 1) // strides[1]
    effective_k_row = (ksizes[0] - 1) * rates[0] + 1
    effective_k_col = (ksizes[1] - 1) * rates[1] + 1
    padding_rows = max(0, (out_rows - 1) * strides[0] + effective_k_row - rows)
    padding_cols = max(0, (out_cols - 1) * strides[1] + effective_k_col - cols)
    padding_top = int(padding_rows / 2.)
    padding_left = int(padding_cols / 2.)
    padding_bottom = padding_rows - padding_top
    padding_right = padding_cols - padding_left
    paddings = (padding_left, padding_right, padding_top, padding_bottom)
    images = torch.nn.ZeroPad2d(paddings)(images)
    return images, paddings


def extract_image_patches(images, ksizes, strides, rates, padding='same'):
    assert len(images.size()) == 4
    assert padding in ['same', 'valid']
    paddings = (0, 0, 0, 0)

    if padding == 'same':
        images, paddings = same_padding(images, ksizes, strides, rates)
    elif padding == 'valid':
        pass

    unfold = torch.nn.Unfold(kernel_size=ksizes, padding=0, stride=strides)
    patches = unfold(images)
    return patches, paddings


"""
Graph model
"""
class CE(nn.Module):
    def __init__(self, ksize=7, stride_1=1 # 4
                 , stride_2=1, softmax_scale=10, in_channels=4, inter_channels=2): #  in_channels=64, inter_channels=16
        super(CE, self).__init__()
        self.ksize = ksize
        self.stride_1 = stride_1
        self.stride_2 = stride_2
        self.softmax_scale = softmax_scale
        self.inter_channels = inter_channels
        self.in_channels = in_channels

        self.g = nn.Conv2d(in_channels=self.in_channels, out_channels=self.inter_channels, kernel_size=3, stride=1, padding=1)
        self.W = nn.Conv2d(in_channels=self.inter_channels, out_channels=self.in_channels, kernel_size=1, stride=1, padding=0)
        self.theta = nn.Conv2d(in_channels=self.in_channels, out_channels=self.inter_channels, kernel_size=1, stride=1, padding=0)


        self.fc1 = nn.Linear(ksize ** 2 * inter_channels, (ksize ** 2 * inter_channels) // 4)
        self.fc2 = nn.Linear(ksize ** 2 * inter_channels, (ksize ** 2 * inter_channels) // 4)

    def forward(self, x):
        b1 = self.g(x)
        b2 = self.theta(x)

        patch_28, _ = extract_image_patches(b1, ksizes=[self.ksize, self.ksize], strides=[self.stride_1, self.stride_1], rates=[1, 1], padding='same')
        patch_28 = patch_28.view(b1.size(0), self.inter_channels, self.ksize, self.ksize, -1)
        patch_28 = patch_28.permute(0, 4, 1, 2, 3)

        patch_112, _ = extract_image_patches(b2, ksizes=[self.ksize, self.ksize], strides=[self.stride_2, self.stride_2], rates=[1, 1], padding='same')
        patch_112 = patch_112.view(b2.size(0), self.inter_channels, self.ksize, self.ksize, -1)
        patch_112 = patch_112.permute(0, 4, 1, 2, 3)
        

        # Before returning from the CE class, we will upsample the output of the patch 
        # extraction to match the spatial dimensions of the input x

        upsampled_patch_28 = F.interpolate(patch_28.mean(dim=1), size=(x.size(2), x.size(3)), mode='bilinear', align_corners=False)
        upsampled_patch_112 = F.interpolate(patch_112.mean(dim=1), size=(x.size(2), x.size(3)), mode='bilinear', align_corners=False)

        return upsampled_patch_28 + upsampled_patch_112  # patch_28.mean(dim=1) + patch_112.mean(dim=1)


class CES(nn.Module):
    def __init__(self, in_channels, num=4):
        super(CES, self).__init__()
        RBS1 = [ResBlock(default_conv, n_feats=in_channels, kernel_size=3, act=nn.PReLU(), res_scale=1) for _ in range(num)]
        self.RBS1 = nn.Sequential(*RBS1)
        RBS2 = [ResBlock(default_conv, n_feats=in_channels, kernel_size=3, act=nn.PReLU(), res_scale=1) for _ in range(num)]
        self.RBS2 = nn.Sequential(*RBS2)

        # stage 1 (4 heads)
        self.c1_1 = CE(in_channels=in_channels)
        self.c1_2 = CE(in_channels=in_channels)
        # self.c1_3 = CE(in_channels=in_channels)
        # self.c1_4 = CE(in_channels=in_channels)
        # self.c1_c = nn.Conv2d(in_channels, in_channels, 1, 1, 0)

        # # stage 2 (4 heads)
        # self.c2_1 = CE(in_channels=in_channels)
        # self.c2_2 = CE(in_channels=in_channels)
        # self.c2_3 = CE(in_channels=in_channels)
        # self.c2_4 = CE(in_channels=in_channels)
        # self.c2_c = nn.Conv2d(in_channels, in_channels, 1, 1, 0)

        # # stage 3 (4 heads)
        # self.c3_1 = CE(in_channels=in_channels)
        # self.c3_2 = CE(in_channels=in_channels)
        # self.c3_3 = CE(in_channels=in_channels)
        # self.c3_4 = CE(in_channels=in_channels)
        # self.c3_c = nn.Conv2d(in_channels, in_channels, 1, 1, 0)

    def forward(self, x):
        # Stage 1 (4-head)
        # print(self.c1_1(x).shape, self.c1_2(x).shape, self.c1_3(x).shape, self.c1_4(x).shape)
        # print(x.shape)
        # out = self.c1_c(torch.cat((self.c1_1(x), self.c1_2(x), self.c1_3(x), self.c1_4(x)), dim=1)) + x
        out = (torch.cat((self.c1_1(x),self.c1_2(x)),dim = 1)) + x
        out = self.RBS1(out)

        # # Stage 2 (4-head)
        # out = self.c2_c(torch.cat((self.c2_1(out), self.c2_2(out), self.c2_3(out), self.c2_4(out)), dim=1)) + out
        # out = self.RBS2(out)

        # # Stage 3 (4-head)
        # out = self.c3_c(torch.cat((self.c3_1(out), self.c3_2(out), self.c3_3(out), self.c3_4(out)), dim=1)) + out
        return out


class RR(nn.Module):
    def __init__(self, args, conv=default_conv):
        super(RR, self).__init__()
        # Define basic settings for grayscale input
        n_resblocks = args['n_resblocks']
        n_feats = args['n_feats']
        kernel_size = 3
        res_scale = args['res_scale']

        # Head module (input channels for grayscale images is 1)
        m_head = [conv(1, n_feats, kernel_size)]  # Adjust input channel to 1 for grayscale

        # Body module
        m_body = [
            ResBlock(conv, n_feats, kernel_size, res_scale=res_scale) for _ in range(n_resblocks // 2)
        ]
        m_body.append(CES(n_feats))
        for i in range(n_resblocks // 2):
            m_body.append(ResBlock(conv, n_feats, kernel_size, res_scale=res_scale))

        m_body.append(conv(n_feats, n_feats, kernel_size))

        m_tail = [conv(n_feats, 2, kernel_size)]  # Adjust output channel to 2 for not causing error in chan_complex_to_last_dim

        self.head = nn.Sequential(*m_head)
        self.body = nn.Sequential(*m_body)
        self.tail = nn.Sequential(*m_tail)

    def forward(self, x):
        # Pass the input through the network
        res = self.head(x)
        res = self.body(res)
        res = self.tail(res)
        return x + res  # Residual connection


class Model(nn.Module):
    def __init__(self, args, ckp):
        super(Model, self).__init__()
        print('Building GNN model for grayscale images...')
        
        self.model = RR(args)

    def forward(self, x):
        return self.model(x)

    def save(self, apath, epoch, is_best=False):
        torch.save(self.model.state_dict(), os.path.join(apath, 'model_latest.pt'))
        if is_best:
            torch.save(self.model.state_dict(), os.path.join(apath, 'model_best.pt'))

    def load(self, apath, pre_train='.', resume=-1):
        if resume == -1:
            self.model.load_state_dict(torch.load(os.path.join(apath, 'model_latest.pt')))
        elif pre_train != '.':
            print('Loading model from {}'.format(pre_train))
            self.model.load_state_dict(torch.load(pre_train))