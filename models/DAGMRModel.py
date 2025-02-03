import torch
import torch.nn as nn
import fastmri
import torch.nn.functional as F
from fastmri.data import transforms
from typing import List
from importlib import import_module
import os
import math
from models.Utilities import test_x8
import torch.utils.checkpoint as checkpoint

#########################################################################
# -------------------------------  DAGMRNET ----------------------------------

def default_conv(in_channels, out_channels, kernel_size = 3, stride=1, bias=True):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size // 2), stride=stride, bias=bias)


class ResBlock(nn.Module):
    def __init__(
        self, conv, n_feats, kernel_size,
        bias=True, bn=False, act=nn.PReLU(), res_scale=0.1):

        super(ResBlock, self).__init__()
        m = []
        for i in range(2):
            m.append(conv(n_feats, n_feats, kernel_size, bias=bias))
            if bn:
                m.append(nn.BatchNorm2d(n_feats))
            if i == 0:
                m.append(act)

        self.body = nn.Sequential(*m)
        self.res_scale = res_scale

    def forward(self, x):
        res = self.body(x).mul(self.res_scale)
        res += x

        return res

"""
Padding functions
"""
def same_padding(images, ksizes, strides, rates):
    assert len(images.size()) == 4
    batch_size, channel, rows, cols = images.size()
    out_rows = (rows + strides[0] - 1) // strides[0]
    out_cols = (cols + strides[1] - 1) // strides[1]
    effective_k_row = (ksizes[0] - 1) * rates[0] + 1
    effective_k_col = (ksizes[1] - 1) * rates[1] + 1
    padding_rows = max(0, (out_rows - 1) * strides[0] + effective_k_row - rows)
    padding_cols = max(0, (out_cols - 1) * strides[1] + effective_k_col - cols)
    # Pad the input
    padding_top = int(padding_rows / 2.)
    padding_left = int(padding_cols / 2.)
    padding_bottom = padding_rows - padding_top
    padding_right = padding_cols - padding_left
    paddings = (padding_left, padding_right, padding_top, padding_bottom)
    images = torch.nn.ZeroPad2d(paddings)(images)
    return images, paddings


def ExtractImagePatches(images, ksizes, strides, rates, padding='same'):
    """
    Extract patches from images and put them in the C output dimension.
    :param padding:
    :param images: [batch, channels, in_rows, in_cols]. A 4-D Tensor with shape
    :param ksizes: [ksize_rows, ksize_cols]. The size of the sliding window for
     each dimension of images
    :param strides: [stride_rows, stride_cols]
    :param rates: [dilation_rows, dilation_cols]
    :return: A Tensor
    """
    assert len(images.size()) == 4
    assert padding in ['same', 'valid']
    paddings = (0, 0, 0, 0)

    if padding == 'same':
        images, paddings = same_padding(images, ksizes, strides, rates)
    elif padding == 'valid':
        pass
    else:
        raise NotImplementedError('Unsupported padding type: {}.\
                Only "same" or "valid" are supported.'.format(padding))

    unfold = torch.nn.Unfold(kernel_size=ksizes,
                             padding=0,
                             stride=strides)
    patches = unfold(images)
    return patches, paddings


"""
Graph model
"""
class DynamicAttentionMechanism(nn.Module):
    def __init__(self, ksize=7, stride_1=4, stride_2=1, softmax_scale=10,shape=64 ,p_len=64,in_channels=48
                 , inter_channels=16,use_multiple_size=False,use_topk=False,add_SE=False,num_edge = 50):
        super(DynamicAttentionMechanism, self).__init__()
        self.ksize = ksize
        self.shape=shape
        self.p_len=p_len
        self.stride_1 = stride_1
        self.stride_2 = stride_2
        self.softmax_scale = softmax_scale
        self.inter_channels = inter_channels
        self.in_channels = in_channels
        self.use_multiple_size=use_multiple_size
        self.use_topk=use_topk
        self.add_SE=add_SE
        self.num_edge = num_edge
        
        # feature compression layer with a kernel size of 3
        self.g = nn.Conv2d(in_channels=self.in_channels, out_channels=self.inter_channels, kernel_size=3, stride=1,
                           padding=1)
        
        # pointwise transformation, adjusting the number of channels without affecting the spatial dimensions.
        self.W = nn.Conv2d(in_channels=self.inter_channels, out_channels=self.in_channels, kernel_size=1, stride=1,
                           padding=0)
        
        #  transforming the input feature maps into a lower dimensional space , similar to g
        self.theta = nn.Conv2d(in_channels=self.in_channels, out_channels=self.inter_channels, kernel_size=1, stride=1,
                               padding=0)
        
        # fully connected layer with relu operating on flattened patch-level features (ksize is kernel size).
        # The output size is one-fourth of the input size, effectively reducing the feature dimension.
        self.fc1 = nn.Sequential(
            nn.Linear(in_features=ksize**2*inter_channels,out_features=(ksize**2*inter_channels)//4),
            nn.ReLU()
        )
        self.fc2 = nn.Sequential(
            nn.Linear(in_features=ksize**2*inter_channels,out_features=(ksize**2*inter_channels)//4),
            nn.ReLU()
        )

        # generating threshold and bias terms for soft thresholding mechanism
        self.thr_conv = nn.Conv2d(in_channels=in_channels,out_channels=1,kernel_size=ksize,stride=stride_1,padding=0)
        self.bias_conv = nn.Conv2d(in_channels=in_channels,out_channels=1,kernel_size=ksize,stride=stride_1,padding=0)

    def forward(self, b):
        b1 = self.g(b)
        b2 = self.theta(b)
        b3 = b1

        raw_int_bs = list(b1.size())  # b*c*h*w
        b4, _ = same_padding(b,[self.ksize,self.ksize],[self.stride_1,self.stride_1],[1,1])
        soft_thr = self.thr_conv(b4).view(raw_int_bs[0],-1)
        soft_bias = self.bias_conv(b4).view(raw_int_bs[0],-1)
        patch_28, paddings_28 = ExtractImagePatches(b1, ksizes=[self.ksize, self.ksize],
                                                      strides=[self.stride_1, self.stride_1],
                                                      rates=[1, 1],
                                                      padding='same')
        patch_28 = patch_28.view(raw_int_bs[0], raw_int_bs[1], self.ksize, self.ksize, -1)
        patch_28 = patch_28.permute(0, 4, 1, 2, 3)
        patch_28_group = torch.split(patch_28, 1, dim=0)

        patch_112, paddings_112 = ExtractImagePatches(b2, ksizes=[self.ksize, self.ksize],
                                                        strides=[self.stride_2, self.stride_2],
                                                        rates=[1, 1],
                                                        padding='same')

        patch_112 = patch_112.view(raw_int_bs[0], raw_int_bs[1], self.ksize, self.ksize, -1)
        patch_112 = patch_112.permute(0, 4, 1, 2, 3)
        patch_112_group = torch.split(patch_112, 1, dim=0)

        patch_112_2, paddings_112_2 = ExtractImagePatches(b3, ksizes=[self.ksize, self.ksize],
                                                        strides=[self.stride_2, self.stride_2],
                                                        rates=[1, 1],
                                                        padding='same')

        patch_112_2 = patch_112_2.view(raw_int_bs[0], raw_int_bs[1], self.ksize, self.ksize, -1)
        patch_112_2 = patch_112_2.permute(0, 4, 1, 2, 3)
        patch_112_group_2 = torch.split(patch_112_2, 1, dim=0)
        y = []
        w, h = raw_int_bs[2], raw_int_bs[3]
        _, paddings = same_padding(b3[0,0].unsqueeze(0).unsqueeze(0), [self.ksize, self.ksize], [self.stride_2, self.stride_2], [1, 1])
        
        for xi, wi,pi,thr,bias in zip(patch_112_group_2, patch_28_group, patch_112_group,soft_thr,soft_bias):
            c_s = pi.shape[2]
            k_s = wi[0].shape[2]
            wi = self.fc1(wi.view(wi.shape[1],-1))
            xi = self.fc2(xi.view(xi.shape[1],-1)).permute(1,0)
            score_map = torch.matmul(wi,xi)
            score_map = score_map.view(1, score_map.shape[0], math.ceil(w / self.stride_2),
                                       math.ceil(h / self.stride_2))
            b_s, l_s, h_s, w_s = score_map.shape
            yi = score_map.view(l_s, -1)
            
            mask = F.relu(yi-yi.mean(dim=1,keepdim=True)*thr.unsqueeze(1)+bias.unsqueeze(1))
            mask_b = (mask!=0.).float()

            yi = yi * mask
            yi = F.softmax(yi * self.softmax_scale, dim=1)
            yi = yi * mask_b
            
            pi = pi.view(h_s * w_s, -1)
            yi = torch.mm(yi, pi)
            yi = yi.view(b_s, l_s, c_s, k_s, k_s)[0]
            zi = yi.view(1, l_s, -1).permute(0, 2, 1)
            zi = torch.nn.functional.fold(zi, (raw_int_bs[2], raw_int_bs[3]), (self.ksize, self.ksize), padding=paddings[0], stride=self.stride_1)
            inp = torch.ones_like(zi)
            inp_unf = torch.nn.functional.unfold(inp, (self.ksize, self.ksize), padding=paddings[0], stride=self.stride_1)
            out_mask = torch.nn.functional.fold(inp_unf, (raw_int_bs[2], raw_int_bs[3]), (self.ksize, self.ksize), padding=paddings[0], stride=self.stride_1)
            out_mask += (out_mask==0.).float()
            zi = zi / out_mask
            y.append(zi)
        y = torch.cat(y, dim=0)
        return y

class M_GFAM(nn.Module):
    def __init__(self, in_channels, num=4):
        super(M_GFAM, self).__init__()
        RBS1 = [ResBlock(default_conv, n_feats=in_channels, kernel_size=3, act=nn.PReLU(), res_scale=1
                         ) for _ in range(num)]
        self.RBS1 = nn.Sequential(*RBS1)
        RBS2 = [ResBlock(default_conv, n_feats=in_channels, kernel_size=3, act=nn.PReLU(), res_scale=1
                         ) for _ in range(num)]
        self.RBS2 = nn.Sequential(*RBS2)

        # stage 1 (3 heads)
        print("in_channels in CES :" , in_channels)
        self.c1_1 = DynamicAttentionMechanism(in_channels=in_channels)
        self.c1_2 = DynamicAttentionMechanism(in_channels=in_channels)
        self.c1_3 = DynamicAttentionMechanism(in_channels=in_channels)
        self.c1_c = nn.Conv2d(in_channels, in_channels, 1, 1, 0)

        # stage 2 (3 heads)
        self.c2_1 = DynamicAttentionMechanism(in_channels=in_channels)
        self.c2_2 = DynamicAttentionMechanism(in_channels=in_channels)
        self.c2_3 = DynamicAttentionMechanism(in_channels=in_channels)
        self.c2_c = nn.Conv2d(in_channels, in_channels, 1, 1, 0)

        # stage 3 (3 heads)
        self.c3_1 = DynamicAttentionMechanism(in_channels=in_channels)
        self.c3_2 = DynamicAttentionMechanism(in_channels=in_channels)
        self.c3_3 = DynamicAttentionMechanism(in_channels=in_channels)
        self.c3_c = nn.Conv2d(in_channels, in_channels, 1, 1, 0)

    def forward(self, x):

        # Stage 1 (3-head)
        out = self.c1_c(torch.cat((self.c1_1(x), self.c1_2(x), self.c1_3(x)), dim=1)) + x
        out = self.RBS1(out)

        # Stage 2 (3-head)
        out = self.c2_c(torch.cat((self.c2_1(out), self.c2_2(out), self.c2_3(out)), dim=1)) + out
        out = self.RBS2(out)


        # # Stage 3 (3-head)
        out = self.c3_c(torch.cat((self.c3_1(out), self.c3_2(out), self.c3_3(out)), dim=1)) + out
        return out


class DAGMR(nn.Module):
    def __init__(self, args, conv=default_conv):
        super(DAGMR, self).__init__()
        # Define basic settings for grayscale input
        n_resblocks = args['n_resblocks']
        n_feats = args['n_feats']
        kernel_size = 3
        res_scale = args['res_scale']
        # print(" res and feats and scale : " , n_resblocks,n_feats,res_scale)
        # Head module (input channels for grayscale images is 1)
        m_head = [conv(2, n_feats, kernel_size)]  # Adjust input channel to 1 for grayscale

        # Body module
        m_body = [
            ResBlock(conv, n_feats, kernel_size, res_scale=res_scale
                     )for _ in range(n_resblocks // 2)
        ]
        m_body.append(M_GFAM(n_feats))
        for i in range(n_resblocks // 2):
            m_body.append(ResBlock(conv, n_feats, kernel_size, res_scale=res_scale))

        m_body.append(conv(n_feats, n_feats, kernel_size))

        m_tail = [conv(n_feats, 2, kernel_size)] 
        self.head = nn.Sequential(*m_head)
        self.body = nn.Sequential(*m_body)
        self.tail = nn.Sequential(*m_tail)

    def forward(self, x):
        # Pass the input through the network
        # print("x shape :" , x.shape)
        original_input = x.clone()
        res = self.head(x)
        res = self.body(res)
        res = self.tail(res)
        # Check if the residual addition is functioning correctly
        output = original_input + res  # Residual connection
        print("RR Output Shape:", output.shape)
        print("RR Output Mean:", output.mean().item())
        # print(" RR forward output : ", (x + res).shape)
        return x + res  # Residual connection
    
    def load_state_dict(self, state_dict, strict=True):
        own_state = self.state_dict()
        for name, param in state_dict.items():
            if name in own_state:
                if isinstance(param, nn.Parameter):
                    param = param.data
                try:
                    own_state[name].copy_(param)
                except Exception:
                    if name.find('tail') == -1:
                        raise RuntimeError('While copying the parameter named {}, '
                                           'whose dimensions in the model are {} and '
                                           'whose dimensions in the checkpoint are {}.'
                                           .format(name, own_state[name].size(), param.size()))
            elif strict:
                if name.find('tail') == -1:
                    raise KeyError('unexpected key "{}" in state_dict'
                                   .format(name))

def make_model(args, parent=False):
    return DAGMR(args)

class ReconModel(nn.Module):
    def __init__(self, args, ckp):
        super(ReconModel, self).__init__()
        print('Building GNN model for grayscale images...')
        self.scale = args.scale
        self.idx_scale = 0
        self.self_ensemble = args.self_ensemble
        self.chop = args.chop
        self.n_GPUs = args.n_GPUs
        self.save_models = args.save_models
        module = import_module('model.' + args.model.lower())
        self.model = module.make_model(args)

        self.load(
            ckp.dir,
            pre_train=args.pre_train,
            resume=args.resume,
            cpu=args.cpu
        )
        if args.print_model:
            print(self.model)

    def forward(self, x):
        self.ensemble=False
        self.idx_scale = 0
        target = self.get_model()
        if hasattr(target, 'set_scale'):
            target.set_scale(self.idx_scale)
        if self.chop and not self.training:
            with torch.no_grad():  # this can save much memory
                Out = self.forward_chop(x)
            return Out
        else:
            return self.model(x)
        
    def get_model(self):
        if self.n_GPUs == 1:
            return self.model
        else:
            return self.model.module

    def state_dict(self, **kwargs):
        target = self.get_model()
        return target.state_dict(**kwargs)

    def save(self, checkpoint_dir, epoch, is_best=False):
        torch.save(self.model.state_dict(), os.path.join(checkpoint_dir, 'model_latest.pt'))
        if is_best:
            torch.save(self.model.state_dict(), os.path.join(checkpoint_dir, 'model_best.pt'))

    def load(self, pre_trained_model, pre_train='.', resume=-1):
        pre_trained_model = 'D:/Paper/codes/PromptMR/trainExamples/cmrxrecon/pretrained/model.pt'
        if resume == -1:
            self.model.load_state_dict(torch.load(os.path.join(pre_trained_model, 'model_latest.pt')))
        elif pre_train != '.':
            print('Loading model from {}'.format(pre_train))
            self.model.load_state_dict(torch.load(pre_train))

    def forward_chop(self, x, shave=10, min_size=10000): # shave=10, min_size=10000
        # print("forward chop mechanism")
        scale = self.scale[self.idx_scale]
        n_GPUs = min(self.n_GPUs, 4)
        b, c, h, w = x.size()
        shave_scale = 4
        # max shave size
        shave_size_max = 24
        # get half size of the hight and width
        h_half, w_half = h // 2, w // 2
        # mod
        mod_h, mod_w = h_half // shave_scale, w_half // shave_scale
        # ditermine midsize along height and width directions
        h_size = mod_h * shave_scale + shave_size_max
        w_size = mod_w * shave_scale + shave_size_max
        # h_size, w_size = h_half + shave, w_half + shave
        ###############################################
        # h_size, w_size = adaptive_shave(h, w)
        lr_list = [
            x[:, :, 0:h_size, 0:w_size],
            x[:, :, 0:h_size, (w - w_size):w],
            x[:, :, (h - h_size):h, 0:w_size],
            x[:, :, (h - h_size):h, (w - w_size):w]]

        if w_size * h_size < min_size:
            sr_list = []
            for i in range(0, 4, n_GPUs):
                lr_batch = torch.cat(lr_list[i:(i + n_GPUs)], dim=0)
                if self.ensemble == False:
                    sr_batch = self.model(lr_batch)
                else:
                    sr_batch = test_x8(self.model, lr_batch)  # data aug
                sr_list.extend(sr_batch.chunk(n_GPUs, dim=0))
        else:
            sr_list = [
                self.forward_chop(patch, shave=shave, min_size=min_size) \
                for patch in lr_list
            ]

        h, w = scale * h, scale * w
        h_half, w_half = scale * h_half, scale * w_half
        h_size, w_size = scale * h_size, scale * w_size
        shave *= scale

        output = torch.tensor(x.data.new(b, c, h, w))
        output[:, :, 0:h_half, 0:w_half] \
            = sr_list[0][:, :, 0:h_half, 0:w_half]
        output[:, :, 0:h_half, w_half:w] \
            = sr_list[1][:, :, 0:h_half, (w_size - w + w_half):w_size]
        output[:, :, h_half:h, 0:w_half] \
            = sr_list[2][:, :, (h_size - h + h_half):h_size, 0:w_half]
        output[:, :, h_half:h, w_half:w] \
            = sr_list[3][:, :, (h_size - h + h_half):h_size, (w_size - w + w_half):w_size]

        return output

    def adaptive_shave(h, w):

        # corresponding to scaling factor of the downscaling and upscaling modules in the network
        shave_scale = 4
        # max shave size
        shave_size_max = 24
        # get half size of the hight and width
        h_half, w_half = h // 2, w // 2
        # mod
        mod_h, mod_w = h_half // shave_scale, w_half // shave_scale
        # ditermine midsize along height and width directions
        midsize_h = mod_h * shave_scale + shave_size_max
        midsize_w = mod_w * shave_scale + shave_size_max
        # print('midsize_h={}, midsize_w={}'.format(midsize_h, midsize_w))
        return midsize_h, midsize_w

    '''
    def adaptive_shave(self, h, w):
        shave_scale = 4
        shave_size_max = 12
        h_half, w_half = h // 2, w // 2
        mod_h, mod_w = h_half // shave_scale, w_half // shave_scale
        midsize_h = mod_h * shave_scale + shave_size_max
        midsize_w = mod_w * shave_scale + shave_size_max

        return midsize_h, midsize_w
    '''

    def forward_x8(self, x, forward_function):
        def _transform(v, op):
            if self.precision != 'single':
                v = v.float()

            v2np = v.data.cpu().numpy()
            if op == 'v':
                tfnp = v2np[:, :, :, ::-1].copy()
            elif op == 'h':
                tfnp = v2np[:, :, ::-1, :].copy()
            elif op == 't':
                tfnp = v2np.transpose((0, 1, 3, 2)).copy()

            if not self.cpu: ret = torch.Tensor(tfnp).cuda()
            if self.precision == 'half': ret = ret.half()

            return torch.tensor(ret)

        lr_list = [x]
        for tf in 'v', 'h', 't':
            lr_list.extend([_transform(t, tf) for t in lr_list])

        sr_list = [forward_function(aug) for aug in lr_list]
        for i in range(len(sr_list)):
            if i > 3:
                sr_list[i] = _transform(sr_list[i], 't')
            if i % 4 > 1:
                sr_list[i] = _transform(sr_list[i], 'h')
            if (i % 4) % 2 == 1:
                sr_list[i] = _transform(sr_list[i], 'v')

        output_cat = torch.cat(sr_list, dim=0)
        output = output_cat.mean(dim=0, keepdim=True)

        return output
