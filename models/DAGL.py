import torch
import torch.nn as nn
import fastmri
import torch.nn.functional as F
from fastmri.data import transforms
from typing import List
from importlib import import_module
import os
import math
from models import Utilities
from importlib import import_module
#########################################################################
# -------------------------------  DAGL ----------------------------------

def default_conv(in_channels, out_channels, kernel_size = 3, stride=1, bias=True):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size // 2), stride=stride, bias=bias)


# class MeanShift(nn.Conv2d):
#     def __init__(
#         self, in_channels, rgb_range, sign=-1):
#         super(MeanShift, self).__init__(in_channels, in_channels, kernel_size=1)
#         for p in self.parameters():
#             p.requires_grad = False


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
    padding_top = padding_rows // 2
    padding_left = padding_cols // 2
    padding_bottom = padding_rows - padding_top
    padding_right = padding_cols - padding_left
    
    paddings = (padding_left, padding_right, padding_top, padding_bottom)
    images = torch.nn.functional.pad(images, paddings)
    
    return images, paddings



def extract_image_patches(images, ksizes, strides, rates, padding='same'):
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
class CE(nn.Module):
    def __init__(self, ksize= 7 , stride_1=4, stride_2=4, softmax_scale=10,shape=64 ,p_len=64,in_channels=4 # ksize = 7 (mem)
                 , inter_channels=2,use_multiple_size=False,use_topk=False,add_SE=False,num_edge = 50):
        """
        This class is responsible for creating and processing patches from the tensor of undersampled, 
        noisy image data, and the code leverages both convolutional operations and attention mechanisms to refine the patches.
        param ksize : The class processes image patches of size ksize x ksize.
        param stride_1 and stride_2 : they define the step size for extracting patches in two different stages.
        param in_channels: Number of input channels for the convolution operations, 2 in this case (gray image with real and imaginery part)
        param inter_channels: Intermediate feature map size, reducing the dimensionality for efficiency in further calculations.
        param softmax_scale helps in controlling the sharpness of the softmax operation, which is used later to scale the computed similarities between patches.
        params g and theta : 2D convolutional layers that transform the input features to different representations used later for patch extraction and attention
        params fc1, fc2 : Fully connected layers that transform the flattened patch features and are used to compute similarities between patches. They reduce the dimensionality of the patch features for efficient comparison.
        params thr_conv, bias_conv: These layers compute the threshold and bias, which are applied to filter out patches that are not important for the reconstruction task
        
        """
        super(CE, self).__init__()
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

        self.g = nn.Conv2d(in_channels=self.in_channels, out_channels=self.inter_channels, kernel_size=3, stride=1,
                           padding=1)
        self.W = nn.Conv2d(in_channels=self.inter_channels, out_channels=self.in_channels, kernel_size=1, stride=1,
                           padding=0)
        self.theta = nn.Conv2d(in_channels=self.in_channels, out_channels=self.inter_channels, kernel_size=1, stride=1,
                               padding=0)
        
        """
        in_features : In the DAGL architecture, patches are extracted from an image, and these patches are represented in feature space. Specifically,
        in Section 3.2 of the paper, the feature extraction process involves creating patches with a size of C × 7 × 7,
        where C = 2 is the number of channels (real and imaginery parts), and the spatial dimensions of the patch are 7 × 7
        out_features : In the DAGL model, this compression happens in the graph learning stage to manage computational complexity and 
        reduce the number of parameters. By reducing the dimension by a factor of 4, 
        the model can retain essential information while avoiding the overhead associated with high-dimensional features.
        """
        self.fc1 = nn.Sequential(
            nn.Linear(in_features=ksize**2*inter_channels,out_features=(ksize**2*inter_channels)//4),
            nn.ReLU()
        )
        self.fc2 = nn.Sequential(
            nn.Linear(in_features=ksize**2*inter_channels,out_features=(ksize**2*inter_channels)//4),
            nn.ReLU()
        )
        self.thr_conv = nn.Conv2d(in_channels=in_channels,out_channels=1,kernel_size=self.ksize,stride=stride_1,padding=0) # stride = 4 , duo to shape mismatch
        self.bias_conv = nn.Conv2d(in_channels=in_channels,out_channels=1,kernel_size=self.ksize,stride=stride_1,padding=0)

    def forward(self, b):
        """
        b1 and b2: After applying convolutions (g, theta) to the input b, the feature maps b1 and b2 are generated. 
        These feature maps are essential for the patch extraction
        """
        b1 = self.g(b)
        b2 = self.theta(b)
        b3 = b1

        print(" b1 shape :" , b1.shape)
        print(" b2 shape :" , b2.shape)
        print(" b3 shape :" , b3.shape)

        raw_int_bs = list(b1.size())  # b*c*h*w
        b4, _ = same_padding(b,[self.ksize,self.ksize],[self.stride_1,self.stride_1],[1,1])
        print("b4 :", b4.shape)
        soft_thr = self.thr_conv(b4).view(raw_int_bs[0],-1)
        soft_bias = self.bias_conv(b4).view(raw_int_bs[0],-1)
        print("soft_thr :", soft_thr.shape)
        print("soft_bias :", soft_bias.shape)

        """
        params soft_thr and soft_bias: These tensors are computed from the convolution layers thr_conv and bias_conv. 
        They represent dynamic thresholds and biases that adapt based on the input data. 
        These values are later applied to control how much influence a patch has on the final output.
        params Patch_28 and Patch_112: These represent two sets of patches extracted from b1 and b2 using different strides.
        The stride sizes control how much the patches overlap with each other.
        final dim = [batch_size, number_of_patches, channels, ksize, ksize]
        """

        patch_28, paddings_28 = extract_image_patches(b1, ksizes=[self.ksize, self.ksize],
                                                      strides=[self.stride_1, self.stride_1],
                                                      rates=[1, 1],
                                                      padding='same')
        num_patches = patch_28.shape[-1]
        print(f"Number of patches1: {num_patches}")

        patch_28 = patch_28.view(raw_int_bs[0], raw_int_bs[1], self.ksize, self.ksize, -1)
        patch_28 = patch_28.permute(0, 4, 1, 2, 3) 
        patch_28_group = torch.split(patch_28, 1, dim=0)

        patch_112, paddings_112 = extract_image_patches(b2, ksizes=[self.ksize, self.ksize],
                                                        strides=[self.stride_2, self.stride_2],
                                                        rates=[1, 1],
                                                        padding='same')
        num_patches = patch_112.shape[-1]
        print(f"Number of patches: {num_patches}")
        
        patch_112 = patch_112.view(raw_int_bs[0], raw_int_bs[1], self.ksize, self.ksize, -1)
        patch_112 = patch_112.permute(0, 4, 1, 2, 3)
        patch_112_group = torch.split(patch_112, 1, dim=0)

        patch_112_2, paddings_112_2 = extract_image_patches(b3, ksizes=[self.ksize, self.ksize],
                                                        strides=[self.stride_2, self.stride_2],
                                                        rates=[1, 1],
                                                        padding='same')

        patch_112_2 = patch_112_2.view(raw_int_bs[0], raw_int_bs[1], self.ksize, self.ksize, -1)
        patch_112_2 = patch_112_2.permute(0, 4, 1, 2, 3)
        patch_112_group_2 = torch.split(patch_112_2, 1, dim=0)

        # Spliting the tensor into 2 chunks along the last dimension to reduce memory usage
        patch_28 = patch_28.chunk(chunks=2, dim=4)  
        patch_112 = patch_112.chunk(chunks=2, dim=4)
        
        # Patch Similarity Computation

        y = []
        w, h = raw_int_bs[2], raw_int_bs[3]
        _, paddings = same_padding(b3[0,0].unsqueeze(0).unsqueeze(0), [self.ksize, self.ksize], [self.stride_2, self.stride_2], [1, 1])
        for xi, wi,pi,thr,bias in zip(patch_112_group_2, patch_28_group, patch_112_group,soft_thr,soft_bias):
            c_s = pi.shape[2]
            k_s = wi[0].shape[2]
            # fc1 & fc2 transform the extracted patches (wi and xi) into flattened vectors for similarity calculation.
            # necessary for graph-based attention mechanisms to work effectively, as the patches need to be represented 
            # in a lower-dimensional space for efficient computation of relationships and attention scores. Section 3.2 of the paper
            wi = self.fc1(wi.view(wi.shape[1],-1))
            xi = self.fc2(xi.view(xi.shape[1],-1)).permute(1,0)
            """
            x shape : torch.Size([1, 2, 448, 204])
            xi shape =  torch.Size([24, 91392])
            wi shape =  torch.Size([91392, 24])
            The core of the attention mechanism is matrix multiplication (torch.matmul(wi, xi)), 
            which computes the similarity score between patches wi and xi.
            This results in a score map, which represents how similar patches are in terms of content.
            Mask operation filters out less important patches by comparing the score map against the dynamically generated soft_thr and soft_bias values.
            Softmax: applied to the masked similarity scores, normalizing them so that the sum of the scores is 1. This ensures that the resulting similarity scores represent probabilities.
            
            """
            print("xi shape = " , xi.shape)
            print("wi shape = ", wi.shape)

            score_map = torch.matmul(wi,xi)
            score_map = score_map.view(1, score_map.shape[0], math.ceil(w / self.stride_2),
                                       math.ceil(h / self.stride_2))
            b_s, l_s, h_s, w_s = score_map.shape
            yi = score_map.view(l_s, -1)
            
            print("thr shape:", thr.shape)
            print("bias shape:", bias.shape)
            print("yi shape:", yi.shape)

            mask = F.relu(yi-yi.mean(dim=1,keepdim=True)*thr.unsqueeze(1)+bias.unsqueeze(1))
            mask_b = (mask!=0.).float()
            yi = yi * mask
            yi = F.softmax(yi * self.softmax_scale, dim=1)
            yi = yi * mask_b
            pi = pi.view(h_s * w_s, -1)
            yi = torch.mm(yi, pi)
            yi = yi.view(b_s, l_s, c_s, k_s, k_s)[0]
            zi = yi.view(1, l_s, -1).permute(0, 2, 1)
            """
            Patch Aggregation and Reconstruction:
            The similar patches identified by the attention mechanism are then weighted 
            (based on the computed similarity scores) and aggregated into a final patch (yi).
            Fold Operation: After aggregation, the patches are "folded" back into 
            the original image shape using torch.nn.functional.fold. This operation reconstructs the output image
            from the processed patches, taking care to avoid overlaps by normalizing the contributions of overlapping patches.
            """
            zi = yi.view(1, l_s, -1).permute(0, 2, 1)
            output_size = (raw_int_bs[2], raw_int_bs[3])  # (448, 204) in this case
            print(f"Expected output size: {output_size}, Folded output size: {zi.shape}")
            zi = yi.view(1, l_s, -1).permute(0, 2, 1)
            kernel_size = (self.ksize, self.ksize)  
            stride = self.stride_1 
            output_size = (raw_int_bs[2], raw_int_bs[3])  # (448, 204)
            zi = torch.nn.functional.fold(zi, output_size=output_size, kernel_size=kernel_size, stride=stride)
            print(f"Expected output size: {output_size}, Folded output size: {zi.shape}")

            inp = torch.ones_like(zi)
            inp_unf = torch.nn.functional.unfold(inp, kernel_size=kernel_size, padding=paddings[0], stride=stride)
            out_mask = torch.nn.functional.fold(inp_unf, output_size=output_size, kernel_size=kernel_size, stride=stride)
            zi = zi / out_mask
            y.append(zi)
        y = torch.cat(y, dim=0)
        print("y : " , y.shape )
        return y


class CES(nn.Module):
    def __init__(self, in_channels, num=4):
        super(CES, self).__init__()
        RBS1 = [ResBlock(default_conv, n_feats=in_channels, kernel_size=3, act=nn.PReLU(), res_scale=1
                         ) for _ in range(num)]
        self.RBS1 = nn.Sequential(*RBS1)
        RBS2 = [ResBlock(default_conv, n_feats=in_channels, kernel_size=3, act=nn.PReLU(), res_scale=1
                         ) for _ in range(num)]
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
        # print(" res and feats and scale : " , n_resblocks,n_feats,res_scale)
        # Head module (input channels for grayscale images is 1)
        m_head = [conv(2, n_feats, kernel_size)]  # Adjust input channel to 1 for grayscale

        # Body module
        m_body = [
            ResBlock(conv, n_feats, kernel_size, res_scale=res_scale
                     )for _ in range(n_resblocks // 2)
        ]
        m_body.append(CES(n_feats))
        for i in range(n_resblocks // 2):
            m_body.append(ResBlock(conv, n_feats, kernel_size, res_scale=res_scale))

        m_body.append(conv(n_feats, n_feats, kernel_size))

        m_tail = [conv(n_feats, 2, kernel_size)] 
        self.head = nn.Sequential(*m_head)
        self.body = nn.Sequential(*m_body)
        self.tail = nn.Sequential(*m_tail)

    def forward(self, x):
        # Pass the input through the network
        print("x shape :" , x.shape)
        res = self.head(x)
        res = self.body(res)
        res = self.tail(res)
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
    return RR(args)

class Model(nn.Module):
    def __init__(self, args, ckp):
        super(Model, self).__init__()
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
        pre_trained_model = 'D:/Paper/codes/PromptMR/promptmr_examples/cmrxrecon/pretrained/model.pt'
        if resume == -1:
            self.model.load_state_dict(torch.load(os.path.join(pre_trained_model, 'model_latest.pt')))
        elif pre_train != '.':
            print('Loading model from {}'.format(pre_train))
            self.model.load_state_dict(torch.load(pre_train))

    def forward_chop(self, x, shave=2, min_size=10): # shave=10, min_size=10000
        print("forward chop mechanism")
        scale = self.scale[self.idx_scale]
        n_GPUs = min(self.n_GPUs, 4)
        b, c, h, w = x.size()
        #############################################
        # adaptive shave
        # corresponding to scaling factor of the downscali/home/ubuntu/Documents/MC/RNAN_V2/DN_Gray/code/DIV2K/Val/DIV2K_HQng and upscaling modules in the network
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
                    sr_batch = Utilities.test_x8(self.model, lr_batch)  # data aug
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
