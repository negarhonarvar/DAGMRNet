'''
Author: Bingyu Xin
Affiliation: Computer Science department, Rutgers University, NJ, USA
Paper: https://arxiv.org/abs/2309.13839
Date: 2023-10-15
'''

import torch
import torch.nn as nn
import fastmri
import math
import torch.nn.functional as F
from fastmri.data import transforms
from typing import (
    List,
    Optional,
    Tuple,
)


def conv(in_channels, out_channels, kernel_size, bias=False, stride=1):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size//2), bias=bias, stride=stride)


class CALayer(nn.Module):
    def __init__(self, channel, reduction= 1, bias=False): # reduction default value = 16 , channel = n_feat0
        super(CALayer, self).__init__()
        # global average pooling: feature --> point
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # feature channel downscale and upscale --> channel weight
        # print(" parameters :" , channel, channel // reduction, bias)
        self.conv_du = nn.Sequential(
            nn.Conv2d(channel, max(channel // reduction, 1), 1, padding=0, bias=bias),
            nn.ReLU(inplace=True),
            nn.Conv2d(max(channel // reduction, 1) , channel, 1, padding=0, bias=bias),
            nn.Sigmoid()
        )

    def forward(self, x):
        # print("forward: " , x.shape)
        y = self.avg_pool(x)
        # print("y: ", y.shape)
        y = self.conv_du(y)
        return x * y


class CAB(nn.Module):
    def __init__(self, n_feat, kernel_size, reduction, bias, act, no_use_ca=False):
        super(CAB, self).__init__()
        modules_body = []
        modules_body.append(conv(n_feat, n_feat, kernel_size, bias=bias))
        modules_body.append(act)
        modules_body.append(conv(n_feat, n_feat, kernel_size, bias=bias))

        if not no_use_ca:
            self.CA = CALayer(n_feat, reduction, bias=bias)
        else:
            self.CA = nn.Identity()
        self.body = nn.Sequential(*modules_body)

    def forward(self, x):
        res = self.body(x)
        res = self.CA(res)
        res += x
        return res

##########################################################################
# ---------- Prompt Block -----------------------

class PromptBlock(nn.Module):
    def __init__(self, prompt_dim=128, prompt_len=5, prompt_size=96, lin_dim=192, learnable_input_prompt = False):
        super(PromptBlock, self).__init__()
        self.prompt_param = nn.Parameter(torch.rand(
            1, prompt_len, prompt_dim, prompt_size, prompt_size), requires_grad=learnable_input_prompt)
        self.linear_layer = nn.Linear(lin_dim, prompt_len)
        self.dec_conv3x3 = nn.Conv2d(
            prompt_dim, prompt_dim, kernel_size=3, stride=1, padding=1, bias=False)

    def forward(self, x):

        B, C, H, W = x.shape
        emb = x.mean(dim=(-2, -1))
        prompt_weights = F.softmax(self.linear_layer(emb), dim=1)
        prompt_param = self.prompt_param.unsqueeze(0).repeat(B, 1, 1, 1, 1, 1).squeeze(1)
        prompt = prompt_weights.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1) * prompt_param
        prompt = torch.sum(prompt, dim=1)

        prompt = F.interpolate(prompt, (H, W), mode="bilinear")
        prompt = self.dec_conv3x3(prompt)

        return prompt


class DownBlock(nn.Module):
    def __init__(self, input_channel, output_channel, n_cab, kernel_size, reduction, bias, act, no_use_ca=False, first_act=False):
        super(DownBlock, self).__init__()
        if first_act:
            self.encoder = [CAB(input_channel, kernel_size, reduction, bias=bias, act=nn.PReLU(), no_use_ca=no_use_ca)]
            self.encoder = nn.Sequential(*(self.encoder+[CAB(input_channel, kernel_size, reduction, bias=bias, act=act, no_use_ca=no_use_ca) for _ in range(n_cab-1)]))
        else:
            self.encoder = nn.Sequential(
                *[CAB(input_channel, kernel_size, reduction, bias=bias, act=act, no_use_ca=no_use_ca) for _ in range(n_cab)])
        self.down = nn.Conv2d(input_channel, output_channel, kernel_size=3, stride=2, padding=1, bias=True)

    def forward(self, x):
        enc = self.encoder(x)
        x = self.down(enc)
        return x, enc


class UpBlock(nn.Module):
    def __init__(self, in_dim, out_dim, prompt_dim, n_cab, kernel_size, reduction, bias, act, no_use_ca=False):
        super(UpBlock, self).__init__()

        self.fuse = nn.Sequential(*[CAB(in_dim+prompt_dim, kernel_size, reduction, bias=bias, act=act, no_use_ca=no_use_ca) for _ in range(n_cab)])
        self.reduce = nn.Conv2d(in_dim+prompt_dim, in_dim, kernel_size=1, bias=bias)

        self.up = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
                                nn.Conv2d(in_dim, out_dim, 1, stride=1, padding=0, bias=False))

        self.ca = CAB(out_dim, kernel_size, reduction, bias=bias, act=act, no_use_ca=no_use_ca)


    def forward(self,x,prompt_dec,skip):

        x = torch.cat([x, prompt_dec], dim=1)
        x = self.fuse(x)
        x = self.reduce(x)

        x = self.up(x) + skip
        x = self.ca(x)

        return x


class SkipBlock(nn.Module):
    def __init__(self, enc_dim, n_cab, kernel_size, reduction, bias, act, no_use_ca=False):
        super(SkipBlock, self).__init__()
        if n_cab == 0:
            self.skip_attn = nn.Identity()
        else:
            self.skip_attn = nn.Sequential(
                *[CAB(enc_dim, kernel_size, reduction, bias=bias, act=act, no_use_ca=no_use_ca) for _ in range(n_cab)])

    def forward(self, x):
        x = self.skip_attn(x)

        return x


class PromptUnet(nn.Module):
    def __init__(self, 
                 in_chans=10, 
                 out_chans=10, 
                 n_feat0=4, # 48
                 feature_dim = [6, 8, 10], # [72, 96, 120]
                 prompt_dim = [2, 4, 6], # [24, 48, 72]
                 len_prompt = [5, 5, 5], 
                 prompt_size = [16, 8, 4], # [64,32,16]
                 n_enc_cab = [1, 1, 1], # [2, 3, 3]
                 n_dec_cab = [1, 1, 1], # [2, 2, 3]
                 n_skip_cab = [1, 1, 1],
                 n_bottleneck_cab = 3,
                 no_use_ca = False,
                 learnable_input_prompt=False,
                 kernel_size=3, 
                 reduction= 3, # 4
                 act=nn.PReLU(), 
                 bias=False,
                 ):
        """
        PromptUnet, see in paper: https://arxiv.org/abs/2309.13839
        Args:
            in_chans: Number of channels in the input.
            out_chans: Number of channels in the output.
            n_feat0: Number of output channels in the first convolution layer.
            feature_dim: Number of output channels in each level of the encoder.
            prompt_dim: Number of channels in the prompt at each level of the decoder.
            len_prompt: number of components in the prompt at each level of the decoder.
            prompt_size: spatial size of the prompt at each level of the decoder.
            n_enc_cab: number of channel attention blocks (CAB) in each level of the encoder.
            n_dec_cab: number of channel attention blocks (CAB) in each level of the decoder.
            n_skip_cab: number of channel attention blocks (CAB) in each skip connection.
            n_bottleneck_cab: number of channel attention blocks (CAB) in the bottleneck.
            kernel_size: kernel size of the convolution layers.
            reduction: reduction factor for the channel attention blocks (CAB).
            act: activation function.
            bias: whether to use bias in the convolution layers.
            no_use_ca: whether to *not* use channel attention blocks (CAB).
            learnable_input_prompt: whether to learn the input prompt in the PromptBlock.
        """
        super(PromptUnet, self).__init__()

        # Feature extraction
        self.feat_extract = conv(in_chans, n_feat0, kernel_size, bias=bias)

        # Encoder - 3 DownBlocks
        self.enc_level1 = DownBlock(n_feat0, feature_dim[0], n_enc_cab[0], kernel_size, reduction, bias, act, no_use_ca=no_use_ca, first_act=True)
        self.enc_level2 = DownBlock(feature_dim[0], feature_dim[1], n_enc_cab[1], kernel_size, reduction, bias, act, no_use_ca=no_use_ca)
        self.enc_level3 = DownBlock(feature_dim[1], feature_dim[2], n_enc_cab[2], kernel_size, reduction, bias, act, no_use_ca=no_use_ca)

        # Skip Connections - 3 SkipBlocks
        self.skip_attn1 = SkipBlock(n_feat0, n_skip_cab[0], kernel_size, reduction, bias=bias, act=act, no_use_ca=no_use_ca)
        self.skip_attn2 = SkipBlock(feature_dim[0], n_skip_cab[1], kernel_size, reduction, bias=bias, act=act, no_use_ca=no_use_ca)
        self.skip_attn3 = SkipBlock(feature_dim[1], n_skip_cab[2], kernel_size, reduction, bias=bias, act=act, no_use_ca=no_use_ca)

        # Bottleneck
        self.bottleneck = nn.Sequential(*[CAB(feature_dim[2], kernel_size, reduction, bias=bias, act=act, no_use_ca=no_use_ca) for _ in range(n_bottleneck_cab)])

        # Decoder - 3 UpBlocks
        self.prompt_level3 = PromptBlock(prompt_dim=prompt_dim[2], prompt_len=len_prompt[2], prompt_size=prompt_size[2], lin_dim=feature_dim[2], learnable_input_prompt=learnable_input_prompt)
        self.dec_level3 = UpBlock(feature_dim[2], feature_dim[1], prompt_dim[2], n_dec_cab[2], kernel_size, reduction, bias, act, no_use_ca=no_use_ca)

        self.prompt_level2 = PromptBlock(prompt_dim=prompt_dim[1], prompt_len=len_prompt[1], prompt_size=prompt_size[1], lin_dim=feature_dim[1], learnable_input_prompt=learnable_input_prompt)
        self.dec_level2 = UpBlock(feature_dim[1], feature_dim[0], prompt_dim[1], n_dec_cab[1], kernel_size, reduction, bias, act, no_use_ca=no_use_ca)

        self.prompt_level1 = PromptBlock(prompt_dim=prompt_dim[0], prompt_len=len_prompt[0], prompt_size=prompt_size[0], lin_dim=feature_dim[0], learnable_input_prompt=learnable_input_prompt)
        self.dec_level1 = UpBlock(feature_dim[0], n_feat0, prompt_dim[0], n_dec_cab[0], kernel_size, reduction, bias, act, no_use_ca=no_use_ca)

        # OutConv
        self.conv_last = conv(n_feat0, out_chans, 5, bias=bias)

    def forward(self, x):
        # 0. featue extraction
        x = self.feat_extract(x)

        # 1. encoder
        x, enc1 = self.enc_level1(x)
        x, enc2 = self.enc_level2(x)
        x, enc3 = self.enc_level3(x)

        # 2. bottleneck
        x = self.bottleneck(x)

        # 3. decoder
        dec_prompt3 = self.prompt_level3(x)
        x = self.dec_level3(x,dec_prompt3,self.skip_attn3(enc3))

        dec_prompt2 = self.prompt_level2(x)
        x = self.dec_level2(x,dec_prompt2,self.skip_attn2(enc2))

        dec_prompt1 = self.prompt_level1(x)
        x = self.dec_level1(x,dec_prompt1,self.skip_attn1(enc1))

        # 4. last conv
        return self.conv_last(x)


class NormPromptUnet(nn.Module):
    def __init__(
        self,
        in_chans: int = 10,
        out_chans: int = 10,
        n_feat0: int = 4, # it used to be 48
        feature_dim: List[int] = [6, 8, 10], # it used to be [72, 96, 120]
        prompt_dim: List[int] =[2, 4, 6] , # it used to be [24, 48, 72]
        len_prompt: List[int] = [5, 5, 5],
        prompt_size: List[int] = [16, 8, 4], # [64, 32, 16]
        n_enc_cab: List[int] = [1, 1, 1], # [2, 3, 3]
        n_dec_cab: List[int] = [1, 1, 1], # [2, 2, 3]
        n_skip_cab: List[int] = [1, 1, 1],
        n_bottleneck_cab: int = 3,
        no_use_ca: bool = False,
        learnable_input_prompt=False,
    ):

        super().__init__()
        self.unet = PromptUnet(in_chans=in_chans,
                                out_chans = out_chans, 
                                n_feat0=n_feat0,
                                feature_dim = feature_dim,
                                prompt_dim = prompt_dim,
                                len_prompt = len_prompt,
                                prompt_size = prompt_size,
                                n_enc_cab = n_enc_cab,
                                n_dec_cab = n_dec_cab,
                                n_skip_cab = n_skip_cab,
                                n_bottleneck_cab = n_bottleneck_cab,
                                no_use_ca = no_use_ca,
                                learnable_input_prompt=learnable_input_prompt,
                                )

    def complex_to_chan_dim(self, x: torch.Tensor) -> torch.Tensor:  # combines complex numbers
        b, c, h, w, two = x.shape
        assert two == 2
        return x.permute(0, 4, 1, 2, 3).reshape(b, 2 * c, h, w)

    def chan_complex_to_last_dim(self, x: torch.Tensor) -> torch.Tensor:
        b, c2, h, w = x.shape
        assert c2 % 2 == 0
        c = c2 // 2
        return x.view(b, 2, c, h, w).permute(0, 2, 3, 4, 1).contiguous()

    def norm(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        b, c, h, w = x.shape
        x = x.reshape(b, c * h * w)

        mean = x.mean(dim=1).view(b, 1, 1, 1)
        std = x.std(dim=1).view(b, 1, 1, 1)

        x = x.view(b, c, h, w)
        return (x - mean) / std, mean, std

    def unnorm(
        self, x: torch.Tensor, mean: torch.Tensor, std: torch.Tensor
    ) -> torch.Tensor:
        return x * std + mean

    def pad(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, Tuple[List[int], List[int], int, int]]:
        _, _, h, w = x.shape
        w_mult = ((w - 1) | 7) + 1   # | for ints = bitwise or
        h_mult = ((h - 1) | 7) + 1
        w_pad = [math.floor((w_mult - w) / 2), math.ceil((w_mult - w) / 2)]
        h_pad = [math.floor((h_mult - h) / 2), math.ceil((h_mult - h) / 2)]
        # TODO: fix this type when PyTorch fixes theirs
        # the documentation lies - this actually takes a list
        # https://github.com/pytorch/pytorch/blob/master/torch/nn/functional.py#L3457
        # https://github.com/pytorch/pytorch/pull/16949
        x = F.pad(x, w_pad + h_pad)
        return x, (h_pad, w_pad, h_mult, w_mult)

    def unpad(
        self,
        x: torch.Tensor,
        h_pad: List[int],
        w_pad: List[int],
        h_mult: int,
        w_mult: int,
    ) -> torch.Tensor:
        return x[..., h_pad[0]: h_mult - h_pad[1], w_pad[0]: w_mult - w_pad[1]]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not x.shape[-1] == 2:
            raise ValueError("Last dimension must be 2 for complex.")

        # get shapes for unet and normalize
        x = self.complex_to_chan_dim(x)
        x, mean, std = self.norm(x)
        x, pad_sizes = self.pad(x)
        print("*************************************************")
        print(x.shape)
        print(x.type)
        print("*************************************************")
        x = self.unet(x)
        
        # get shapes back and unnormalize
        x = self.unpad(x, *pad_sizes)
        x = self.unnorm(x, mean, std)
        x = self.chan_complex_to_last_dim(x)

        return x


class PromptMRBlock(nn.Module):
    """
    Model block for end-to-end variational network.

    This model applies a combination of soft data consistency with the input
    model as a regularizer. A series of these blocks can be stacked to form
    the full variational network.
    """

    def __init__(self, model: nn.Module, num_adj_slices=5):
        """
        Args:
            model: Module for "regularization" component of variational
                network.
        """
        super().__init__()
        self.num_adj_slices = num_adj_slices
        self.model = model
        self.dc_weight = nn.Parameter(torch.ones(1))

    def sens_expand(self, x: torch.Tensor, sens_maps: torch.Tensor) -> torch.Tensor:
        _, c, _, _, _ = sens_maps.shape
        return fastmri.fft2c(fastmri.complex_mul(x.repeat_interleave(c // self.num_adj_slices, dim=1), sens_maps))

    def sens_reduce(self, x: torch.Tensor, sens_maps: torch.Tensor) -> torch.Tensor:
        b, c, h, w, _ = x.shape
        x = fastmri.ifft2c(x)
        return fastmri.complex_mul(x, fastmri.complex_conj(sens_maps)).view(b, self.num_adj_slices, c // self.num_adj_slices, h, w, 2).sum(
            dim=2, keepdim=False
        )

    def forward(
        self,
        current_kspace: torch.Tensor,
        ref_kspace: torch.Tensor,
        mask: torch.Tensor,
        sens_maps: torch.Tensor,
    ) -> torch.Tensor:
        zero = torch.zeros(1, 1, 1, 1, 1).to(current_kspace)
        soft_dc = torch.where(mask, current_kspace -
                              ref_kspace, zero) * self.dc_weight

        model_term = self.sens_expand(
            self.model(self.sens_reduce(current_kspace, sens_maps)), sens_maps
        )

        return current_kspace - soft_dc - model_term


class SensitivityModel(nn.Module):
    """
    Model for learning sensitivity estimation from k-space data.

    This model applies an IFFT to multichannel k-space data and then a U-Net
    to the coil images to estimate coil sensitivities. It can be used with the
    end-to-end variational network.
    """

    def __init__(
        self,
        in_chans: int = 2,
        out_chans: int = 2,
        num_adj_slices: int = 1, # 5
        n_feat0: int = 2, # 24
        feature_dim: List[int] = [3, 4, 5], # [36,48,60]
        prompt_dim: List[int] = [1, 2, 3], # [12,24,36]
        len_prompt: List[int] = [5, 5, 5],
        prompt_size: List[int] = [16, 8, 4], #[64, 32, 16]
        n_enc_cab: List[int] = [1, 1, 1], # [2, 3, 3]
        n_dec_cab: List[int] = [1, 1, 1], # [2, 2, 3]
        n_skip_cab: List[int] = [1, 1, 1],
        n_bottleneck_cab: int = 3,
        no_use_ca: bool = False,
        mask_center: bool = True,
        low_mem: bool = False,

    ):
        """
        Args:
            in_chans: Number of channels in the input.
            out_chans: Number of channels in the output.
            num_adj_slices: Number of adjacent slices.
            n_feat0: Number of top-level feature channels for PromptUnet.
            feature_dim: feature dim for each level in PromptUnet.
            prompt_dim: prompt dim for each level in PromptUnet.
            len_prompt: number of prompt component in each level.
            prompt_size: prompt spatial size.
            n_enc_cab: number of CABs (channel attention Blocks) in DownBlock.
            n_dec_cab: number of CABs (channel attention Blocks) in UpBlock.
            n_skip_cab: number of CABs (channel attention Blocks) in SkipBlock.
            n_bottleneck_cab: number of CABs (channel attention Blocks) in
                BottleneckBlock.
            no_use_ca: not using channel attention.
            mask_center: Whether to mask center of k-space for sensitivity map
                calculation.
        """
        super().__init__()
        self.mask_center = mask_center
        self.num_adj_slices = num_adj_slices
        self.low_mem = low_mem
        self.norm_unet = NormPromptUnet(in_chans=in_chans,
                                out_chans = out_chans,
                                n_feat0=n_feat0,
                                feature_dim = feature_dim,
                                prompt_dim = prompt_dim,
                                len_prompt = len_prompt,
                                prompt_size = prompt_size,
                                n_enc_cab = n_enc_cab,
                                n_dec_cab = n_dec_cab,
                                n_skip_cab = n_skip_cab,
                                n_bottleneck_cab = n_bottleneck_cab,
                                no_use_ca = no_use_ca)

    def chans_to_batch_dim(self, x: torch.Tensor) -> Tuple[torch.Tensor, int]:
        b, c, h, w, comp = x.shape

        return x.view(b * c, 1, h, w, comp), b

    def batch_chans_to_chan_dim(self, x: torch.Tensor, batch_size: int) -> torch.Tensor:
        bc, _, h, w, comp = x.shape
        c = bc // batch_size

        return x.view(batch_size, c, h, w, comp)

    def divide_root_sum_of_squares(self, x: torch.Tensor) -> torch.Tensor:
        b, adj_coil, h, w, two = x.shape
        coil = adj_coil//self.num_adj_slices
        x = x.view(b, self.num_adj_slices, coil, h, w, two)
        x = x / fastmri.rss_complex(x, dim=2).unsqueeze(-1).unsqueeze(2)

        return x.view(b, adj_coil, h, w, two)

    def get_pad_and_num_low_freqs(
        self, mask: torch.Tensor, num_low_frequencies: Optional[int] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if num_low_frequencies is None or num_low_frequencies == 0:
            # get low frequency line locations and mask them out
            squeezed_mask = mask[:, 0, 0, :, 0].to(torch.int8)
            cent = squeezed_mask.shape[1] // 2
            # running argmin returns the first non-zero
            left = torch.argmin(squeezed_mask[:, :cent].flip(1), dim=1)
            right = torch.argmin(squeezed_mask[:, cent:], dim=1)
            num_low_frequencies_tensor = torch.max(
                2 * torch.min(left, right), torch.ones_like(left)
            )  # force a symmetric center unless 1
        else:
            num_low_frequencies_tensor = num_low_frequencies * torch.ones(
                mask.shape[0], dtype=mask.dtype, device=mask.device
            )

        pad = (mask.shape[-2] - num_low_frequencies_tensor + 1) // 2

        return pad.type(torch.long), num_low_frequencies_tensor.type(torch.long)

    def compute_sens(self, model:nn.Module, images: torch.Tensor, compute_per_coil: bool) -> torch.Tensor:
        # batch_size * n_coils
        bc = images.shape[0]
        if compute_per_coil:
            output = []
            for i in range(bc):
                output.append(model(images[i].unsqueeze(0)))
            output = torch.cat(output, dim=0)
        else:
            output = model(images)
        return output

    def forward(
        self,
        masked_kspace: torch.Tensor,
        mask: torch.Tensor,
        num_low_frequencies: Optional[int] = None,
    ) -> torch.Tensor:
        if self.mask_center:
            pad, num_low_freqs = self.get_pad_and_num_low_freqs(
                mask, num_low_frequencies
            )
            masked_kspace = transforms.batched_mask_center(
                masked_kspace, pad, pad + num_low_freqs
            )
        # convert to image space
        images, batches = self.chans_to_batch_dim(
            fastmri.ifft2c(masked_kspace))

        # estimate sensitivities
        return self.divide_root_sum_of_squares(
            self.batch_chans_to_chan_dim(self.compute_sens(self.norm_unet,images,self.low_mem), batches)
        )


class PromptMR(nn.Module):
    """
    An prompt-learning based unrolled model for multi-coil MR reconstruction, 
    see https://arxiv.org/abs/2309.13839.

    """

    def __init__(
        self,
        num_cascades: int = 1, # 12
        num_adj_slices: int = 1, # 5
        n_feat0: int = 4, # 48
        feature_dim: List[int] = [6, 8, 10], # [72,96,120]
        prompt_dim: List[int] = [2, 4, 6], # [24,48,72]
        sens_n_feat0: int =2, # 24
        sens_feature_dim: List[int] = [3, 4, 5], #[36,48,60]
        sens_prompt_dim: List[int] = [1, 2, 3], #[12,24,36]
        len_prompt: List[int] = [5, 5, 5],
        prompt_size: List[int] = [16, 8, 4], # [64, 32, 16]
        n_enc_cab: List[int] = [1, 1, 1], # [2, 3, 3]
        n_dec_cab: List[int] = [1, 1, 1], # [2, 2, 3]
        n_skip_cab: List[int] = [1, 1, 1],
        n_bottleneck_cab: int = 3,
        no_use_ca: bool = False,
        sens_len_prompt: Optional[List[int]] = None,
        sens_prompt_size: Optional[List[int]] = None,
        sens_n_enc_cab: Optional[List[int]] = None,
        sens_n_dec_cab: Optional[List[int]] = None,
        sens_n_skip_cab: Optional[List[int]] = None,
        sens_n_bottleneck_cab: Optional[List[int]] = None,
        sens_no_use_ca: Optional[bool] = None,
        mask_center: bool = True,
        use_checkpoint: bool = False,
        low_mem: bool = False,
    ):
        """
        Args:
            num_cascades: Number of cascades (i.e., layers) for variational network.
            num_adj_slices: Number of adjacent slices.
            n_feat0: Number of top-level feature channels for PromptUnet.
            feature_dim: feature dim for each level in PromptUnet.
            prompt_dim: prompt dim for each level in PromptUnet.
            sens_n_feat0: Number of top-level feature channels for sense map
                estimation PromptUnet in PromptMR.
            sens_feature_dim: feature dim for each level in PromptUnet for
                sensitivity map estimation (SME) network.
            sens_prompt_dim: prompt dim for each level in PromptUnet in
                sensitivity map estimation (SME) network.
            len_prompt: number of prompt component in each level.
            prompt_size: prompt spatial size.
            n_enc_cab: number of CABs (channel attention Blocks) in DownBlock.
            n_dec_cab: number of CABs (channel attention Blocks) in UpBlock.
            n_skip_cab: number of CABs (channel attention Blocks) in SkipBlock.
            n_bottleneck_cab: number of CABs (channel attention Blocks) in
                BottleneckBlock.
            no_use_ca: not using channel attention.
            mask_center: Whether to mask center of k-space for sensitivity map
                calculation.
            use_checkpoint: Whether to use checkpointing to trade compute for GPU memory.
            low_mem: Whether to compute sensitivity map coil by coil to save GPU memory.
        """
        super().__init__()
        assert num_adj_slices % 2 == 1, "num_adj_slices must be odd"
        self.num_adj_slices = num_adj_slices
        self.center_slice = num_adj_slices//2
        self.sens_net = SensitivityModel(
            num_adj_slices=num_adj_slices,
            n_feat0=sens_n_feat0,
            feature_dim= sens_feature_dim,
            prompt_dim = sens_prompt_dim,
            len_prompt = sens_len_prompt if sens_len_prompt is not None else len_prompt,
            prompt_size = sens_prompt_size if sens_prompt_size is not None else prompt_size,
            n_enc_cab = sens_n_enc_cab if sens_n_enc_cab is not None else n_enc_cab,
            n_dec_cab = sens_n_dec_cab if sens_n_dec_cab is not None else n_dec_cab,
            n_skip_cab = sens_n_skip_cab if sens_n_skip_cab is not None else n_skip_cab,
            n_bottleneck_cab = sens_n_bottleneck_cab if sens_n_bottleneck_cab is not None else n_bottleneck_cab,
            no_use_ca = sens_no_use_ca if sens_no_use_ca is not None else no_use_ca,
            mask_center=mask_center,
            low_mem=low_mem,

        )
        self.cascades = nn.ModuleList(
            [PromptMRBlock(NormPromptUnet(2*num_adj_slices, 2*num_adj_slices, n_feat0, feature_dim, prompt_dim, len_prompt, prompt_size, n_enc_cab, n_dec_cab, n_skip_cab, n_bottleneck_cab, no_use_ca), num_adj_slices) for _ in range(num_cascades)]
        )
        self.use_checkpoint = use_checkpoint

    def forward(
        self,
        masked_kspace: torch.Tensor,
        mask: torch.Tensor,
        num_low_frequencies: Optional[int] = None,
    ) -> torch.Tensor:

        if self.use_checkpoint and self.training:
            sens_maps = torch.utils.checkpoint.checkpoint(
                self.sens_net, masked_kspace, mask, num_low_frequencies, use_reentrant=False)
        else:
            sens_maps = self.sens_net(masked_kspace, mask, num_low_frequencies)
        kspace_pred = masked_kspace.clone()
        for cascade in self.cascades:
            if self.use_checkpoint and self.training:
                kspace_pred = torch.utils.checkpoint.checkpoint(
                cascade, kspace_pred, masked_kspace, mask, sens_maps, use_reentrant=False)
            else:
                kspace_pred = cascade(kspace_pred, masked_kspace, mask, sens_maps)

        kspace_pred = torch.chunk(kspace_pred, self.num_adj_slices, dim=1)[
            self.center_slice]

        return fastmri.rss(fastmri.complex_abs(fastmri.ifft2c(kspace_pred)), dim=1)

##########################################################################
# -------------------------------  DAGL ----------------------------------

# """
# Model Construction functions
# """

# def default_conv(in_channels, out_channels, kernel_size,stride=1, bias=True):
#     return nn.Conv2d(
#         in_channels, out_channels, kernel_size,
#         padding=(kernel_size//2),stride=stride, bias=bias)

# class MeanShift(nn.Conv2d):
#     def __init__(
#         self, rgb_range,
#         rgb_mean=(0.4488, 0.4371, 0.4040), rgb_std=(1.0, 1.0, 1.0), sign=-1):

#         super(MeanShift, self).__init__(3, 3, kernel_size=1)
#         std = torch.Tensor(rgb_std)
#         self.weight.data = torch.eye(3).view(3, 3, 1, 1) / std.view(3, 1, 1, 1)
#         self.bias.data = sign * rgb_range * torch.Tensor(rgb_mean) / std
#         for p in self.parameters():
#             p.requires_grad = False

# class ResBlock(nn.Module):
#     def __init__(
#         self, conv, n_feats, kernel_size,
#         bias=True, bn=False, act=nn.PReLU(), res_scale=1):

#         super(ResBlock, self).__init__()
#         m = []
#         for i in range(2):
#             m.append(conv(n_feats, n_feats, kernel_size, bias=bias))
#             if bn:
#                 m.append(nn.BatchNorm2d(n_feats))
#             if i == 0:
#                 m.append(act)

#         self.body = nn.Sequential(*m)
#         self.res_scale = res_scale

#     def forward(self, x):
#         res = self.body(x).mul(self.res_scale)
#         res += x

#         return res
    
# """
# fundamental functions
# """
# def same_padding(images, ksizes, strides, rates):
#     assert len(images.size()) == 4
#     batch_size, channel, rows, cols = images.size()
#     out_rows = (rows + strides[0] - 1) // strides[0]
#     out_cols = (cols + strides[1] - 1) // strides[1]
#     effective_k_row = (ksizes[0] - 1) * rates[0] + 1
#     effective_k_col = (ksizes[1] - 1) * rates[1] + 1
#     padding_rows = max(0, (out_rows - 1) * strides[0] + effective_k_row - rows)
#     padding_cols = max(0, (out_cols - 1) * strides[1] + effective_k_col - cols)
#     # Pad the input
#     padding_top = int(padding_rows / 2.)
#     padding_left = int(padding_cols / 2.)
#     padding_bottom = padding_rows - padding_top
#     padding_right = padding_cols - padding_left
#     paddings = (padding_left, padding_right, padding_top, padding_bottom)
#     images = torch.nn.ZeroPad2d(paddings)(images)
#     return images, paddings


# def extract_image_patches(images, ksizes, strides, rates, padding='same'):
#     """
#     Extract patches from images and put them in the C output dimension.
#     :param padding:
#     :param images: [batch, channels, in_rows, in_cols]. A 4-D Tensor with shape
#     :param ksizes: [ksize_rows, ksize_cols]. The size of the sliding window for
#      each dimension of images
#     :param strides: [stride_rows, stride_cols]
#     :param rates: [dilation_rows, dilation_cols]
#     :return: A Tensor
#     """
#     assert len(images.size()) == 4
#     assert padding in ['same', 'valid']
#     paddings = (0, 0, 0, 0)

#     if padding == 'same':
#         images, paddings = same_padding(images, ksizes, strides, rates)
#     elif padding == 'valid':
#         pass
#     else:
#         raise NotImplementedError('Unsupported padding type: {}.\
#                 Only "same" or "valid" are supported.'.format(padding))

#     unfold = torch.nn.Unfold(kernel_size=ksizes,
#                              padding=0,
#                              stride=strides)
#     patches = unfold(images)
#     return patches, paddings

# """
# Graph model
# """
# class CE(nn.Module):
#     def __init__(self, ksize=7, stride_1=4, stride_2=1, softmax_scale=10,shape=64 ,p_len=64,in_channels=64
#                  , inter_channels=16,use_multiple_size=False,use_topk=False,add_SE=False,num_edge = 50):
#         super(CE, self).__init__()
#         self.ksize = ksize
#         self.shape=shape
#         self.p_len=p_len
#         self.stride_1 = stride_1
#         self.stride_2 = stride_2
#         self.softmax_scale = softmax_scale
#         self.inter_channels = inter_channels
#         self.in_channels = in_channels
#         self.use_multiple_size=use_multiple_size
#         self.use_topk=use_topk
#         self.add_SE=add_SE
#         self.num_edge = num_edge

#         self.g = nn.Conv2d(in_channels=self.in_channels, out_channels=self.inter_channels, kernel_size=3, stride=1,
#                            padding=1)
#         self.W = nn.Conv2d(in_channels=self.inter_channels, out_channels=self.in_channels, kernel_size=1, stride=1,
#                            padding=0)
#         self.theta = nn.Conv2d(in_channels=self.in_channels, out_channels=self.inter_channels, kernel_size=1, stride=1,
#                                padding=0)
#         self.fc1 = nn.Sequential(
#             nn.Linear(in_features=ksize**2*inter_channels,out_features=(ksize**2*inter_channels)//4),
#             nn.ReLU()
#         )
#         self.fc2 = nn.Sequential(
#             nn.Linear(in_features=ksize**2*inter_channels,out_features=(ksize**2*inter_channels)//4),
#             nn.ReLU()
#         )
#         self.thr_conv = nn.Conv2d(in_channels=in_channels,out_channels=1,kernel_size=ksize,stride=stride_1,padding=0)
#         self.bias_conv = nn.Conv2d(in_channels=in_channels,out_channels=1,kernel_size=ksize,stride=stride_1,padding=0)

#     def forward(self, b):
#         b1 = self.g(b)
#         b2 = self.theta(b)
#         b3 = b1
        
#         raw_int_bs = list(b1.size())  # b*c*h*w
#         b4, _ = same_padding(b,[self.ksize,self.ksize],[self.stride_1,self.stride_1],[1,1])
#         soft_thr = self.thr_conv(b4).view(raw_int_bs[0],-1)
#         soft_bias = self.bias_conv(b4).view(raw_int_bs[0],-1)

#         patch_28, paddings_28 = extract_image_patches(b1, ksizes=[self.ksize, self.ksize],
#                                                       strides=[self.stride_1, self.stride_1],
#                                                       rates=[1, 1],
#                                                       padding='same')
#         patch_28 = patch_28.view(raw_int_bs[0], raw_int_bs[1], self.ksize, self.ksize, -1)
#         patch_28 = patch_28.permute(0, 4, 1, 2, 3)
#         patch_28_group = torch.split(patch_28, 1, dim=0)

#         patch_112, paddings_112 = extract_image_patches(b2, ksizes=[self.ksize, self.ksize],
#                                                         strides=[self.stride_2, self.stride_2],
#                                                         rates=[1, 1],
#                                                         padding='same')

#         patch_112 = patch_112.view(raw_int_bs[0], raw_int_bs[1], self.ksize, self.ksize, -1)
#         patch_112 = patch_112.permute(0, 4, 1, 2, 3)
#         patch_112_group = torch.split(patch_112, 1, dim=0)

#         patch_112_2, paddings_112_2 = extract_image_patches(b3, ksizes=[self.ksize, self.ksize],
#                                                         strides=[self.stride_2, self.stride_2],
#                                                         rates=[1, 1],
#                                                         padding='same')

#         patch_112_2 = patch_112_2.view(raw_int_bs[0], raw_int_bs[1], self.ksize, self.ksize, -1)
#         patch_112_2 = patch_112_2.permute(0, 4, 1, 2, 3)
#         patch_112_group_2 = torch.split(patch_112_2, 1, dim=0)

#         y = []
#         w, h = raw_int_bs[2], raw_int_bs[3]
#         _, paddings = same_padding(b3[0,0].unsqueeze(0).unsqueeze(0), [self.ksize, self.ksize], [self.stride_2, self.stride_2], [1, 1])
#         for xi, wi,pi,thr,bias in zip(patch_112_group_2, patch_28_group, patch_112_group,soft_thr,soft_bias):
#             c_s = pi.shape[2]
#             k_s = wi[0].shape[2]
#             wi = self.fc1(wi.view(wi.shape[1],-1))
#             xi = self.fc2(xi.view(xi.shape[1],-1)).permute(1,0)
#             score_map = torch.matmul(wi,xi)
#             score_map = score_map.view(1, score_map.shape[0], math.ceil(w / self.stride_2),
#                                        math.ceil(h / self.stride_2))
#             b_s, l_s, h_s, w_s = score_map.shape
#             yi = score_map.view(l_s, -1)
#             mask = F.relu(yi-yi.mean(dim=1,keepdim=True)*thr.unsqueeze(1)+bias.unsqueeze(1))
#             mask_b = (mask!=0.).float()
#             yi = yi * mask
#             yi = F.softmax(yi * self.softmax_scale, dim=1)
#             yi = yi * mask_b
#             pi = pi.view(h_s * w_s, -1)
#             yi = torch.mm(yi, pi)
#             yi = yi.view(b_s, l_s, c_s, k_s, k_s)[0]
#             zi = yi.view(1, l_s, -1).permute(0, 2, 1)
#             zi = torch.nn.functional.fold(zi, (raw_int_bs[2], raw_int_bs[3]), (self.ksize, self.ksize), padding=paddings[0], stride=self.stride_1)
#             inp = torch.ones_like(zi)
#             inp_unf = torch.nn.functional.unfold(inp, (self.ksize, self.ksize), padding=paddings[0], stride=self.stride_1)
#             out_mask = torch.nn.functional.fold(inp_unf, (raw_int_bs[2], raw_int_bs[3]), (self.ksize, self.ksize), padding=paddings[0], stride=self.stride_1)
#             zi = zi / out_mask
#             y.append(zi)
#         y = torch.cat(y, dim=0)
#         return y
    
# class CES(nn.Module):
#     def __init__(self,in_channels,num=4):
#         super(CES,self).__init__()
#         RBS1 = [
#             ResBlock(
#                 default_conv, n_feats=in_channels, kernel_size=3, act=nn.PReLU(), res_scale=1
#             ) for _ in range(num)
#         ]
#         self.RBS1 = nn.Sequential(
#             *RBS1
#         )
#         RBS2 = [
#             ResBlock(
#                 default_conv, n_feats=in_channels, kernel_size=3, act=nn.PReLU(), res_scale=1
#             ) for _ in range(num)
#         ]
#         self.RBS2 = nn.Sequential(
#             *RBS2
#         )
#         # stage 1 (4 head)
#         self.c1_1 = CE(in_channels=in_channels)
#         self.c1_2 = CE(in_channels=in_channels)
#         self.c1_3 = CE(in_channels=in_channels)
#         self.c1_4 = CE(in_channels=in_channels)
#         self.c1_c = nn.Conv2d(in_channels,in_channels,1,1,0)
#         # stage 2 (4 head)
#         self.c2_1 = CE(in_channels=in_channels)
#         self.c2_2 = CE(in_channels=in_channels)
#         self.c2_3 = CE(in_channels=in_channels)
#         self.c2_4 = CE(in_channels=in_channels)
#         self.c2_c = nn.Conv2d(in_channels,in_channels,1,1,0)
#         # stage 3 (4 head)
#         self.c3_1 = CE(in_channels=in_channels)
#         self.c3_2 = CE(in_channels=in_channels)
#         self.c3_3 = CE(in_channels=in_channels)
#         self.c3_4 = CE(in_channels=in_channels)
#         self.c3_c = nn.Conv2d(in_channels,in_channels,1,1,0)
        
#     def forward(self, x):
#         # 4head-3stages
#         out = self.c1_c(torch.cat((self.c1_1(x),self.c1_2(x),self.c1_3(x),self.c1_4(x)),dim=1))+x
#         out = self.RBS1(out)
#         out = self.c2_c(torch.cat((self.c2_1(out),self.c2_2(out),self.c2_3(out),self.c2_4(out)),dim=1))+out
#         out  = self.RBS2(out)
#         out = self.c3_c(torch.cat((self.c3_1(out),self.c3_2(out),self.c3_3(out),self.c3_4(out)),dim=1))+out
#         return out
    
# class RR(nn.Module):
#     def __init__(self, args, conv=default_conv):
#         super(RR, self).__init__()
#         # define basic setting
#         n_resblocks = args.n_resblocks
#         n_feats = args.n_feats
#         kernel_size = 3

#         rgb_mean = (0.4488, 0.4371, 0.4040)
#         rgb_std = (1.0, 1.0, 1.0)

#         msa = CES(in_channels=n_feats)
#         # define head module
#         m_head = [conv(args.n_colors, n_feats, kernel_size)]

#         # define body module
#         m_body = [
#             ResBlock(
#                 conv, n_feats, kernel_size, nn.PReLU(), res_scale=args.res_scale
#             ) for _ in range(n_resblocks // 2)
#         ]
#         m_body.append(msa)
#         for i in range(n_resblocks // 2):
#             m_body.append(ResBlock(conv, n_feats, kernel_size, nn.PReLU(), res_scale=args.res_scale))

#         m_body.append(conv(n_feats, n_feats, kernel_size))
#         m_tail = [
#             conv(n_feats, args.n_colors, kernel_size)
#         ]

#         self.add_mean = MeanShift(args.rgb_range, rgb_mean, rgb_std, 1)

#         self.head = nn.Sequential(*m_head)
#         self.body = nn.Sequential(*m_body)
#         self.tail = nn.Sequential(*m_tail)

#     def forward(self, x):
#         res = self.head(x)

#         res = self.body(res)

#         res = self.tail(res)

#         return x+res

#     def load_state_dict(self, state_dict, strict=True):
#         own_state = self.state_dict()
#         for name, param in state_dict.items():
#             if name in own_state:
#                 if isinstance(param, nn.Parameter):
#                     param = param.data
#                 try:
#                     own_state[name].copy_(param)
#                 except Exception:
#                     if name.find('tail') == -1:
#                         raise RuntimeError('While copying the parameter named {}, '
#                                            'whose dimensions in the model are {} and '
#                                            'whose dimensions in the checkpoint are {}.'
#                                            .format(name, own_state[name].size(), param.size()))
#             elif strict:
#                 if name.find('tail') == -1:
#                     raise KeyError('unexpected key "{}" in state_dict'
#                                    .format(name))

# class Model(nn.Module):
#     def __init__(self, args, ckp):
#         super(Model, self).__init__()
#         print('Making model...')

#         self.scale = args.scale
#         self.idx_scale = 0
#         self.self_ensemble = args.self_ensemble
#         self.chop = args.chop
#         # print(self.chop,self.self_ensemble)
#         # exit(0)
#         self.precision = args.precision
#         self.cpu = args.cpu
#         self.n_GPUs = args.n_GPUs

#         self.save_models = args.save_models

#         module = import_module('model.' + args.model.lower())
#         self.model = module.make_model(args)

#         if not args.cpu:
#             torch.cuda.manual_seed(args.seed)
#             self.model.cuda()
#             if args.precision == 'half':
#                 self.model.half()

#             if args.n_GPUs > 1:
#                 gpu_list = range(0, args.n_GPUs)
#                 self.model = nn.DataParallel(self.model, gpu_list)

#         self.load(
#             ckp.dir,
#             pre_train=args.pre_train,
#             resume=args.resume,
#             cpu=args.cpu
#         )
#         if args.print_model:
#             print(self.model)

#     def forward(self, x, idx_scale,ensemble=False):
#         self.ensemble=ensemble
#         self.idx_scale = idx_scale
#         target = self.get_model()
#         if hasattr(target, 'set_scale'):
#             target.set_scale(idx_scale)
#         if self.chop and not self.training:
#             with torch.no_grad():  # this can save much memory
#                 Out = self.forward_chop(x)
#             return Out
#         else:
#             return self.model(x)

#     def get_model(self):
#         if self.n_GPUs == 1:
#             return self.model
#         else:
#             return self.model.module

#     def state_dict(self, **kwargs):
#         target = self.get_model()
#         return target.state_dict(**kwargs)

#     def save(self, apath, epoch, is_best=False):
#         target = self.get_model()
#         torch.save(
#             target.state_dict(),
#             os.path.join(apath, 'model', 'model_latest.pt')
#         )
#         if is_best:
#             torch.save(
#                 target.state_dict(),
#                 os.path.join(apath, 'model', 'model_best.pt')
#             )

#     def load(self, apath, pre_train='.', resume=-1, cpu=False):
#         if cpu:
#             kwargs = {'map_location': lambda storage, loc: storage}
#         else:
#             kwargs = {}

#         if resume == -1:
#             self.get_model().load_state_dict(
#                 torch.load(
#                     os.path.join(apath, 'model', 'model_latest.pt'),
#                     **kwargs
#                 ),
#                 strict=False
#             )
#         elif resume == 0:
#             if pre_train != '.':
#                 print('Loading model from {}'.format(pre_train))
#                 self.get_model().load_state_dict(
#                     torch.load(pre_train, **kwargs),
#                     strict=False
#                 )
#         else:
#             self.get_model().load_state_dict(
#                 torch.load(
#                     os.path.join(apath, 'model', 'model_{}.pt'.format(resume)),
#                     **kwargs
#                 ),
#                 strict=False
#             )

#     def forward_chop(self, x, shave=10, min_size=10000):
#         scale = self.scale[self.idx_scale]
#         n_GPUs = min(self.n_GPUs, 4)
#         b, c, h, w = x.size()
#         #############################################
#         # adaptive shave
#         # corresponding to scaling factor of the downscali/home/ubuntu/Documents/MC/RNAN_V2/DN_Gray/code/DIV2K/Val/DIV2K_HQng and upscaling modules in the network
#         shave_scale = 4
#         # max shave size
#         shave_size_max = 24
#         # get half size of the hight and width
#         h_half, w_half = h // 2, w // 2
#         # mod
#         mod_h, mod_w = h_half // shave_scale, w_half // shave_scale
#         # ditermine midsize along height and width directions
#         h_size = mod_h * shave_scale + shave_size_max
#         w_size = mod_w * shave_scale + shave_size_max
#         # h_size, w_size = h_half + shave, w_half + shave
#         ###############################################
#         # h_size, w_size = adaptive_shave(h, w)
#         lr_list = [
#             x[:, :, 0:h_size, 0:w_size],
#             x[:, :, 0:h_size, (w - w_size):w],
#             x[:, :, (h - h_size):h, 0:w_size],
#             x[:, :, (h - h_size):h, (w - w_size):w]]

#         if w_size * h_size < min_size:
#             sr_list = []
#             for i in range(0, 4, n_GPUs):
#                 lr_batch = torch.cat(lr_list[i:(i + n_GPUs)], dim=0)
#                 if self.ensemble == False:
#                     sr_batch = self.model(lr_batch)
#                 else:
#                     sr_batch = test_x8(self.model, lr_batch)  # data aug
#                 sr_list.extend(sr_batch.chunk(n_GPUs, dim=0))
#         else:
#             sr_list = [
#                 self.forward_chop(patch, shave=shave, min_size=min_size) \
#                 for patch in lr_list
#             ]

#         h, w = scale * h, scale * w
#         h_half, w_half = scale * h_half, scale * w_half
#         h_size, w_size = scale * h_size, scale * w_size
#         shave *= scale

#         output = torch.tensor(x.data.new(b, c, h, w))
#         output[:, :, 0:h_half, 0:w_half] \
#             = sr_list[0][:, :, 0:h_half, 0:w_half]
#         output[:, :, 0:h_half, w_half:w] \
#             = sr_list[1][:, :, 0:h_half, (w_size - w + w_half):w_size]
#         output[:, :, h_half:h, 0:w_half] \
#             = sr_list[2][:, :, (h_size - h + h_half):h_size, 0:w_half]
#         output[:, :, h_half:h, w_half:w] \
#             = sr_list[3][:, :, (h_size - h + h_half):h_size, (w_size - w + w_half):w_size]

#         return output

#     def adaptive_shave(h, w):

#         # corresponding to scaling factor of the downscaling and upscaling modules in the network
#         shave_scale = 4
#         # max shave size
#         shave_size_max = 24
#         # get half size of the hight and width
#         h_half, w_half = h // 2, w // 2
#         # mod
#         mod_h, mod_w = h_half // shave_scale, w_half // shave_scale
#         # ditermine midsize along height and width directions
#         midsize_h = mod_h * shave_scale + shave_size_max
#         midsize_w = mod_w * shave_scale + shave_size_max
#         # print('midsize_h={}, midsize_w={}'.format(midsize_h, midsize_w))
#         return midsize_h, midsize_w

#     '''
#     def adaptive_shave(self, h, w):
#         shave_scale = 4
#         shave_size_max = 12
#         h_half, w_half = h // 2, w // 2
#         mod_h, mod_w = h_half // shave_scale, w_half // shave_scale
#         midsize_h = mod_h * shave_scale + shave_size_max
#         midsize_w = mod_w * shave_scale + shave_size_max

#         return midsize_h, midsize_w
#     '''

#     def forward_x8(self, x, forward_function):
#         def _transform(v, op):
#             if self.precision != 'single':
#                 v = v.float()

#             v2np = v.data.cpu().numpy()
#             if op == 'v':
#                 tfnp = v2np[:, :, :, ::-1].copy()
#             elif op == 'h':
#                 tfnp = v2np[:, :, ::-1, :].copy()
#             elif op == 't':
#                 tfnp = v2np.transpose((0, 1, 3, 2)).copy()

#             if not self.cpu: ret = torch.Tensor(tfnp).cuda()
#             if self.precision == 'half': ret = ret.half()

#             return torch.tensor(ret)

#         lr_list = [x]
#         for tf in 'v', 'h', 't':
#             lr_list.extend([_transform(t, tf) for t in lr_list])

#         sr_list = [forward_function(aug) for aug in lr_list]
#         for i in range(len(sr_list)):
#             if i > 3:
#                 sr_list[i] = _transform(sr_list[i], 't')
#             if i % 4 > 1:
#                 sr_list[i] = _transform(sr_list[i], 'h')
#             if (i % 4) % 2 == 1:
#                 sr_list[i] = _transform(sr_list[i], 'v')

#         output_cat = torch.cat(sr_list, dim=0)
#         output = output_cat.mean(dim=0, keepdim=True)

#         return output
                
