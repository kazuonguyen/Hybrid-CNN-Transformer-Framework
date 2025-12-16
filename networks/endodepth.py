import torch
import torch.nn as nn
from typing import Dict, List
import torch.nn.functional as F
from networks.resnet_encoder import ResnetEncoder
from networks.swin_transformer import SwinTransformer
from networks.layers import convbn


class SpatialAttention(nn.Module):
    """
    Ozyoruk, K. B. et al. EndoSLAM dataset and an unsupervised monocular visual odometry and depth estimation approach for endoscopic videos. Med. Image Anal. 71, (2021).
    """

    def __init__(self, input_plane: int = 32, kernel_size: int = 3):

        super(SpatialAttention, self).__init__()

        # assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = kernel_size // 2
        self.conv1 = nn.Conv2d(input_plane, 4, kernel_size, padding=padding, bias=False)
        self.maxpool = nn.MaxPool2d(4, stride=4)
        self.relu = nn.LeakyReLU(inplace=True)

        self.conv2 = nn.Conv2d(4, 4, kernel_size, padding=padding, bias=False)
        self.softmax = nn.Softmax(dim=1)

        self.conv3 = nn.Conv2d(4, input_plane, kernel_size, padding=padding, bias=False)
        self.upsample = nn.Upsample(scale_factor=4)

    def forward(self, input: torch.Tensor):
        # input layer
        x = self.conv1(input)
        x = self.maxpool(x)
        # attention score
        s = torch.matmul(x, x.transpose(-2, -1).contiguous())
        s = self.relu(s)
        s = self.conv2(s)
        # attention weight
        w = self.softmax(s)
        z = torch.matmul(w, x)
        # output layer
        z = self.conv3(z)
        z = self.upsample(z)
        # residual output
        return z + input


class ResnetAttentionEncoder(ResnetEncoder):

    def __init__(self, *args, **kwargs):
        super(ResnetAttentionEncoder, self).__init__(*args, **kwargs)
        if hasattr(kwargs, "kernel_size"):
            kernel = kwargs['kernel_size']
        else:
            kernel = 3
        self.SAB = SpatialAttention(self.num_ch_enc[1], kernel_size=kernel)

    def forward(self, input_image):
        if self.training:
            return self.forward_train(input_image)
        else:
            return self.forward_test(input_image)

    def forward_train(self, input_image):
        features: Dict[int, torch.Tensor] = {}
        x = input_image
        x = self.encoder.relu(self.encoder.bn1(self.encoder.conv1(x)))
        features[0] = self.SAB(x)
        features[1] = self.encoder.layer1(self.encoder.maxpool(features[0]))
        features[2] = self.encoder.layer2(features[1])
        features[3] = self.encoder.layer3(features[2])
        features[4] = self.encoder.layer4(features[3])
        return features

    def forward_test(self, input_image):
        features: List[torch.Tensor] = []
        x = input_image
        x = self.encoder.relu(self.encoder.bn1(self.encoder.conv1(x)))
        features.append(self.SAB(x))
        features.append(self.encoder.layer1(self.encoder.maxpool(features[0])))
        features.append(self.encoder.layer2(features[1]))
        features.append(self.encoder.layer3(features[2]))
        features.append(self.encoder.layer4(features[3]))
        return features



class DepthDecoder(nn.Module):
    def __init__(self, num_ch_enc, scales=[0, 1, 2, 3],
                 num_output_channels=1,
                 use_skips=True):
        super(DepthDecoder, self).__init__()

        self.num_output_channels = num_output_channels
        self.upsample_mode = 'nearest'

        if isinstance(use_skips, bool):
            self.use_skips = [0, 1, 2, 3, 4]
        else:
            self.use_skips = use_skips

        enc_n = len(num_ch_enc)
        num_ch_enc = tuple(num_ch_enc)
        num_ch_dec = (16, 32, 64, 128, 256)
        
        # Store encoder length to handle different architectures
        self.enc_n = enc_n

        self.decoder = nn.ModuleList()
        for l in range(4, -1, -1):
            # Transformer has 4 layers (starting at 1/4 resolution)
            # ResNet has 5 layers (starting at 1/2 resolution)
            if enc_n == 4:
                i = l-1
            else:
                i=l

            layer = nn.ModuleDict()
            # upconv_0
            num_ch_in = num_ch_enc[-1] if l == 4 else num_ch_dec[l + 1]
            num_ch_out = num_ch_dec[l]
            layer["upconv_0"] = convbn(int(num_ch_in), num_ch_out, kernel=3)

            # upconv_1
            if l in self.use_skips:
                num_ch_in = num_ch_dec[l]
                if i > 0 and i <= enc_n:
                    num_ch_in += num_ch_enc[i-1]
                num_ch_out = num_ch_dec[l]
                layer["upconv_1"] = convbn(int(num_ch_in), num_ch_out, kernel=3)
            else:
                layer["upconv_1"] = nn.Identity()

            # Disparity conv
            if l in scales:
                layer["dispconv"] = nn.Conv2d(
                    num_ch_dec[l], self.num_output_channels, kernel_size=3,
                    stride=1, padding=1, bias=False, padding_mode='reflect')
            else:
                layer["dispconv"] = nn.Identity()
            # Add layer to decoder
            self.decoder.append(layer)
        
        # Extra conv for transformer to output depth map
        if enc_n == 4:
            self.extra_conv = nn.Conv2d(num_ch_dec[0], self.num_output_channels, kernel_size=3,
                         stride=1, padding=1, bias=False, padding_mode='reflect')
        else:
            self.extra_conv = None

    def forward(self, input_features, scales: List[int] = None):
        if self.training:
            return self.forward_train(input_features, scales)
        else:
            return self.forward_test(input_features,scales)

    def forward_train(self, input_features: Dict[int, torch.Tensor], scales: List[int] = None):

        scales = [-1] if scales is None else scales
        outputs : torch.tensor = None
        feature_n = len(input_features)

        x = input_features[feature_n-1]
        for j, layer in enumerate(self.decoder):

            i: int = feature_n-1 - j
            x = layer["upconv_0"](x)
            x = F.interpolate(x, scale_factor=2.0, mode=self.upsample_mode)  # upsample

            if i-1 in input_features.keys():
                x = torch.cat((x, input_features[i - 1]), 1)
            x = layer["upconv_1"](x)

            if i in scales and self.extra_conv is None:
                outputs = torch.sigmoid(layer["dispconv"](x))
        
        # Extra conv for transformer to get depth output
        if self.extra_conv is not None and 0 in scales:
            outputs = torch.sigmoid(self.extra_conv(x))

        return outputs

    def forward_test(self, input_features: List[torch.Tensor], scales: List[int] = None):
        output = None
        feature_n = len(input_features)
        x = input_features[feature_n-1]
        for j, layer in enumerate(self.decoder):
            i: int = feature_n-1 - j
            x = layer["upconv_0"](x)
            x = F.interpolate(x, scale_factor=2.0, mode=self.upsample_mode)  # upsample
            if i - 1 >= 0 and i - 1 < feature_n:
                x = torch.cat((x, input_features[i - 1]), 1)
            x = layer["upconv_1"](x)
            if i in scales and self.extra_conv is None:
                output = torch.sigmoid(layer["dispconv"](x))
        
        # Extra conv for transformer
        if self.extra_conv is not None and 0 in scales:
            output = torch.sigmoid(self.extra_conv(x))
                
        return output
