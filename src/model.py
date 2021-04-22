import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from utils import load_config, maskFOV_on_BEV


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_channels, channels, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)

        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

        self.conv3 = nn.Conv2d(channels, self.expansion * channels, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion * channels)

        self.downsample = downsample
        self.relu = nn.ReLU(inplace=True)


    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out = self.relu(residual + out)
        return out


class BackBone(nn.Module):
    def __init__(self, block, geom):
        super(BackBone, self).__init__()
        # The first block consists of two convolutional layers with channel number 32 and stride 1 (3.2.1)
        self.block1 = nn.Sequential(
            nn.Conv2d(in_channels=36, out_channels=32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(num_features=32),

            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(num_features=32),

            nn.ReLU(inplace=True)
        )

        # The second to fifth blocks are composed of residual layers
        # (with number of layers equals to 3, 6, 6, 3 respectively) (3.2.1)
        self.in_channels = self.block1[3].num_features
        self.block2 = self.make_residual_layer(block, 24, num_blocks=3)
        self.block3 = self.make_residual_layer(block, 48, num_blocks=6)
        self.block4 = self.make_residual_layer(block, 64, num_blocks=6)
        self.block5 = self.make_residual_layer(block, 96, num_blocks=3)

        # To up-sample the feature map, we add a top-down path that
        # up-samples the feature map by 2 each time (3.2.1)
        self.latlayer1 = nn.Conv2d(384, 196, kernel_size=1, stride=1, padding=0)
        self.latlayer2 = nn.Conv2d(256, 128, kernel_size=1, stride=1, padding=0)
        self.latlayer3 = nn.Conv2d(192, 96, kernel_size=1, stride=1, padding=0)

        # Top-down layers
        self.deconv1 = nn.ConvTranspose2d(196, 128, kernel_size=3, stride=2, padding=1, output_padding=1)
        p = 0 if geom['label_shape'][1] == 175 else 1
        self.deconv2 = nn.ConvTranspose2d(128, 96, kernel_size=3, stride=2, padding=1, output_padding=(1, p))

    def forward(self, x):
        # Variables & comments are with respect to Figure 2 in 3.1
        res_1_output = self.block1(x)
        res_2_output = self.block2(res_1_output)
        res_3_output = self.block3(res_2_output)
        res_4_output = self.block4(res_3_output)
        res_5_output = self.block5(res_4_output)

        # 1x1, 196 layer at the top right of the graph
        lat_1_output = self.latlayer1(res_5_output)

        # 1x1, 128 layer in the Up_sample_6 block
        lat_2_output = self.latlayer2(res_4_output)

        # Deconv 3x3, 128, x2 in the Up_sample_6 block
        up_sample_6_output = lat_2_output + self.deconv1(lat_1_output)

        # Repeat the same operation as in Up_sample_6, but with the output from Res_block_3
        lat_3_output = self.latlayer3(res_3_output)
        up_sample_7_output = lat_3_output + self.deconv2(up_sample_6_output)

        return up_sample_7_output

    def make_residual_layer(self, block, channels, num_blocks):
        layers = []

        # The first convolution of each residual block has a stride of 2
        # in order to down-sample the feature map (3.2.1)
        downsample = nn.Sequential(
            nn.Conv2d(self.in_channels, channels * block.expansion,
                        kernel_size=1, stride=2, bias=False),
            nn.BatchNorm2d(channels * block.expansion))
        layers.append(block(self.in_channels, channels, stride=2, downsample=downsample))

        self.in_channels = channels * block.expansion
        for _ in range(num_blocks - 1):
            layers.append(block(self.in_channels, channels, stride=1))
            # Update in_channels so the next block has the correct channel input
            self.in_channels = channels * block.expansion

        return nn.Sequential(*layers)


class Header(nn.Module):
    def __init__(self):
        super(Header, self).__init__()

        # As shown in Figure 2 in 3.1, the Header network starts with (3x3, 96) X 4
        self.conv1 = nn.Conv2d(in_channels=96, out_channels=96, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(num_features=96)
    
        self.conv2 = nn.Conv2d(in_channels=96, out_channels=96, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(num_features=96)

        self.conv3 = nn.Conv2d(in_channels=96, out_channels=96, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(num_features=96)

        self.conv4 = nn.Conv2d(in_channels=96, out_channels=96, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn4 = nn.BatchNorm2d(num_features=96)

        # 3x3, 1 block in the bottom middle of the figure
        self.clshead = nn.Conv2d(in_channels=96, out_channels=1, kernel_size=3, stride=1, padding=1, bias=True)

        # 3x3, 6 block in the bottom right of the figure
        self.reghead = nn.Conv2d(in_channels=96, out_channels=6, kernel_size=3, stride=1, padding=1, bias=True)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)

        x = self.conv2(x)
        x = self.bn2(x)

        x = self.conv3(x)
        x = self.bn3(x)

        x = self.conv4(x)
        x = self.bn4(x)

        # The classification branch outputs 1-channel feature map followed with sigmoid activation function (3.2.2)
        cls = torch.sigmoid(self.clshead(x))

        # The regression branch outputs 6-channel feature maps without non-linearity
        reg = self.reghead(x)

        return cls, reg


class Decoder(nn.Module):
    def __init__(self, geom):
        super(Decoder, self).__init__()
        self.geometry = [geom['L1'], geom['L2'], geom['W1'], geom['W2']]
        self.grid_size = 0.4

        self.target_mean = [0.008, 0.001, 0.202, 0.2, 0.43, 1.368]
        self.target_std_dev = [0.866, 0.5, 0.954, 0.668, 0.09, 0.111]

    def forward(self, x):
        '''
        :param x: Tensor 6-channel geometry
        6 channel map of [cos(yaw), sin(yaw), log(x), log(y), w, l]
        Shape of x: (B, C=6, H=200, W=175)
        :return: Concatenated Tensor of 8 channel geometry map of bounding box corners
        8 channel are [rear_left_x, rear_left_y,
                        rear_right_x, rear_right_y,
                        front_right_x, front_right_y,
                        front_left_x, front_left_y]
        Return tensor has a shape (B, C=8, H=200, W=175), and is located on the same device as x
        '''
        # Tensor in (B, C, H, W)

        device = torch.device('cpu')
        if x.is_cuda:
            device = x.get_device()

        for i in range(6):
            x[:, i, :, :] = x[:, i, :, :] * self.target_std_dev[i] + self.target_mean[i]

        cos_t, sin_t, dx, dy, log_w, log_l = torch.chunk(x, 6, dim=1)
        theta = torch.atan2(sin_t, cos_t)
        cos_t = torch.cos(theta)
        sin_t = torch.sin(theta)

        x = torch.arange(self.geometry[2], self.geometry[3], self.grid_size, dtype=torch.float32, device=device)
        y = torch.arange(self.geometry[0], self.geometry[1], self.grid_size, dtype=torch.float32, device=device)
        yy, xx = torch.meshgrid([y, x])
        centre_y = yy + dy
        centre_x = xx + dx
        l = log_l.exp()
        w = log_w.exp()
        rear_left_x = centre_x - l/2 * cos_t - w/2 * sin_t
        rear_left_y = centre_y - l/2 * sin_t + w/2 * cos_t
        rear_right_x = centre_x - l/2 * cos_t + w/2 * sin_t
        rear_right_y = centre_y - l/2 * sin_t - w/2 * cos_t
        front_right_x = centre_x + l/2 * cos_t + w/2 * sin_t
        front_right_y = centre_y + l/2 * sin_t - w/2 * cos_t
        front_left_x = centre_x + l/2 * cos_t - w/2 * sin_t
        front_left_y = centre_y + l/2 * sin_t + w/2 * cos_t

        decoded_reg = torch.cat([rear_left_x, rear_left_y, rear_right_x, rear_right_y,
                                 front_right_x, front_right_y, front_left_x, front_left_y], dim=1)

        return decoded_reg


class PIXOR(nn.Module):
    '''
    PIXOR PyTorch Module
    '''
    def __init__(self, decode=True):
        super(PIXOR, self).__init__()
        config, _, _, _ = load_config()
        self.backbone = BackBone(Bottleneck, config['geometry'])
        self.header = Header()
        self.corner_decoder = Decoder(config['geometry'])
        self.use_decode = decode
        self.cam_fov_mask = maskFOV_on_BEV(config['geometry']['label_shape'])
        
        # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
        
        prior = 0.01
        self.header.clshead.weight.data.fill_(-math.log((1.0-prior)/prior))
        self.header.clshead.bias.data.fill_(0)
        self.header.reghead.weight.data.fill_(0)
        self.header.reghead.bias.data.fill_(0)

    def set_decode(self, decode):
        '''
        Whether to run the decoder after each forward pass
        '''
        self.use_decode = decode

    def forward(self, x):
        '''
        Parameters:
            x (torch.Tensor): [batch_size, channels, height, width]

        Returns:
            torch.Tensor: [batch_size, 7, height, width] without decoder, or
                          [batch_size, 15, height, width] with decoder
        '''
        features = self.backbone(x)
        cls, reg = self.header(features)

        if self.use_decode:
            decoded = self.corner_decoder(reg)
            return torch.cat([cls, reg, decoded], dim=1)
        else:
            return torch.cat([cls, reg], dim=1)