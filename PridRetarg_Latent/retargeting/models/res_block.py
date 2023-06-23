import torch
import torch.nn as nn
import torch.nn.functional as F
from models.skeleton import SkeletonUnpool, SkeletonPool, SkeletonConv, find_neighbor, SkeletonLinear


class res_block(nn.Module):
    def __init__(self, args, topology, in_channels, joint_num, in_offset_channel, pooling_list):
        super(res_block, self).__init__()
        # initial parameters of res_block
        self.args = args
        self.kernel_size = 3
        self.joint_topology = topology
        self.joint_num = joint_num
        self.neighbor_list = find_neighbor(self.joint_topology, self.args.skeleton_dist)
        self.in_channels = in_channels
        self.out_channels = self.in_channels
        self.hidden_channels = 16 * self.joint_num
        self.padding = (self.kernel_size - 1) // 2
        self.padding_mode = self.args.padding_mode
        self.stride = 1
        self.bias = True
        self.in_offset_channel = in_offset_channel
        self.pooling_list = pooling_list

        self.create_layers()

    def create_layers(self):
        self.unpool = SkeletonUnpool(self.pooling_list, self.in_channels//len(self.neighbor_list))
        self.skeleton_conv_1 = SkeletonConv(self.neighbor_list, in_channels=self.in_channels, out_channels=self.hidden_channels,
                                            joint_num=self.joint_num, kernel_size=self.kernel_size, stride=self.stride,
                                            padding=self.padding, padding_mode=self.padding_mode, bias=self.bias, add_offset=True,
                                            in_offset_channel=self.in_offset_channel)
        self.bn_1 = nn.BatchNorm1d(self.hidden_channels)
        self.relu_1 = nn.ReLU()
        self.skeleton_conv_2 = SkeletonConv(self.neighbor_list, in_channels=self.hidden_channels, out_channels=self.out_channels,
                                            joint_num=self.joint_num, kernel_size=self.kernel_size, stride=self.stride,
                                            padding=self.padding, padding_mode=self.padding_mode, bias=self.bias, add_offset=True,
                                            in_offset_channel=self.in_offset_channel)
        self.bn_2 = nn.BatchNorm1d(self.out_channels)
        self.relu_2 = nn.ReLU()

    def forward(self, input):
        residual = input
        x = nn.Upsample(scale_factor=2, mode=self.args.upsampling, align_corners=False)(input)
        x = self.unpool(x)
        x = self.skeleton_conv_1(x)
        x = self.bn_1(x)
        x = self.relu_1(x)
        x = self.skeleton_conv_2(x)
        x = self.bn_2(x)
        x += residual
        x = self.relu_2(x)
        return x