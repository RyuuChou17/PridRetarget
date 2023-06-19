import torch
import torch.nn as nn
import torch.nn.functional as F
from models.skeleton import SkeletonUnpool, SkeletonPool, SkeletonConv, find_neighbor, SkeletonLinear


class res_block(nn.Module):
    def __init__(self, args, topology):
        super(res_block, self).__init__()
        # initial parameters of res_block
        self.args = args
        self.edge_num = [len(topology) + 1]
        if args.rotation == 'euler_angle': self.channel_base = [3]
        elif args.rotation == 'quaternion': self.channel_base = [4]
        self.channels = self.channel_base[0] * self.edge_num[0]

        # initial paramaters of skeleton_conv
        self.topologies = [topology]
        self.edge_num = [len(topology) + 1]
        self.neighbor_list = find_neighbor(self.topologies[0], args.skeleton_dist)
        self.in_channels = self.channel_base[0] * self.edge_num[0]
        self.joint_num = self.edge_num[0]
        self.kernel_size = args.kernel_size
        self.stride = 1
        self.padding = (self.kernel_size - 1) // 2
        self.padding_mode = args.padding_mode
        self.bias = True

    def create_layers(self):
        self.skeleton_conv_1 = SkeletonConv(self.neighbor_list, in_channels=self.in_channels, out_channels=self.in_channels,
                                            joint_num=self.joint_num, kernel_size=self.kernel_size, stride=self.stride,
                                            padding=self.padding, padding_mode=self.padding_mode, bias=self.bias)
        self.bn_1 = nn.BatchNorm1d(self.in_channels)
        self.relu_1 = nn.ReLU()
        self.skeleton_conv_2 = SkeletonConv()
        self.bn_2 = nn.BatchNorm1d(self.in_channels)
        self.relu_2 = nn.ReLU()

    def forward(self, input):
        residual = input

        x = self.skeleton_conv_1(input)
        x = self.bn_1(x)
        x = self.relu_1(x)
        x = self.skeleton_conv_2(x)
        x = self.bn_2(x)
        x += residual
        x = self.relu_2(x)
        return x