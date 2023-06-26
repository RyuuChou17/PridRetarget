import torch
import torch.nn as nn
import torch.nn.functional as F
from models.skeleton import SkeletonUnpool, SkeletonPool, SkeletonConv, find_neighbor, SkeletonLinear


class res_block(nn.Module):
    def __init__(self, args, topology, enc):
        super(res_block, self).__init__()
        # initial parameters of res_block
        self.args = args
        self.kernel_size = 3
        self.joint_topology = topology
        self.joint_num = enc.edge_num[args.num_layers - 1]
        self.neighbor_list = find_neighbor(self.joint_topology, self.args.skeleton_dist)
        self.in_channels = enc.channel_list[args.num_layers]
        self.out_channels = self.in_channels
        self.hidden_channels = 16 * self.joint_num
        self.padding = (self.kernel_size - 1) // 2
        self.padding_mode = self.args.padding_mode
        self.stride = 1
        self.bias = True
        self.in_offset_channel = 3 * enc.channel_base[args.num_layers - 1] // enc.channel_base[0]
        self.pooling_list = enc.pooling_list[args.num_layers - 1]


        self.create_layers()

    def create_layers(self):
        self.unpool = SkeletonUnpool(self.pooling_list, self.in_channels//len(self.neighbor_list))
        self.skeleton_conv_1 = SkeletonConv(self.neighbor_list, in_channels=self.in_channels, out_channels=self.hidden_channels,
                                            joint_num=self.joint_num, kernel_size=self.kernel_size, stride=self.stride,
                                            padding=self.padding, padding_mode=self.padding_mode, bias=self.bias, add_offset=False,
                                            in_offset_channel=self.in_offset_channel)
        self.bn_1 = nn.BatchNorm1d(self.hidden_channels)
        self.relu_1 = nn.ReLU()
        self.skeleton_conv_2 = SkeletonConv(self.neighbor_list, in_channels=self.hidden_channels, out_channels=self.out_channels,
                                            joint_num=self.joint_num, kernel_size=self.kernel_size, stride=self.stride,
                                            padding=self.padding, padding_mode=self.padding_mode, bias=self.bias, add_offset=False,
                                            in_offset_channel=self.in_offset_channel)
        self.bn_2 = nn.BatchNorm1d(self.out_channels)
        self.relu_2 = nn.ReLU()
        self.pool = SkeletonPool(edges=self.joint_topology, pooling_mode=self.args.skeleton_pool,
                                    channels_per_edge=self.out_channels // len(self.neighbor_list), last_pool=False)

    def forward(self, input):
        residual = input
        x = input
        x = self.unpool(x)
        x = self.skeleton_conv_1(x)
        x = self.bn_1(x)
        x = self.relu_1(x)
        x = self.skeleton_conv_2(x)
        x = self.bn_2(x)
        x = self.pool(x)
        x += residual
        x = self.relu_2(x)
        return x