import torch
import torch.nn as nn
import torch.nn.functional as F
from models.skeleton import SkeletonUnpool, SkeletonPool, SkeletonConv, find_neighbor, SkeletonLinear
from res_block import res_block
from models.skeleton import build_edge_topology

class PredNet(nn.Module):
    def __init__(self, args, joint_topology, device, window_size, pred_window_size):
        super(PredNet, self).__init__()
        # num of  res blocks
        self.block_num = args.block_num
        # the decay rate of predcition loss
        self.gamma = args.gamma
        # window size of input
        self.input_window_size = window_size / 4
        # window size of prediction
        self.pred_window_size = args.pred_window_size / 4
        # num of horizon
        self.horizon = args.window_size // args.pred_window_size - 1

        self.device = device

        # initial parameters of block
        self.args = args
        self.joint_topology = joint_topology
        self.edges = build_edge_topology(joint_topology, torch.zeros((len(joint_topology), 3)))

        # criterion
        self.criterion = nn.MSELoss()

        # create layer
        self.create_layer()

    def create_layer(self):
        self.res_blocks = nn.ModuleList()
        for i in range(self.block_num):
            self.res_blocks.append(res_block(self.args, self.edges))

    def set_input(self, latent, offset=None):
        self.latent = latent
        self.offset = offset

        # saperate input into horizon windows
        self.motions_input_seq =[]
        for i in range(self.horizon + 1):
            motion_temp = motions[:,:,i*self.pred_window_size:(i+1)*self.pred_window_size]
            self.motions_input_seq.append(motion_temp)

    def forward(self):
        if self.args.is_train:
            pred_list = []
            for i in range(self.horizon):
                motion = self.motions_input_seq[i]
                input = motion.to(self.device)
                for module in self.res_blocks:
                    motion = module(motion)
                pred_list.append(motion)


    def backword(self):
        Loss = 0
        declay_rate = 1
        for i in range(self.horizon):
            Loss += self.criterion(self.pred_list[i], self.motions_input_seq[i+1]) * declay_rate
            declay_rate *= self.gamma

        Loss.backward()

    def optimize_parameters(self):
        self.forward()
        self.optimizer.zero_grad()
        self.backword()
        self.optimizer.step()