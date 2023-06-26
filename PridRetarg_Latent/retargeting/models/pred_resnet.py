import torch
import torch.nn as nn
import torch.nn.functional as F
from models.skeleton import SkeletonUnpool, SkeletonPool, SkeletonConv, find_neighbor, SkeletonLinear
from models.res_block import res_block
from models.skeleton import build_edge_topology
import os

class PredNet(nn.Module):
    def __init__(self, args, joint_topology, enc, device, loss_recoder=None):
        super(PredNet, self).__init__()
        # num of  res blocks
        self.block_num = args.block_num
        # the decay rate of predcition loss
        self.gamma = args.gamma
        # window size of input
        self.input_window_size = args.window_size / 4
        # window size of prediction
        self.pred_window_size = args.pred_window_size / 4
        # num of horizon
        self.horizon = args.window_size // args.pred_window_size - 1

        #res_block parameters
        self.args = args
        self.enc = enc
        self.joint_topology = joint_topology

        # device and loss_recoder
        self.device = device
        self.loss_recoder = loss_recoder
        # criterion
        self.criterion = nn.MSELoss()
        # create layer
        self.para = []
        self.create_layer()
        self.optimizer = torch.optim.Adam(self.para, args.learning_rate, betas=(0.9, 0.999))

    def create_layer(self):
        self.res_blocks = nn.ModuleList()
        for i in range(self.block_num):
            block = res_block(self.args, self.joint_topology, self.enc)
            self.para += block.parameters()
            self.res_blocks.append(block)

    def set_input(self, latent, offset=None):
        self.latent = latent
        self.offset = offset

        # for module in self.res_blocks:
        #     module.skeleton_conv_1.set_offset(self.offset)
        #     module.skeleton_conv_2.set_offset(self.offset)

        # saperate input into horizon windows
        self.latents_input_seq =[]
        # print("pred_window_size: ", self.pred_window_size)
        for i in range(self.horizon + 1):
            latent_temp = latent[:,:,i*int(self.pred_window_size):(i+1)*int(self.pred_window_size)]
            self.latents_input_seq.append(latent_temp)
        return self.latents_input_seq

    def forward(self, horizon=None):
        if horizon is None:
            horizon = self.horizon
        if self.args.is_train:
            self.pred_list = []
            for i in range(horizon):
                latent = self.latents_input_seq[i]
                for module in self.res_blocks:
                    latent = module(latent)
                self.pred_list.append(latent)
            return self.pred_list
        else:
            input = self.latents_input_seq[0].to(self.device)
            for module in self.res_blocks:
                input = module(input)
            return input


    def backword(self):
        Loss = 0
        declay_rate = 1
        for i in range(self.horizon):
            block_loss = self.criterion(self.pred_list[i], self.latents_input_seq[i+1])
            Loss += block_loss * declay_rate
            self.loss_recoder.add_scalar('block_{}_loss'.format(i), block_loss)
            declay_rate *= self.gamma
        self.loss_recoder.add_scalar('total_loss', Loss)

        Loss.backward()

    def optimize_parameters(self):
        self.forward(self.horizon)
        self.optimizer.zero_grad()
        self.backword()
        self.optimizer.step()

    def save(self, path, epoch):
        from option_parser import try_mkdir
        path = os.path.join(path, str(epoch))
        try_mkdir(path)

        torch.save(self.state_dict(), os.path.join(path, 'pred_net.pt'))

        print('Save at {} succeed!'.format(path))

    def load(self, path, epoch=None):
        if epoch is None:
            self.load_state_dict(torch.load(path))
        else:
            path = os.path.join(path, str(epoch))
            self.load_state_dict(torch.load(os.path.join(path, 'pred_net.pt')))
