from models.integrated import IntegratedModel
from torch import optim
import torch
from models.utils import GAN_loss, ImagePool, get_ee, Criterion_EE, Eval_Criterion, Criterion_EE_2
from models.base_model import BaseModel
from option_parser import try_mkdir
from modlels.pred_resnet import PredNet

import os

class predictvie_model(BaseModel):
    def __init__(self, args, character_names, dataset):
        super(predictvie_model, self).__init__(args)
        self.character_names = character_names
        self.dataset = dataset
        self.n_topology = len(character_names)

        # initial pretrianed retargeting models
        self.pretrained_models = []
        for i in range(self.n_topology):
            model = IntegratedModel(args, dataset.joint_topologies[i], None, self.device, character_names[i])
            # laod para for pretrained models
            model.load(os.path.join(self.model_save_dir, 'topology{}'.format(i)), 20000)
            self.pretrained_models.append(model)

        self.latent_predictor = PredNet(args, dataset.joint_topologies[0], self.device)

        if not self.is_train:
            import option_parser
            self.id_test = 0
            self.bvh_path = os.path.join(args.save_dir, 'results/bvh')
            option_parser.try_mkdir(self.bvh_path)

            from datasets.bvh_writer import BVH_writer
            from datasets.bvh_parser import BVH_file
            import option_parser
            file = BVH_file(option_parser.get_std_bvh(dataset=self.character_names[1][0]))
            self.writer = BVH_writer(file.edges, file.names)


    def set_input(self, motions):
        self.motions_input = motions

        if not self.is_train:
            self.motion_backup = []
            for i in range(self.n_topology):
                self.motion_backup.append(motions[i][0].clone())
                self.motions_input[i][0][1:] = self.motions_input[i][0][0]
                self.motions_input[i][1] = [0] * len(self.motions_input[i][1])


    def forward(self):
        motion, offset_idx = self.motions_input[0]
        motion = motion.to(self.device)
        self.offset_repr = self.models[0].static_encoder(self.dataset.offsets[0])

        motion_denorm = self.dataset.denorm(0, offset_idx, motion)
        offsets = [self.offset_repr[p][offset_idx] for p in range(self.args.num_layers + 1)]
        latent, res = self.models[0].auto_encoder(motion, offsets)
        
        # predict latent
        self.latent_predictor.set_input(latent,)


    def backward(self):
        pass


    def optimize_parameters(self):
        pass


    def save(self):
        self.Ea.save(os.path.join(self.model_save_dir, 'predict'), self.epoch_cnt)

    def compute_test_result(self):
        for i in range(len(self.character_names[1])):
            dst_path = os.path.join(self.bvh_path, self.character_names[1][i])
            self.writer.write_raw(self.res_pred_denorm[i], 'quaternion',
                                            os.path.join(dst_path, 'pred.bvh'.format(self.id_test, 0)))

        self.id_test += 1

    def load(self, epoch=None):
        self.Ea.load(os.path.join(self.model_save_dir, 'predict'), epoch)