from models.integrated import IntegratedModel
from torch import optim
import torch
from models.utils import GAN_loss, ImagePool, get_ee, Criterion_EE, Eval_Criterion, Criterion_EE_2
from models.base_model import BaseModel
from option_parser import try_mkdir
from models.pred_resnet import PredNet

import os

class Pred_model(BaseModel):
    def __init__(self, args, character_names, dataset):
        super(Pred_model, self).__init__(args)
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

        # initial latent predictor
        enc = self.pretrained_models[0].auto_encoder.enc
        latent_topology = self.pretrained_models[0].auto_encoder.enc.topologies[args.num_layers - 1]
        self.latent_predictor = PredNet(args, latent_topology, enc, self.device, self.loss_recoder).to(self.device)
        self.predictor_para = self.latent_predictor.parameters()

        if self.is_train:
            self.optimizer = optim.Adam(self.predictor_para, lr=args.learning_rate, betas=(0.9, 0.999))
            self.criterion_latent = torch.nn.MSELoss()
            self.criterion_rec = torch.nn.MSELoss()
        else:
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
            self.motions_input[0][0] = motions[0][0][:,:,:self.args.pred_window_size]
            for i in range(self.n_topology):
                self.motions_input[i][0][1:] = self.motions_input[i][0][0]
                self.motions_input[i][1] = [0] * len(self.motions_input[i][1])

    def forward(self):
        self.offset_repr = []
        for i in range(self.n_topology):
            self.offset_repr.append(self.pretrained_models[i].static_encoder(self.dataset.offsets[i]))

        motion, offset_idx = self.motions_input[0]
        motion = motion.to(self.device)

        offsets = [self.offset_repr[0][p][offset_idx] for p in range(self.args.num_layers + 1)]
        self.latent, _ = self.pretrained_models[0].auto_encoder(motion, offsets)

        if self.args.is_train:
            self.latent_input_seq = self.latent_predictor.set_input(self.latent, offsets[len(self.pretrained_models[0].auto_encoder.dec.layers) - 1])
            self.latent_pred_seq = self.latent_predictor.forward()
            self.horizon = len(self.latent_pred_seq)

            # rec
            rnd_idx = torch.randint(len(self.character_names[0]), (self.latent_pred_seq[0].shape[0],))
            offsets = [self.offset_repr[0][p][rnd_idx] for p in range(self.args.num_layers + 1)]
            # pose
            self.pose_gt_seq = []
            self.pose_pred_seq = []
            for i in range(self.horizon):
                # pose
                res_gt = self.pretrained_models[0].auto_encoder.dec(self.latent_input_seq[i+1], offsets)
                res_pred = self.pretrained_models[0].auto_encoder.dec(self.latent_pred_seq[i], offsets)
                res_gt_denorm = self.dataset.denorm(0, rnd_idx, res_gt)
                res_pred_denorm = self.dataset.denorm(0, rnd_idx, res_pred)
                pose_gt = self.pretrained_models[0].fk.forward_from_raw(res_gt_denorm, self.dataset.offsets[0][offset_idx])
                pose_pred = self.pretrained_models[0].fk.forward_from_raw(res_pred_denorm, self.dataset.offsets[0][offset_idx])
                self.pose_gt_seq.append(pose_gt)
                self.pose_pred_seq.append(pose_pred)
        else:
            # self.latent_predictor.set_input(self.latent, offsets[len(self.pretrained_models[0].auto_encoder.dec.layers) - 1])
            # self.pred_latent = self.latent_predictor.forward(1)
            self.pred_latent = self.latent
            rnd_idx = list(range(self.latent.shape[0]))
            offsets_repr = [self.offset_repr[1][p][rnd_idx] for p in range(self.args.num_layers + 1)]
            self.res_pred = self.pretrained_models[1].auto_encoder.dec(self.latent, offsets_repr)
            self.res_pred_denorm = self.dataset.denorm(1, rnd_idx, self.res_pred)

    def backward(self):
        # latent loss
        latent_loss = 0
        pose_loss = 0
        declay_rate = 1
        for i in range(self.horizon):
            latent_loss_temp = self.criterion_latent(self.latent_pred_seq[i], self.latent_input_seq[i+1])
            self.loss_recoder.add_scalar('latent_loss_{}'.format(i), latent_loss_temp.item())
            latent_loss += latent_loss_temp * declay_rate

            pos_loss_temp = self.criterion_rec(self.pose_pred_seq[i], self.pose_gt_seq[i])
            pose_loss += pos_loss_temp * declay_rate
            self.loss_recoder.add_scalar('pose_loss_{}'.format(i), pos_loss_temp.item())
            declay_rate *= self.args.gamma

        LOSS = latent_loss * self.args.latent_loss_lambda\
                + pose_loss * self.args.pose_loss_lambda
        self.loss_recoder.add_scalar('total_loss', LOSS.item())

        LOSS.backward()


    def optimize_parameters(self):
        self.forward()
        self.optimizer.zero_grad()
        self.backward()
        self.optimizer.step()


    def save(self):
        self.latent_predictor.save(os.path.join(self.model_save_dir, "prednet"), self.epoch_cnt)

    def compute_test_result(self):
        for i in range(len(self.character_names[1])):
            dst_path = os.path.join(self.bvh_path, self.character_names[1][i])
            self.writer.write_raw(self.res_pred_denorm[i], 'quaternion',
                                            os.path.join(dst_path, 'pred.bvh'.format(self.id_test, 0)))

        self.id_test += 1

    def load(self, epoch=None):
        self.latent_predictor.load(os.path.join(self.model_save_dir, "prednet"), epoch)