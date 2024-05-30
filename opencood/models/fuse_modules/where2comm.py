from turtle import update
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from opencood.models.sub_modules.torch_transformation_utils import warp_affine_simple
from opencood.models.comm_modules.where2comm import Communication
from opencood.models.sub_modules.quantizer import Quantizer_multiscale


class MaxFusion(nn.Module):
    def __init__(self):
        super(MaxFusion, self).__init__()

    def forward(self, x):
        return torch.max(x, dim=0)[0]


class Where2comm(nn.Module):
    def __init__(self, args):
        super(Where2comm, self).__init__()

        self.communication = False
        self.round = 1
        if 'communication' in args:
            self.communication = True
            self.naive_communication = Communication(args['communication'])
            if 'round' in args['communication']:
                self.round = args['communication']['round']
        self.discrete_ratio = args['voxel_size'][0]  # voxel_size[0]=0.4
        self.downsample_rate = args['downsample_rate']  # 2/4, downsample rate from original feature map [200, 704]
        self.downsample_rate = 2

        self.agg_mode = args['agg_operator']['mode']
        # self.multi_scale = args['multi_scale']
        self.multi_scale = False
        if self.multi_scale:
            layer_nums = args['layer_nums']
            self.num_levels = len(layer_nums)
        else:
            self.num_levels = 1
        self.fuse_modules = MaxFusion()

        self.channel_compressor_flag = False
        if 'channel_compressor' in args:
            self.channel_compressor_flag = True
            self.channel_compressor = Quantizer_multiscale
        self.vis_count = 0

    def regroup(self, x, record_len):
        cum_sum_len = torch.cumsum(record_len, dim=0)
        split_x = torch.tensor_split(x, cum_sum_len[:-1].cpu())
        return split_x

    def forward(self, x, confidence_map, record_len, pairwise_t_matrix, request_map=None, channel_compression=False, config_thre=True, backbone=None):
        """
        Fusion forwarding.

        Parameters
        ----------
        x : torch.Tensor
            input data, (sum(n_cav), C, H, W)

        record_len : list
            shape: (B)

        pairwise_t_matrix : torch.Tensor
            The transformation matrix from each cav to ego,
            shape: (B, L, L, 4, 4)

        Returns
        -------
        Fused feature.
        """
        _, C, H, W = x.shape

        parameter_dict = {}
        if self.channel_compressor_flag:
            if channel_compression:
                x_gt = x.clone()
                x = x.permute(0, 2, 3, 1).contiguous().view(-1, C)
                x, _, _, codebook_loss = self.channel_compressor(x)
                x = x.view(-1, H, W, C).permute(0, 3, 1, 2).contiguous()
                x[0] = x_gt[0]
                parameter_dict.update({'codebook_loss': codebook_loss})
                # parameter_dict.update({'x_gt': x_gt})
                debug = False
                if debug:
                    x = x_gt

        B, L = pairwise_t_matrix.shape[:2]

        # (B,L,L,2,3)
        pairwise_t_matrix = pairwise_t_matrix[:, :, :, [0, 1], :][:, :, :, :, [0, 1, 3]]  # [B, L, L, 2, 3]
        pairwise_t_matrix[..., 0, 1] = pairwise_t_matrix[..., 0, 1] * H / W
        pairwise_t_matrix[..., 1, 0] = pairwise_t_matrix[..., 1, 0] * W / H
        pairwise_t_matrix[..., 0, 2] = pairwise_t_matrix[..., 0, 2] / (
                    self.downsample_rate * self.discrete_ratio * W) * 2
        pairwise_t_matrix[..., 1, 2] = pairwise_t_matrix[..., 1, 2] / (
                    self.downsample_rate * self.discrete_ratio * H) * 2

        ############ 1. Split the features #######################
        # split x:[(L1, C, H, W), (L2, C, H, W), ...]
        # for example [[2, 256, 50, 176], [1, 256, 50, 176], ...]
        batch_node_features = self.regroup(x, record_len)
        batch_confidence_maps = self.regroup(confidence_map, record_len)
        batch_request_maps = self.regroup(request_map, record_len) if request_map is not None else None

        ############ 2. Communication (Mask the features) #########
        if self.communication:
            _, communication_masks, communication_rates = self.naive_communication(batch_confidence_maps, record_len,
                                                                                   pairwise_t_matrix,
                                                                                   batch_request_maps, config_thre=config_thre)
        else:
            communication_rates = torch.tensor(0).to(x.device)

        ############ 3. Fusion ####################################
        x_fuse = [[] for _ in range(self.num_levels)]
        for b in range(B):
            # number of valid agent
            N = record_len[b]
            # (N,N,4,4)
            # t_matrix[i, j]-> from i to j
            t_matrix = pairwise_t_matrix[b][:N, :N, :, :]
            node_features = batch_node_features[b]
            if self.communication:
                comm_masks = communication_masks[:N ** 2].view(N, N, 1, communication_masks.shape[-2], communication_masks.shape[-1])

            if self.multi_scale and backbone is not None:
                feats = backbone.resnet(node_features)
            else:
                feats = [node_features]
            
            for i in range(N):
                for l_i in range(self.num_levels):
                    feats_l = feats[l_i]  # (BxN, C', H, W)
                    H, W = feats_l.shape[-2:]
                    if self.communication:
                        if l_i == 0:
                            curr_node_features = feats_l * comm_masks[i]
                        else:
                            curr_node_features = feats_l * F.max_pool2d(comm_masks[i], kernel_size=2**(l_i))
                    else:
                        curr_node_features = feats_l
                    
                    neighbor_feature = warp_affine_simple(curr_node_features,
                                                        t_matrix[i, :, :, :],
                                                        (H, W))
                    # self.vis_feat(neighbor_feature)
                    # self.vis_count += 1

                    x_fuse[l_i].append(self.fuse_modules(neighbor_feature))
        if self.num_levels > 1:
            for i in range(self.num_levels):
                x_fuse[i] = torch.stack(x_fuse[i])
            x_fuse = backbone.decode_multiscale_feature(x_fuse)
        else:
            x_fuse = torch.stack(x_fuse[0])
        return x_fuse, communication_rates, parameter_dict

    def vis_feat(self, neighbor_feature_list, save_name='feat'):
        import seaborn as sns
        import matplotlib.pyplot as plt
        save_dir = '/remote-home/share/yhu/Co_Flow/opencood/visualization/{}/{}.png'.format(save_name, self.vis_count)
        fig, axes = plt.subplots(2, 1, figsize=(20, 9))
        for ax_i, _ in enumerate(neighbor_feature_list):
            feat = neighbor_feature_list[ax_i].max(dim=0)[0].cpu().numpy()
            sns.heatmap(feat, ax=axes[ax_i], cbar=True)
            axes[ax_i].set_xticks([])
            axes[ax_i].set_yticks([])
            
        fig.tight_layout()
        plt.savefig(save_dir)
        plt.close()