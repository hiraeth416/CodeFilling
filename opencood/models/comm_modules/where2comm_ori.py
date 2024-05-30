# -*- coding: utf-8 -*-
# Author: Yue Hu <phyllis1sjtu@outlook.com>
# License: TDG-Attribution-NonCommercial-NoDistrib

import torch
import torch.nn as nn
import numpy as np
from opencood.models.sub_modules.torch_transformation_utils import warp_affine_simple

class Communication(nn.Module):
    def __init__(self, args):
        super(Communication, self).__init__()
        
        self.smooth = False
        self.thre = args['thre']
        if 'solver' in args:
            self.solver = True
            self.solver_thre = args['solver']['thre']
            self.solver_method = args['solver']['method']
        else:
            self.solver = False
        if 'gaussian_smooth' in args:
            # Gaussian Smooth
            self.smooth = True
            kernel_size = args['gaussian_smooth']['k_size']
            c_sigma = args['gaussian_smooth']['c_sigma']
            self.gaussian_filter = nn.Conv2d(1, 1, kernel_size=kernel_size, stride=1, padding=(kernel_size-1)//2)
            self.init_gaussian_filter(kernel_size, c_sigma)
            self.gaussian_filter.requires_grad = False
        self.debug = False #True # 
        self.vis_count = 0
        
    def init_gaussian_filter(self, k_size=5, sigma=1):
        def _gen_gaussian_kernel(k_size=5, sigma=1):
            center = k_size // 2
            x, y = np.mgrid[0 - center : k_size - center, 0 - center : k_size - center]
            g = 1 / (2 * np.pi * sigma) * np.exp(-(np.square(x) + np.square(y)) / (2 * np.square(sigma)))
            return g
        gaussian_kernel = _gen_gaussian_kernel(k_size, sigma)
        self.gaussian_filter.weight.data = torch.Tensor(gaussian_kernel).to(self.gaussian_filter.weight.device).unsqueeze(0).unsqueeze(0)
        self.gaussian_filter.bias.data.zero_()

    def forward_single(self, batch_confidence_maps, record_len, pairwise_t_matrix):
        # batch_confidence_maps:[(L1, H, W), (L2, H, W), ...]
        # pairwise_t_matrix: (B,L,L,2,3)
        # thre: threshold of objectiveness
        # a_ji = (1 - q_i)*q_ji
        B, L, _, _, _ = pairwise_t_matrix.shape
        _, _, H, W = batch_confidence_maps[0].shape
        
        communication_masks = []
        communication_rates = []
        batch_communication_maps = []
        for b in range(B):
            # number of valid agent
            N = record_len[b]
            # (N,N,4,4)
            # t_matrix[i, j]-> from i to j
            # t_matrix = pairwise_t_matrix[b][:N, :N, :, :]

            ori_communication_maps = batch_confidence_maps[b].sigmoid().max(dim=1)[0].unsqueeze(1) # dim1=2 represents the confidence of two anchors
            
            if self.smooth:
                communication_maps = self.gaussian_filter(ori_communication_maps)
            else:
                communication_maps = ori_communication_maps

            ones_mask = torch.ones_like(communication_maps).to(communication_maps.device)
            zeros_mask = torch.zeros_like(communication_maps).to(communication_maps.device)
            communication_mask = torch.where(communication_maps>self.thre, ones_mask, zeros_mask)

            communication_rate = communication_mask[0].sum()/(H*W)

            # communication_mask = warp_affine_simple(communication_mask,
            #                                 t_matrix[0, :, :, :],
            #                                 (H, W))
            
            communication_mask_nodiag = communication_mask.clone()
            ones_mask = torch.ones_like(communication_mask).to(communication_mask.device)
            communication_mask_nodiag[::2] = ones_mask[::2]

            communication_masks.append(communication_mask_nodiag)
            communication_rates.append(communication_rate)
            batch_communication_maps.append(ori_communication_maps*communication_mask_nodiag)
        communication_rates = sum(communication_rates)/B
        communication_masks = torch.concat(communication_masks, dim=0)
        return batch_communication_maps, communication_masks, communication_rates
    
    def forward(self, batch_confidence_maps, record_len, pairwise_t_matrix, batch_request_maps=None, config_thre=True):
        # batch_confidence_maps:[(L1, H, W), (L2, H, W), ...]
        # pairwise_t_matrix: (B,L,L,2,3)
        # thre: threshold of objectiveness
        # a_ji = (1 - q_i)*q_ji
        B, L, _, _, _ = pairwise_t_matrix.shape
        _, _, H, W = batch_confidence_maps[0].shape

        # if not config_thre:
        #     self.thre = 0.0

        # comm_thre_list = [0, 0.001, 0.01, 0.03, 0.10, 0.30, 0.60, 1.0]
        # if config_thre:
        #     for i, thre in enumerate(comm_thre_list):
        #         if thre > self.thre:
        #             break
        
        communication_masks = []
        communication_rates = []
        batch_communication_maps = []
        for b in range(B):
            # number of valid agent
            N = record_len[b]
            # (N,N,4,4)
            # t_matrix[i, j]-> from i to j
            t_matrix = pairwise_t_matrix[b][:N, :N, :, :]
            curr_comm_maps = []
            comm_mask_list = []
            for i in range(N):
                if batch_request_maps is not None:
                    curr_request_maps = batch_request_maps[b][i:i+1]
                    # curr_request_mask = 1.0 - ((1 - curr_request_maps) > self.R_thre) * 1.0
                    # curr_request_maps = curr_request_mask
                    request_maps = warp_affine_simple(curr_request_maps.expand(N,-1,-1,-1),
                                            t_matrix[:, i, :, :],
                                            (H, W)) # warp request map of agent i to other agents' coordnate
                    comm_maps = batch_confidence_maps[b] * request_maps # (N, 1, H, W)
                    if self.debug:
                        self.vis_CR_mask('vis_cr_masks', batch_confidence_maps[b], request_maps)
                        self.vis_count += 1
                else:
                    comm_maps = batch_confidence_maps[b] # (N, 1, H, W)
                curr_comm_maps.append(comm_maps.unsqueeze(0))
            curr_comm_maps = torch.cat(curr_comm_maps, dim=0) # (N, N, 1, H, W)
            curr_comm_maps = curr_comm_maps.flatten(0,1)

            if self.smooth:
                curr_comm_maps = self.gaussian_filter(curr_comm_maps)

            # if config_thre:
            #     comm_mask = (curr_comm_maps>thre) * 1.0
            # else:
            comm_mask = (curr_comm_maps>self.thre) * 1.0

            comm_rate = comm_mask.sum()/(H*W*N*N)
            
            comm_mask_nodiag = comm_mask.clone()
            ones_mask = torch.ones_like(comm_mask).to(comm_mask.device)
            comm_mask_nodiag[::N+1] = ones_mask[::N+1] 
            # comm_mask_nodiag = ones_mask

            communication_masks.append(comm_mask_nodiag)
            communication_rates.append(comm_rate)
            batch_communication_maps.append(curr_comm_maps*comm_mask_nodiag)
        communication_rates = sum(communication_rates)/B
        communication_masks = torch.concat(communication_masks, dim=0)
        return batch_communication_maps, communication_masks, communication_rates
    
    def forward_CR(self, batch_confidence_maps, record_len, pairwise_t_matrix, batch_request_maps=None):
        # batch_confidence_maps:[(L1, H, W), (L2, H, W), ...]
        # pairwise_t_matrix: (B,L,L,2,3)
        # thre: threshold of objectiveness
        # a_ji = (1 - q_i)*q_ji
        B, L, _, _, _ = pairwise_t_matrix.shape
        _, _, H, W = batch_confidence_maps[0].shape
        
        communication_masks = []
        communication_rates = []
        batch_communication_maps = []
        for b in range(B):
            # number of valid agent
            N = record_len[b]
            # (N,N,4,4)
            # t_matrix[i, j]-> from i to j
            t_matrix = pairwise_t_matrix[b][:N, :N, :, :]
            curr_comm_maps = []
            for i in range(N):
                if batch_request_maps is not None:
                    request_maps = warp_affine_simple(batch_request_maps[b][i:i+1].expand(N,-1,-1,-1),
                                            t_matrix[:, i, :, :],
                                            (H, W)) # warp request map of agent i to other agents' coordnate
                    comm_maps = batch_confidence_maps[b] * request_maps # (N, 1, H, W)
                    if self.debug:
                        self.vis_CR_mask('vis_cr_masks', batch_confidence_maps[b], request_maps)
                        self.vis_count += 1
                else:
                    comm_maps = batch_confidence_maps[b] # (N, 1, H, W)
                curr_comm_maps.append(comm_maps.unsqueeze(0))
            curr_comm_maps = torch.cat(curr_comm_maps, dim=0) # (N, N, 1, H, W)
            curr_comm_maps = curr_comm_maps.flatten(0,1)

            if self.smooth:
                curr_comm_maps = self.gaussian_filter(curr_comm_maps)

            comm_mask = (curr_comm_maps>self.thre) * 1.0

            comm_rate = comm_mask.sum()/(H*W*N*N)
            
            comm_mask_nodiag = comm_mask.clone()
            ones_mask = torch.ones_like(comm_mask).to(comm_mask.device)
            comm_mask_nodiag[::N+1] = ones_mask[::N+1] 
            # comm_mask_nodiag = ones_mask

            communication_masks.append(comm_mask_nodiag)
            communication_rates.append(comm_rate)
            batch_communication_maps.append(curr_comm_maps*comm_mask_nodiag)
        communication_rates = sum(communication_rates)/B
        communication_masks = torch.concat(communication_masks, dim=0)
        return batch_communication_maps, communication_masks, communication_rates

    def vis_CR_mask(self, save_name, confidence_maps_list, request_maps_list):
        import seaborn as sns
        import matplotlib.pyplot as plt
        save_dir = '/remote-home/share/yhu/Co_Flow/opencood/visualization/{}/{}_{:.04f}.png'.format(save_name, self.vis_count, self.thre)
        fig, axes = plt.subplots(len(confidence_maps_list), 3, figsize=(20, 9))
        # fig, axes = plt.subplots(len(confidence_maps_list), 5, figsize=(28, 9))
        for ax_i, _ in enumerate(confidence_maps_list):
            conf_single = (confidence_maps_list[ax_i].cpu().numpy()[0]>self.thre)*1.0
            # conf_single = confidence_maps_list[ax_i].cpu().numpy()[0]
            total_size = conf_single.shape[0] * conf_single.shape[1]
            rate_single = conf_single.sum()/total_size
            sns.heatmap(conf_single, ax=axes[ax_i][0], cbar=True)
            axes[ax_i][0].set_xticks([])
            axes[ax_i][0].set_yticks([])
            axes[ax_i][0].set_title('t_{}_conf:{:.06f}'.format(ax_i, rate_single))

            # conf_whist = request_maps_list[ax_i].cpu().numpy()[0]
            conf_whist = (confidence_maps_list[ax_i].cpu().numpy()[0] * request_maps_list[ax_i].cpu().numpy()[0]>self.thre)*1.0
            rate_whist = conf_whist.sum()/total_size
            sns.heatmap(conf_whist, ax=axes[ax_i][1], cbar=True)
            axes[ax_i][1].set_xticks([])
            axes[ax_i][1].set_yticks([])
            axes[ax_i][1].set_title('t_{}_whist:{:.06f}'.format(ax_i, rate_whist))

            conf_diff = (conf_single>self.thre)*1.0 - (conf_single>self.thre)*1.0 * conf_whist
            rate_diff = conf_diff.sum()/total_size
            sns.heatmap(conf_diff, ax=axes[ax_i][2], cbar=True)
            axes[ax_i][2].set_xticks([])
            axes[ax_i][2].set_yticks([])
            axes[ax_i][2].set_title('t_{}_diff:{:.06f}'.format(ax_i, rate_diff))
        fig.tight_layout()
        plt.savefig(save_dir)
        plt.close()