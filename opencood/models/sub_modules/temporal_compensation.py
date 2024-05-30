# -*- coding: utf-8 -*-
# Author: Hao Xiang <haxiang@g.ucla.edu>
# License: TDG-Attribution-NonCommercial-NoDistrib

import torch
import torch.nn as nn
import torch.nn.functional as F

from opencood.models.sub_modules.torch_transformation_utils import \
    get_discretized_transformation_matrix, get_transformation_matrix, \
    warp_affine_simple, get_rotated_roi
from opencood.models.sub_modules.convgru import ConvGRU
from icecream import ic
from matplotlib import pyplot as plt
from opencood.models.sub_modules.SyncLSTM import SyncLSTM
from opencood.models.sub_modules.MotionNet import STPN, MotionPrediction, StateEstimation
from opencood.models.sub_modules.dcn_net import DCNNet

class TemporalCompensation(nn.Module):
    def __init__(self, args):
        super(TemporalCompensation, self).__init__()
        self.discrete_ratio = args['voxel_size'][0]  # voxel_size[0]=0.4    
        self.downsample_rate = args['downsample_rate']  # 2/4, downsample rate from original feature map [200, 704]
        self.downsample_rate = 2
        # channel_size = args['channel_size']
        # spatial_size = args['spatial_size']
        # TM_Flag = args['TM_Flag']
        # compressed_size = args['compressed_size']
        # self.compensation_net = SyncLSTM(channel_size, spatial_size, TM_Flag, compressed_size)
        self.stpn = STPN(height_feat_size=args['channel_size'])
        self.flow_thre = args['flow_thre']
        self.motion_pred = MotionPrediction(seq_len=1)
        self.state_classify = StateEstimation(motion_category_num=1)

        self.dcn = False
        if 'dcn' in args:
            self.dcn = True
            self.dcn_net = DCNNet(args['dcn'])

    def regroup(self, x, record_len):
        cum_sum_len = torch.cumsum(record_len, dim=0)
        split_x = torch.tensor_split(x, cum_sum_len[:-1].cpu())
        return split_x

    def forward(self, x, record_len, pairwise_t_matrix, flow_gt=None):
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
        B, L = pairwise_t_matrix.shape[:2]

        # split x:[(L1, C, H, W), (L2, C, H, W), ...]
        # for example [[2, 256, 50, 176], [1, 256, 50, 176], ...]
        feat_seqs = self.regroup(x, record_len)

        # (B,L,L,2,3)
        pairwise_t_matrix = pairwise_t_matrix[:,:,:,[0, 1],:][:,:,:,:,[0, 1, 3]] # [B, L, L, 2, 3]
        pairwise_t_matrix[...,0,1] = pairwise_t_matrix[...,0,1] * H / W
        pairwise_t_matrix[...,1,0] = pairwise_t_matrix[...,1,0] * W / H
        pairwise_t_matrix[...,0,2] = pairwise_t_matrix[...,0,2] / (self.downsample_rate * self.discrete_ratio * W) * 2
        pairwise_t_matrix[...,1,2] = pairwise_t_matrix[...,1,2] / (self.downsample_rate * self.discrete_ratio * H) * 2

        # iteratively warp feature to current timestamp
        batch_feat_seqs = []
        for b in range(B):
            # number of valid timestamps
            K = record_len[b]
            t_matrix = pairwise_t_matrix[b][:K, :K, :, :]
            curr_feat_seq = warp_affine_simple(feat_seqs[b],
                                            t_matrix[0, :, :, :],
                                            (H, W))
            batch_feat_seqs.append(curr_feat_seq[None,...])
        batch_feat_seqs = torch.cat(batch_feat_seqs, dim=0) # b, K, c, h, w
        batch_hist_feat_seqs = batch_feat_seqs.flip(1)

        # Backbone network
        bevs = self.stpn(batch_hist_feat_seqs) # b, K, c, h, w

        # Motion State Classification head
        state_class_pred = self.state_classify(bevs)

        # Motion Displacement prediction
        flow = self.motion_pred(bevs)
        flow = flow.view(-1, 2, bevs.size(-2), bevs.size(-1))
        
        # flow = flow_gt.float()
        # state_class_pred = None
        # import ipdb; ipdb.set_trace()
        
        # Given disp shift feature
        x_coord = torch.arange(batch_feat_seqs.size(-1)).float()
        y_coord = torch.arange(batch_feat_seqs.size(-2)).float()
        y, x = torch.meshgrid(y_coord, x_coord)
        grid = torch.cat([x.unsqueeze(0),y.unsqueeze(0)], dim=0).to(flow.device)
        ori_grid = torch.cat([x.unsqueeze(0),y.unsqueeze(0)], dim=0).unsqueeze(0).expand(flow.shape[0],-1,-1,-1).to(flow.device)
        
        # updated_grid = grid - updated_flow
        # grid_list = []
        # for i in range(flow.shape[0]):
        #     grid_list.append(self.get_source_grid(grid, flow[i]))
        # updated_grid = torch.stack(grid_list).float()
        # updated_grid = ori_grid 
        # updated_grid = ori_grid - flow
        # updated_flow = flow * (state_class_pred.sigmoid() > self.flow_thre)
        updated_flow = flow
        updated_grid = ori_grid - updated_flow

        # mask out unmoved points #
        # masks = []
        # for _i in range(flow.shape[0]):
        #     mask = self.get_warped_feature_mask(updated_flow[_i], updated_grid[_i])
        #     masks.append(mask.unsqueeze(0))
        # masks = torch.stack(masks).to(flow.device)

        updated_grid[:,0,:,:] = updated_grid[:,0,:,:]/ (batch_feat_seqs.size(-1)/2.0) - 1.0
        updated_grid[:,1,:,:] = updated_grid[:,1,:,:]/ (batch_feat_seqs.size(-2)/2.0) - 1.0
        out = F.grid_sample(batch_feat_seqs[:,0], grid=updated_grid.permute(0,2,3,1), mode='bilinear')
        # out = F.grid_sample(batch_feat_seqs[:,0], grid=updated_grid.permute(0,2,3,1), mode='bilinear')
        # out = F.grid_sample(batch_feat_seqs[:,0], grid=updated_grid.permute(0,2,3,1), mode='nearest')
        # out = F.grid_sample(batch_feat_seqs[:,0], grid=grid.permute(0,2,3,1), mode='bilinear')
        # out = batch_feat_seqs[:,0].contiguous()
        
        # out = out * masks
        # out = batch_feat_seqs[:,0]

        # if self.dcn:
        #     out = self.dcn_net(out)
        return flow, state_class_pred, out
    
    def get_source_grid(self, grid, flow):
        ''' 
        get the mask of the warped feature map, where the value is 1 means the location is valid, and 0 means the location is invalid.
        ----------
        Args:
            grid: (2, H, W)
            flow: (2, H, W)

        Returns:
            updated_grid: (2, H, W)
        '''
        _, H, W = grid.shape
        flow_mask = (flow>=0.5).flatten()
        target_grid = grid + flow
        target_idx = target_grid[1]*W + target_grid[0]
        target_idx = target_idx.flatten()
        target_idx_int = torch.round(target_idx).int()
        target_idx_diff = torch.abs(target_idx - target_idx_int)
        unique_value, inverse_idx = torch.unique(target_idx_int*flow_mask, return_inverse=True)
        import ipdb; ipdb.set_trace()
        source_idx = torch.arange(H*W)
        moved_source_idx = []
        for v in unique_value:
            if v > (H*W-1) or v == 0:
                continue
            idx = torch.where(target_idx_int == v)[0]
            source_id = idx[torch.argmin(target_idx_diff[idx])]
            source_idx[v] = source_id
            moved_source_idx.append(source_id)
            if source_id not in unique_value:
                source_idx[source_id] = 0
        source_idx = source_idx.reshape([H,W])
        source_idx_mat = torch.stack([torch.fmod(source_idx, W), torch.div(source_idx, W, rounding_mode="trunc")])
        return source_idx_mat.to(flow.device)

    def get_warped_feature_mask(self, flow, updated_grid):
        ''' 
        get the mask of the warped feature map, where the value is 1 means the location is valid, and 0 means the location is invalid.
        ----------
        Args:
            flow: (2, H, W)
            updated_grid: (2, H, W)

        Returns:
            mask: (H, W)
        '''
        def get_large_flow_indices(flow, threshold=0.5):
            '''
            find the indices of the flow map where the flow is larger than the threshold, which means these locations have been moved.
            ----------
            Args:
                flow: (2, H, W)

            '''
            max_values, max_indices = torch.max(torch.abs(flow[:2]), dim=0)
            large_indices = torch.nonzero(max_values > threshold, as_tuple=False)
            return large_indices

        def remove_duplicate_points(points):
            unique_points, inverse_indices = torch.unique(points, sorted=True, return_inverse=True, dim=0)
            return unique_points, inverse_indices

        def get_nonzero_idx(flow, idx):
            flow_values = flow[:, idx[:, 0], idx[:, 1]]
            nonzero_idx = torch.nonzero(torch.abs(flow_values).sum(dim=0) == 0, as_tuple=False).squeeze()
            return idx[nonzero_idx]

        flow_idx = get_large_flow_indices(flow)

        mask = torch.ones(flow.shape[-2], flow.shape[-1])
        if flow_idx.shape[0] == 0:
            return mask

        # print(flow_idx)
        updated_grid_points_tmp = updated_grid[:, flow_idx[:,0], flow_idx[:,1]].to(torch.int64).T
        # change the order of dim 1
        updated_grid_points = torch.zeros_like(updated_grid_points_tmp)
        updated_grid_points[:, 0] = updated_grid_points_tmp[:, 1]
        updated_grid_points[:, 1] = updated_grid_points_tmp[:, 0]
        # print(updated_grid_points)

        unique_points_idx, _ = remove_duplicate_points(updated_grid_points)
        # print(unique_points_idx)

        nonzero_idx = get_nonzero_idx(flow, unique_points_idx)
        # print(nonzero_idx)

        # TODO: mask out the outlier idx

        if len(nonzero_idx.shape) > 1:
            mask[nonzero_idx[:, 0], nonzero_idx[:, 1]] = 0
        else: 
            mask[nonzero_idx[0], nonzero_idx[1]] = 0

        return mask

    def forward_bp(self, x, record_len, pairwise_t_matrix, flow_gt=None):
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
        B, L = pairwise_t_matrix.shape[:2]

        # split x:[(L1, C, H, W), (L2, C, H, W), ...]
        # for example [[2, 256, 50, 176], [1, 256, 50, 176], ...]
        feat_seqs = self.regroup(x, record_len)

        # (B,L,L,2,3)
        pairwise_t_matrix = pairwise_t_matrix[:,:,:,[0, 1],:][:,:,:,:,[0, 1, 3]] # [B, L, L, 2, 3]
        pairwise_t_matrix[...,0,1] = pairwise_t_matrix[...,0,1] * H / W
        pairwise_t_matrix[...,1,0] = pairwise_t_matrix[...,1,0] * W / H
        pairwise_t_matrix[...,0,2] = pairwise_t_matrix[...,0,2] / (self.downsample_rate * self.discrete_ratio * W) * 2
        pairwise_t_matrix[...,1,2] = pairwise_t_matrix[...,1,2] / (self.downsample_rate * self.discrete_ratio * H) * 2

        # iteratively warp feature to current timestamp
        batch_feat_seqs = []
        for b in range(B):
            # number of valid timestamps
            K = record_len[b]
            t_matrix = pairwise_t_matrix[b][:K, :K, :, :]
            curr_feat_seq = warp_affine_simple(feat_seqs[b],
                                            t_matrix[0, :, :, :],
                                            (H, W))
            batch_feat_seqs.append(curr_feat_seq[None,...])
        batch_feat_seqs = torch.cat(batch_feat_seqs, dim=0) # b, K, c, h, w
        batch_hist_feat_seqs = batch_feat_seqs.flip(1)

        # Backbone network
        bevs = self.stpn(batch_hist_feat_seqs) # b, K, c, h, w

        # Motion Displacement prediction
        flow = self.motion_pred(bevs)
        flow = flow.view(-1, 2, bevs.size(-2), bevs.size(-1))

        # flow = flow_gt[0].float()
        
        # Motion State Classification head
        state_class_pred = self.state_classify(bevs)

        # Given disp shift feature
        x_coord = torch.arange(batch_feat_seqs.size(-1)).float()
        y_coord = torch.arange(batch_feat_seqs.size(-2)).float()
        y, x = torch.meshgrid(y_coord, x_coord)
        grid = torch.cat([x.unsqueeze(0),y.unsqueeze(0)], dim=0).unsqueeze(0).expand(flow.shape[0],-1,-1,-1).to(flow.device)
        updated_grid = grid - flow * (state_class_pred.sigmoid() > self.flow_thre)
        # updated_grid = grid - flow
        updated_grid[:,0,:,:] = updated_grid[:,0,:,:]/ (batch_feat_seqs.size(-1)/2.0) - 1.0
        updated_grid[:,1,:,:] = updated_grid[:,1,:,:]/ (batch_feat_seqs.size(-2)/2.0) - 1.0
        out = F.grid_sample(batch_feat_seqs[:,0], grid=updated_grid.permute(0,2,3,1), mode='bilinear')
        # out = F.grid_sample(batch_feat_seqs[:,0], grid=updated_grid.permute(0,2,3,1), mode='nearest')
        # out = F.grid_sample(batch_feat_seqs[:,0], grid=grid.permute(0,2,3,1), mode='bilinear')
        # out = batch_feat_seqs[:,0].contiguous()
        if self.dcn:
            out = self.dcn_net(out)
        return flow, state_class_pred, out
        # return flow, None, out


    def forward_debug(self, x, origin_x, record_len, pairwise_t_matrix):
        """
        Fusion forwarding
        Used for debug and visualization

        
        Parameters
        ----------
        x : torch.Tensor
            input data, (sum(n_cav), C, H, W)

        origin_x: torch.Tensor
            pillars (sum(n_cav), C, H * downsample_rate, W * downsample_rate)
            
        record_len : list
            shape: (B)
            
        pairwise_t_matrix : torch.Tensor
            The transformation matrix from each cav to ego, 
            shape: (B, L, L, 4, 4) 
            
        Returns
        -------
        Fused feature.
        """
        from matplotlib import pyplot as plt

        _, C, H, W = x.shape
        B, L = pairwise_t_matrix.shape[:2]

        split_x = self.regroup(x, record_len)
        split_origin_x = self.regroup(origin_x, record_len)

        # (B,L,L,2,3)
        pairwise_t_matrix = pairwise_t_matrix[:,:,:,[0, 1],:][:,:,:,:,[0, 1, 3]] # [B, L, L, 2, 3]
        pairwise_t_matrix[...,0,1] = pairwise_t_matrix[...,0,1] * H / W
        pairwise_t_matrix[...,1,0] = pairwise_t_matrix[...,1,0] * W / H
        pairwise_t_matrix[...,0,2] = pairwise_t_matrix[...,0,2] / (self.downsample_rate * self.discrete_ratio * W) * 2
        pairwise_t_matrix[...,1,2] = pairwise_t_matrix[...,1,2] / (self.downsample_rate * self.discrete_ratio * H) * 2


        # (B*L,L,1,H,W)
        roi_mask = torch.zeros((B, L, L, 1, H, W)).to(x)
        for b in range(B):
            N = record_len[b]
            for i in range(N):
                one_tensor = torch.ones((L,1,H,W)).to(x)
                roi_mask[b,i] = warp_affine_simple(one_tensor, pairwise_t_matrix[b][i, :, :, :],(H, W))

        batch_node_features = split_x
        # iteratively update the features for num_iteration times

        # visualize warped feature map
        for b in range(B):
            # number of valid agent
            N = record_len[b]
            # (N,N,4,4)
            # t_matrix[i, j]-> from i to j
            t_matrix = pairwise_t_matrix[b][:N, :N, :, :]

            # update each node i
            i = 0 # ego
            mask = roi_mask[b, i, :N, ...]
            # (N,C,H,W) neighbor_feature is agent i's neighborhood warping to agent i's perspective
            # Notice we put i one the first dim of t_matrix. Different from original.
            # t_matrix[i,j] = Tji
            neighbor_feature = warp_affine_simple(batch_node_features[b],
                                            t_matrix[i, :, :, :],
                                            (H, W))
            for idx in range(N):
                plt.imshow(torch.max(neighbor_feature[idx],0)[0].detach().cpu().numpy())
                plt.savefig(f"/GPFS/rhome/yifanlu/workspace/OpenCOOD/vis_result/debug_warp_feature/feature_{b}_{idx}")
                plt.clf()
                plt.imshow(mask[idx][0].detach().cpu().numpy())
                plt.savefig(f"/GPFS/rhome/yifanlu/workspace/OpenCOOD/vis_result/debug_warp_feature/mask_feature_{b}_{idx}")
                plt.clf()


        
        # visualize origin pillar feature 
        origin_node_features = split_origin_x

        for b in range(B):
            N = record_len[b]
            # (N,N,4,4)
            # t_matrix[i, j]-> from i to j
            t_matrix = pairwise_t_matrix[b][:N, :N, :, :]

            i = 0 # ego
            # (N,C,H,W) neighbor_feature is agent i's neighborhood warping to agent i's perspective
            # Notice we put i one the first dim of t_matrix. Different from original.
            # t_matrix[i,j] = Tji
            neighbor_feature = warp_affine_simple(origin_node_features[b],
                                            t_matrix[i, :, :, :],
                                            (H*self.downsample_rate, W*self.downsample_rate))

            for idx in range(N):
                plt.imshow(torch.max(neighbor_feature[idx],0)[0].detach().cpu().numpy())
                plt.savefig(f"/GPFS/rhome/yifanlu/workspace/OpenCOOD/vis_result/debug_warp_feature/origin_{b}_{idx}")
                plt.clf()