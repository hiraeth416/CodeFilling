import torch.nn as nn
import numpy as np
import time
from opencood.utils.flow_utils import generate_flow_map
from opencood.models.sub_modules.pillar_vfe import PillarVFE
from opencood.models.sub_modules.point_pillar_scatter import PointPillarScatter
from opencood.models.sub_modules.base_bev_backbone import BaseBEVBackbone
from opencood.models.sub_modules.base_bev_backbone_resnet import ResNetBEVBackbone
from opencood.models.sub_modules.downsample_conv import DownsampleConv
from opencood.models.sub_modules.naive_compress import NaiveCompressor
from opencood.models.sub_modules.dcn_net import DCNNet
from opencood.models.sub_modules.SyncLSTM import SyncLSTM
from opencood.models.fuse_modules.where2comm import Where2comm
from opencood.models.sub_modules.torch_transformation_utils import warp_affine_simple
# from opencood.models.fuse_modules.where2comm_attn import Where2comm
# from opencood.models.fuse_modules.temporal_fuse import TemporalFusion
from opencood.models.sub_modules.temporal_compensation import TemporalCompensation
from opencood.utils.transformation_utils import *
import torch
from opencood.models.fuse_modules.max_fuse import MaxFusion, SMaxFusion
from opencood.models.fuse_modules.pointwise_fuse import PointwiseFusion
from opencood.visualization import vis_utils, my_vis, simple_vis
# import seaborn as sns
from opencood.tools.track.AB3DMOT import *
from opencood.utils import box_utils
import matplotlib.pyplot as plt
from opencood.tools.visFlow import vis_flow, vis_flow_sns, vis_feat
from opencood.models.sub_modules.matcher_sizhe import Matcher
from opencood.utils.common_utils import limit_period


def measure_time(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        execution_time = end_time - start_time
        print(f"函数 {func.__name__} 的执行时间：{execution_time} 秒")
        return result

    return wrapper


class PointPillarWhere2commKalman(nn.Module):
    def __init__(self, args):
        super(PointPillarWhere2commKalman, self).__init__()

        # PIllar VFE
        self.pillar_vfe = PillarVFE(args['pillar_vfe'],
                                    num_point_features=4,
                                    voxel_size=args['voxel_size'],
                                    point_cloud_range=args['lidar_range'])
        self.scatter = PointPillarScatter(args['point_pillar_scatter'])
        if 'resnet' in args['base_bev_backbone']:
            self.backbone = ResNetBEVBackbone(args['base_bev_backbone'], 64)
        else:
            self.backbone = BaseBEVBackbone(args['base_bev_backbone'], 64)

        self.voxel_size = args['voxel_size']
        # self.out_size_factor = args['out_size_factor']
        self.cav_lidar_range = args['lidar_range']

        self.out_channel = sum(args['base_bev_backbone']['num_upsample_filter'])

        # used to downsample the feature map for efficient computation
        self.shrink_flag = False
        if 'shrink_header' in args:
            self.shrink_flag = True
            self.shrink_conv = DownsampleConv(args['shrink_header'])
            self.out_channel = args['shrink_header']['dim'][-1]

        self.compression = False
        if 'compression' in args and args['compression'] > 0:
            self.compression = True
            self.naive_compressor = NaiveCompressor(self.out_channel, args['compression'])

        self.dcn = False
        if 'dcn' in args:
            self.dcn = True
            self.dcn_net = DCNNet(args['dcn'])

        # self.fusion_net = TransformerFusion(args['fusion_args'])
        self.fusion_net = Where2comm(args['fusion_args'])
        # self.temporal_fusion_net = TemporalFusion(args['fusion_args'])
        # self.temporal_fusion_net = TemporalCompensation(args['fusion_args'])
        self.colla_fusion_net = MaxFusion(args['fusion_args'])
        self.multi_scale = args['fusion_args']['multi_scale']
        self.hist_len = args['hist_len']

        temporal_fusion_mode = args['fusion_args']['temporal_fusion'] if 'temporal_fusion' in args[
            'fusion_args'] else 'max'
        if temporal_fusion_mode == 'attn':
            self.hist_fusion_net = PointwiseFusion(args['fusion_args'])
        else:
            self.hist_fusion_net = SMaxFusion()

        self.process_frame_count = 0
        self.skip_scale = args['skip_scale']

        self.only_hist = False
        if 'temporal_args' in args and 'only_hist' in args['temporal_args']:
            self.only_hist = args['temporal_args']['only_hist']
        self.with_hist = True
        if 'temporal_args' in args and 'with_hist' in args['temporal_args']:
            self.with_hist = args['temporal_args']['with_hist']
        self.with_compensation = True
        if 'temporal_args' in args and 'with_compensation' in args['temporal_args']:
            self.with_compensation = args['temporal_args']['with_compensation']
        self.wo_colla = False
        if 'temporal_args' in args and 'wo_colla' in args['temporal_args']:
            self.wo_colla = args['temporal_args']['wo_colla']
        self.sampling_gap = 1
        if 'temporal_args' in args and 'sampling_gap' in args['temporal_args']:
            self.sampling_gap = args['temporal_args']['sampling_gap']
        self.temporal_thre = 0.001
        if 'temporal_args' in args and 'temporal_thre' in args['temporal_args']:
            self.temporal_thre = args['temporal_args']['temporal_thre']

        print('only_hist: ', self.only_hist)
        print('with_hist: ', self.with_hist)
        print('with_compensation: ', self.with_compensation)
        print('wo_colla: ', self.wo_colla)

        self.cls_head = nn.Conv2d(self.out_channel, args['anchor_number'],
                                  kernel_size=1)
        self.reg_head = nn.Conv2d(self.out_channel, 7 * args['anchor_number'],
                                  kernel_size=1)
        self.use_dir = False
        if 'dir_args' in args.keys():
            self.use_dir = True
            self.dir_head = nn.Conv2d(self.out_channel, args['dir_args']['num_bins'] * args['anchor_number'],
                                  kernel_size=1) # BIN_NUM = 2
        if 'backbone_fix' in args.keys() and args['backbone_fix']:
            self.backbone_fix()
        if 'only_train_compressor' in args.keys() and args['only_train_compressor']:
            self.only_train_compressor()
        if 'finetune_w_compressor' in args.keys() and args['finetune_w_compressor']:
            self.finetune_w_compressor()
        

        self.detection_loss_codebook = False
        if 'detection_loss_codebook' in args.keys() and args['detection_loss_codebook']:
            self.detection_loss_codebook = True
        print('model detection loss codebook: ', self.detection_loss_codebook)

        self.hist_box_global = []
        self.hist_score_global = []
        self.prev_feat_box_dict = None

        self.init_weight()
        self.debug = False  # True #
        self.vis_count = 0

    def init_weight(self):
        pi = 0.01
        nn.init.constant_(self.cls_head.bias, -np.log((1 - pi) / pi))
        nn.init.normal_(self.reg_head.weight, mean=0, std=0.001)

    def backbone_fix(self):
        """
        Fix the parameters of backbone during finetune on timedelay。
        """
        for p in self.pillar_vfe.parameters():
            p.requires_grad = False

        for p in self.scatter.parameters():
            p.requires_grad = False

        for p in self.backbone.parameters():
            p.requires_grad = False

        if self.compression:
            for p in self.naive_compressor.parameters():
                p.requires_grad = False
        if self.shrink_flag:
            for p in self.shrink_conv.parameters():
                p.requires_grad = False

        for p in self.cls_head.parameters():
            p.requires_grad = False
        for p in self.reg_head.parameters():
            p.requires_grad = False

    def only_train_compressor(self):
        print('-------- only_train_compressor --------')
        for p in self.pillar_vfe.parameters():
            p.requires_grad = False

        for p in self.scatter.parameters():
            p.requires_grad = False

        for p in self.backbone.parameters():
            p.requires_grad = False

        if self.compression:
            for p in self.naive_compressor.parameters():
                p.requires_grad = False
        if self.shrink_flag:
            for p in self.shrink_conv.parameters():
                p.requires_grad = False

        for p in self.cls_head.parameters():
            p.requires_grad = False
        for p in self.reg_head.parameters():
            p.requires_grad = False
        for p in self.temporal_fusion_net.parameters():
            p.requires_grad = False
        for p in self.colla_fusion_net.parameters():
            p.requires_grad = False

    def finetune_w_compressor(self):
        print('-------- finetune_w_compressor --------')
        for p in self.pillar_vfe.parameters():
            p.requires_grad = False
        for p in self.scatter.parameters():
            p.requires_grad = False
        for p in self.temporal_fusion_net.parameters():
            p.requires_grad = False

    def regroup(self, x, record_len):
        cum_sum_len = torch.cumsum(record_len, dim=0)
        split_x = torch.tensor_split(x, cum_sum_len[:-1].cpu())
        return split_x

    def encode(self, x):
        feat_list = self.backbone.get_multiscale_feature(x)
        return feat_list

    def decode(self, feat_list):
        feat = self.backbone.decode_multiscale_feature(feat_list)
        # downsample feature to reduce memory
        if self.shrink_flag:
            feat = self.shrink_conv(feat)
        # compressor
        if self.compression:
            feat = self.naive_compressor(feat)
        # dcn
        if self.dcn:
            feat = self.dcn_net(feat)
        return feat

    def get_ego_feat(self, feats, record_len):
        # feats [N,C,H,W]
        split_feats = self.regroup(feats, record_len)
        curr_ego_feat = [feat[0:1] for feat in split_feats]
        curr_ego_feat = torch.cat(curr_ego_feat, dim=0) if len(curr_ego_feat) > 1 else curr_ego_feat[0]
        return curr_ego_feat  # [B, C, H, W]

    def get_voxel_feat(self, data_dict):
        voxel_features = data_dict['processed_lidar']['voxel_features']
        voxel_coords = data_dict['processed_lidar']['voxel_coords']
        voxel_num_points = data_dict['processed_lidar']['voxel_num_points']
        record_len = data_dict['record_len']

        batch_dict = {'voxel_features': voxel_features,
                      'voxel_coords': voxel_coords,
                      'voxel_num_points': voxel_num_points,
                      'record_len': record_len}
        # n, 4 -> n, c
        batch_dict = self.pillar_vfe(batch_dict)
        # n, c -> N, C, H, W
        batch_dict = self.scatter(batch_dict)
        return batch_dict

    def fusion(self, feat, conf_map, record_len, pairwise_t_matrix, request_map=None, channel_compression=False,
               config_thre=True):
        '''
        Function: collaboration at current time.
        Input:
            feat: [N, C, H, W]
            conf_map: [N, 1, H, W]
            record_len: list
            pairwise_t_matrix: [N, L, L, 4, 4]
        Output:
            fused_feat: [N, C, H, W]
            communication_rates: scalar
        '''
        fused_feature, comm_rates, result_dict = self.fusion_net(feat,
                                                                 conf_map,
                                                                 record_len,
                                                                 pairwise_t_matrix,
                                                                 request_map,
                                                                 channel_compression=channel_compression,
                                                                 config_thre=config_thre)
        return fused_feature, comm_rates, result_dict

    def get_colla_feat_list(self, feat_list_single, fused_feat, fuse_layer_id=0):
        feat_list_colla = []
        for layer_id, feat in enumerate(feat_list_single):
            if layer_id == fuse_layer_id:
                feat_list_colla.append(fused_feat)
            else:
                feat_list_colla.append(feat)
        return feat_list_colla

    def resetKal(self):
        self.hist_box_global = []
        self.hist_score_global = []
        self.prev_feat_box_dict = None
        
    def predictKal(self, warm_up=False):
        trk_hist0 = np.zeros((len(self.MOTtracker.trackers), 7))  # N x 7 , #get previous locations from trackers.
        for t, trk in enumerate(trk_hist0):
            pos = self.MOTtracker.trackers[t].get_state().reshape((-1, 1))
            trk_hist0[t] = [pos[0], pos[1], pos[2], pos[3], pos[4], pos[5], pos[6]]
        trk_hist0 = trk_hist0[:, self.MOTtracker.reorder_back]

        trk_hist1 = np.zeros((len(self.MOTtracker.trackers), 7))  # N x 7 , #get previous locations from trackers.
        for t, trk in enumerate(trk_hist1):
            pos = self.MOTtracker.trackers[t].get_hist().reshape((-1, 1))
            trk_hist1[t] = [pos[0], pos[1], pos[2], pos[3], pos[4], pos[5], pos[6]]
        trk_hist1 = trk_hist1[:, self.MOTtracker.reorder_back]

        trks = np.zeros((len(self.MOTtracker.trackers), 7))  # N x 7 , #get predicted locations from existing trackers.
        score = np.zeros((len(self.MOTtracker.trackers), 1))
        # import ipdb; ipdb.set_trace()
        for t, trk in enumerate(trks):
            flow = (trk_hist0[t] - trk_hist1[t])
            trk[:6] = trk_hist0[t][:6] + flow[:6]  # n, 4, 2
            trk[-1] = trk_hist0[t][-1]
            score[t] = self.MOTtracker.trackers[t].track_score.cpu().numpy()
        trks = np.ma.compress_rows(np.ma.masked_invalid(trks))
        score = np.ma.compress_rows(np.ma.masked_invalid(score))

        to_del = []
        for t, trk in enumerate(trks):
            pos = self.MOTtracker.trackers[t].predict().reshape((-1, 1))
            if (pd.isnull(np.isnan(pos.all()))):  # ly
                to_del.append(t)
        for t in reversed(to_del):
            self.MOTtracker.trackers.pop(t)

        # trks: x,y,z,l,w,h,theta

        trks = box_utils.boxes_to_corners_3d(trks, order='lwh')
        trk_previous = box_utils.boxes_to_corners_3d(trk_hist0, order='lwh')

        if warm_up:
            return trk_previous, score, trk_previous
        else:
            return trks, score, trk_previous

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
            flow_values = flow[:, idx[:, 0], idx[:, 1]].contiguous()
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
        unique_points_idx[:,0] = torch.clamp(unique_points_idx[:,0], 0, mask.shape[0]-1)
        unique_points_idx[:,1] = torch.clamp(unique_points_idx[:,1], 0, mask.shape[1]-1)

        nonzero_idx = get_nonzero_idx(flow, unique_points_idx)
        # print(nonzero_idx)

        # TODO: mask out the outlier idx

        if len(nonzero_idx.shape) > 1:
            mask[nonzero_idx[:, 0], nonzero_idx[:, 1]] = 0
        else: 
            mask[nonzero_idx[0], nonzero_idx[1]] = 0

        return mask
    
    def warp_feat_with_flow(self, feats, flow, mask_out_hist=True):
        x_coord = torch.arange(feats.shape[-1]).float()
        y_coord = torch.arange(feats.shape[-2]).float()
        y, x = torch.meshgrid(y_coord, x_coord)
        ori_grid = torch.cat([x.unsqueeze(0), y.unsqueeze(0)], dim=0).unsqueeze(0).expand(flow.shape[0], -1, -1,
                                                                                            -1).to(flow.device)
        updated_grid = ori_grid - flow

        if mask_out_hist:
            masks = []
            for _i in range(flow.shape[0]):
                mask = self.get_warped_feature_mask(flow[_i], updated_grid[_i])
                masks.append(mask.unsqueeze(0))
            masks = torch.stack(masks).to(flow.device)

        updated_grid[:,0,:,:] = updated_grid[:,0,:,:]/ (feats.size(-1)/2.0) - 1.0
        updated_grid[:,1,:,:] = updated_grid[:,1,:,:]/ (feats.size(-2)/2.0) - 1.0
        out = F.grid_sample(feats, grid=updated_grid.permute(0,2,3,1), mode='bilinear')
        
        if mask_out_hist:
            out = out * masks
        
        return out

    def pred_box(self, device):
        if len(self.hist_box_global) < 2:
            return None, None, None

        # import ipdb; ipdb.set_trace()

        coord_past1 = torch.Tensor(corner_to_center(self.hist_box_global[-1], order='hwl')).to(device)   # (N, 7)
        coord_past2 = torch.Tensor(corner_to_center(self.hist_box_global[-2], order='hwl')).to(device)   # (N, 7)

        center_points_past1 = coord_past1[:,:2] # (N, 2)
        center_points_past2 = coord_past2[:,:2]

        cost_mat_center = torch.zeros((center_points_past2.shape[0], center_points_past1.shape[0])).to(device)

        center_points_past1_repeat = center_points_past1.unsqueeze(0).repeat(center_points_past2.shape[0], 1, 1)
        center_points_past2_repeat = center_points_past2.unsqueeze(1).repeat(1, center_points_past1.shape[0], 1)

        delta_mat = center_points_past1_repeat - center_points_past2_repeat

        angle_mat = torch.atan2(delta_mat[:,:,1], delta_mat[:,:,0]) # [num_cav_past2,num_cav_past1]

        coord_past2_angle_reverse = coord_past2[:,6].unsqueeze(1).repeat(1, center_points_past1.shape[0]).clone()

        visible_mat = torch.where((torch.abs(angle_mat-coord_past2[:,6].unsqueeze(1).repeat(1, center_points_past1.shape[0])) < 0.785) | (torch.abs(angle_mat-coord_past2[:,6].unsqueeze(1).repeat(1, center_points_past1.shape[0])) > 5.495) , 1, 0) # [num_cav_past2,num_cav_past1]

        cost_mat_center = torch.cdist(center_points_past2, center_points_past1) # [num_cav_past2,num_cav_past1]

        visible_mat = torch.where(cost_mat_center<0.5, 1, visible_mat)

        tmp_thre = torch.tensor(1000.).to(torch.float32).to(visible_mat.device)
        cost_mat_center = torch.where(visible_mat==1, cost_mat_center, tmp_thre)

        if cost_mat_center.shape[1] == 0 or cost_mat_center.shape[0] == 0:
            return None, None, None

        match = torch.min(cost_mat_center, dim=1)
        match_to_keep = torch.where(match[0] < 5)

        past2_ids = match_to_keep[0]
        past1_ids = match[1][match_to_keep[0]]
        
        coord_past2_angle_reverse += 3.1415926
        coord_past2_angle_reverse[coord_past2_angle_reverse>3.1415926] -= 6.2831852

        left_past2_id = [n for n in range(cost_mat_center.shape[0]) if n not in past2_ids]
        left_past1_id = [n for n in range(cost_mat_center.shape[1]) if n not in past1_ids]

        angle_mat_left = angle_mat[left_past2_id, :][:, left_past1_id]

        coord_past2_angle_reverse_left = coord_past2_angle_reverse[left_past2_id, :][: ,left_past1_id]

        visible_mat_left = torch.where((torch.abs(angle_mat_left-coord_past2_angle_reverse_left) < 0.785) | (torch.abs(angle_mat_left-coord_past2_angle_reverse_left) > 5.495) , 1, 0) # [num_cav_past2,num_cav_past1]
        
        cost_mat_center_left = torch.cdist(center_points_past2[left_past2_id], center_points_past1[left_past1_id]) # [num_cav_past2,num_cav_past1]
        # cost_mat_center_left = cost_mat_center[left_past2_id, :][:, left_past1_id]

        visible_mat_left = torch.where(cost_mat_center_left<0.5, 1, visible_mat_left)
        cost_mat_center_left = torch.where(visible_mat_left==1, cost_mat_center_left, tmp_thre)

        if cost_mat_center_left.shape[1] != 0 and cost_mat_center_left.shape[0] != 0:
            match_left = torch.min(cost_mat_center_left, dim=1)
            match_to_keep_left = torch.where(match_left[0] < 5)

            if match_to_keep_left[0].shape[0] != 0:
                past2_ids_left = match_to_keep_left[0]
                past2_ids = torch.cat([past2_ids, torch.tensor(left_past2_id)[past2_ids_left].to(past2_ids.device)])
                past1_ids_left = match_left[1][match_to_keep_left[0]]
                past1_ids = torch.cat([past1_ids, torch.tensor(left_past1_id)[past1_ids_left].to(past1_ids.device)])

        matched_past2 = center_points_past2[past2_ids]
        matched_past1 = center_points_past1[past1_ids]

        flow = (matched_past1 - matched_past2) 

        selected_box_3dcenter_past0 = coord_past1[past1_ids,]
        box_score = self.hist_score_global[-1][past1_ids,]
        selected_box_3dcorner_past0 = box_utils.boxes_to_corners_3d(selected_box_3dcenter_past0, order='hwl') # TODO: box order should be a parameter

        box_preds = selected_box_3dcorner_past0[:, :, :2] + flow.unsqueeze(1).repeat(1,8,1)  # (N, 8, 3)
        box_preds = torch.cat([box_preds, selected_box_3dcorner_past0[:,:,2:3]], dim=-1)
        return box_preds, selected_box_3dcorner_past0, box_score

    def updateKal(self, dets, dets_score):
        self.hist_box_global.append(dets)
        if len(self.hist_box_global) > 2:
            self.hist_box_global.pop(0)

        self.hist_score_global.append(dets_score)
        if len(self.hist_score_global) > 2:
            self.hist_score_global.pop(0)
        
    # @measure_time
    # about 0.3-0.8s except first frame
    def iterateKal(self, cur_feat_list, record_len, pairwise_t_matrix, trans_mats_dict, frame_id, colla_flag=False,
                   fuse_layer_id=0, predict_decay=0.8, prev_feat_box=None, anchors=None, hist_flag=True):

        # 1. current ego / collaborative results
        feats = cur_feat_list[fuse_layer_id]
        psm_single = self.cls_head(self.decode(cur_feat_list))
        confidence_maps = psm_single.sigmoid().max(dim=1)[0].unsqueeze(1)  # dim1=2 represents the confidence of two anchors
        if colla_flag:
            request_maps = None
            detect_feat, comm_rate, fusion_result_dict = self.fusion(feats, confidence_maps, record_len,
                                                                     pairwise_t_matrix, request_maps,
                                                                     channel_compression=False, config_thre=True)
        else:
            detect_feat = feats
        
        # 2.1 decode current observation
        feat_list = self.get_colla_feat_list(cur_feat_list, detect_feat)
        feat_list_colla = []
        for layer_id, feat in enumerate(feat_list):
            feat_list_colla.append(self.get_ego_feat(feat, record_len))

        feats_colla = self.decode(feat_list_colla)
        cls = self.cls_head(feats_colla)
        bbox = self.reg_head(feats_colla)
        dir_preds = None
        if self.use_dir:
            dir_preds = self.dir_head(feats_colla)
        bbox_temp = self.generate_predicted_boxes(cls, bbox, anchors, dir_preds)

        box_colla, box_colla_score = self.feature2box(cls, bbox_temp,
                                                    trans_mats_dict['trans_mats_track'])  # shape [N, 8, 3]

        debug_dict = {
            'box_colla': box_colla,  # current frame detection
            'box_colla_score': box_colla_score,
            'colla_feat': detect_feat.clone(),
            'colla_conf': confidence_maps.clone()
        }
            
        # 2. whether use history
        if len(self.hist_box_global) < 2:
            hist_flag = False
        if hist_flag:
            # 2.2 predict history to current timestamp
            box_preds, box_previous, box_preds_score = self.pred_box(device=bbox.device)  # shape [N, 8, 3]
            if box_preds is not None:
                box_preds = box_utils.project_box3d(box_preds, torch.inverse(trans_mats_dict['ego2world_list'][frame_id]))
                box_previous = box_utils.project_box3d(box_previous, torch.inverse(trans_mats_dict['ego2world_list'][frame_id]))
            
            # 2.3 choose to fuse box or feature 
            ############# Option1 : warp feature #########
            if prev_feat_box is not None:
                lidar_pose = torch.cat([trans_mats_dict['lidar_pose'][0][frame_id][None, ...],
                                    trans_mats_dict['lidar_pose'][0][frame_id + 1][None, ...]], dim=0)
                # flow = prev_feat_box['flow_gt'].float()
                prev_feat = self.warp_feat(prev_feat_box['prev_feat'][0][None, ...], lidar_pose)
                prev_conf = self.warp_feat(prev_feat_box['prev_conf'][0][None, ...], lidar_pose)
                
                if box_previous is not None:
                    # flow_map = self.generate_flow(box_previous, box_preds)
                    # flow_map = torch.from_numpy(flow_map).to(feats.device)[None, ...]  # shape [1, 2, H, W]
                    # out = self.warp_feat_with_flow(prev_feat, flow_map)
                    
                    flow_map, reserved_mask = self.generate_box_flow_map(box_previous, box_preds, shape_list=torch.tensor(prev_feat.shape[-3:]).to(prev_feat.device))
                    out = self.warp_w_box_flow(prev_feat, flow_map, reserved_mask)
                    # out = self.hist_fusion_net(out, feats[0].unsqueeze(0),prev_conf,0.01)
                else:
                    out = feats[0].unsqueeze(0)
                
                fused_feat_list = self.get_colla_feat_list(cur_feat_list, out)
                fused_feat_list_colla = []
                for layer_id, feat in enumerate(fused_feat_list):
                    fused_feat_list_colla.append(self.get_ego_feat(feat, record_len))
                fused_feats = self.decode(fused_feat_list_colla)
                cls = self.cls_head(fused_feats)
                bbox = self.reg_head(fused_feats)
                dir_preds = None
                if self.use_dir:
                    dir_preds = self.dir_head(fused_feats)
                bbox_temp = self.generate_predicted_boxes(cls, bbox, anchors, dir_preds)
                fused_feat_box, fused_feat_box_score = self.feature2box(cls, bbox_temp, trans_mats_dict['trans_mats_track'])

                debug_dict.update({'colla_feat': out.clone(),
                                'colla_conf': cls.sigmoid().max(dim=1)[0].unsqueeze(1).clone(),
                                'fused_feat_box': fused_feat_box.clone(),
                                'fused_feat_box_score': fused_feat_box_score.clone()})

                box_update = fused_feat_box
                box_update_score = fused_feat_box_score
            
            ############# Option2 : warp box #########
            else:
                if box_colla is not None and box_preds is not None:
                    if box_preds_score.dim() == 0:
                        box_preds_score = box_preds_score.unsqueeze(0)
                    if box_colla_score.dim() == 0:
                        box_colla_score = box_colla_score.unsqueeze(0)
                    box_preds_score *= predict_decay
                    box_update, box_update_score = self.late_fusion(box_preds, box_colla, box_preds_score, box_colla_score)
        else:
            box_update = box_colla
            box_update_score = box_colla_score
        
        box_update_save = box_update.clone()  # shape = [N, 8, 3]
        if box_update_score.dim() == 0:
            box_update_score = box_update_score.unsqueeze(0)

        # 4. kalman filter update
        box_update = box_utils.project_box3d(box_update, trans_mats_dict['ego2world_list'][frame_id]).cpu().numpy()
        self.updateKal(box_update, box_update_score)

        debug_dict.update({
            'box_update': box_update_save,  # fused result
            'box_update_score': box_update_score
        })
        
        output_box_dict = {'fused_box': box_update_save,
                           'fused_box_score': box_update_score
                           }
        return output_box_dict, debug_dict

    def warp_feat(self, feat, pose):
        _, _, H, W = feat.shape
        downsample_rate = 2
        discrete_ratio = 0.4
        frame_record_len = torch.tensor([2]).to(feat.device)
        pairwise_t_matrix = get_pairwise_transformation_torch(pose, 2, frame_record_len)

        pairwise_t_matrix = pairwise_t_matrix[:, :, :, [0, 1], :][:, :, :, :, [0, 1, 3]]  # [B, L, L, 2, 3]
        pairwise_t_matrix[..., 0, 1] = pairwise_t_matrix[..., 0, 1] * H / W
        pairwise_t_matrix[..., 1, 0] = pairwise_t_matrix[..., 1, 0] * W / H
        pairwise_t_matrix[..., 0, 2] = pairwise_t_matrix[..., 0, 2] / (
                downsample_rate * discrete_ratio * W) * 2
        pairwise_t_matrix[..., 1, 2] = pairwise_t_matrix[..., 1, 2] / (
                downsample_rate * discrete_ratio * H) * 2

        b = 0
        K = 2
        t_matrix = pairwise_t_matrix[b][:K, :K, :, :][0, 1][None, ...]  # [L, 2, 3]
        curr_feat = warp_affine_simple(feat, t_matrix, (H, W))

        return curr_feat

    def warp_w_box_flow(self, feat, flow, mask):
        updated_feat = F.grid_sample(feat, grid=flow, mode='nearest', align_corners=False)
        updated_feat = updated_feat * mask
        return updated_feat


    def generate_box_flow_map(self, box_previous, box_preds, scale=1.25, shape_list=None, align_corners=False):
        # only use x and y
        flow = (box_preds - box_previous)[:, 0, :2]
        bbox_list = box_previous[:, :4, :2]

        # scale meters to voxel, feature_length / lidar_range_length = 1.25
        flow = flow * scale
        bbox_list = bbox_list * scale

        C, H, W = shape_list
        num_cav = bbox_list.shape[0]
        basic_mat = torch.tensor([[1,0,0],[0,1,0]]).unsqueeze(0).to(torch.float32)
        basic_warp_mat = F.affine_grid(basic_mat, [1, C, H, W], align_corners=align_corners).to(shape_list.device)
        reserved_area = torch.zeros((C, H, W)).to(shape_list.device)  # C, H, W 
        if flow.shape[0] == 0 : 
            # reserved_area = torch.ones((C, H, W)).to(shape_list.device)  # C, H, W 
            return basic_warp_mat,  reserved_area.unsqueeze(0)  # 返回不变的矩阵

        flow_clone = flow.detach().clone()

        affine_matrices = torch.eye(3).unsqueeze(0).repeat(flow.shape[0], 1, 1)
        flow_clone = -2 * flow_clone / torch.tensor([W, H]).to(torch.float32).to(shape_list.device)
        # flow_clone = flow_clone[:, [1, 0]]
        affine_matrices[:, :2, 2] = flow_clone 
        
        cav_t_mat = affine_matrices[:, :2, :]   # n, 2, 3
        # print("cav_t_mat", cav_t_mat)

        cav_warp_mat = F.affine_grid(cav_t_mat,
                            [num_cav, C, H, W],
                            align_corners=align_corners).to(shape_list.device) # .to() 统一数据格式 float32
        
        flowed_bbx_list = bbox_list + flow.unsqueeze(1).repeat(1,4,1)  # n, 4, 2

        x_min = torch.min(flowed_bbx_list[:,:,0],dim=1)[0] - 1
        x_max = torch.max(flowed_bbx_list[:,:,0],dim=1)[0] + 1
        y_min = torch.min(flowed_bbx_list[:,:,1],dim=1)[0] - 1
        y_max = torch.max(flowed_bbx_list[:,:,1],dim=1)[0] + 1
        x_min_fid = (x_min + int(W/2)).to(torch.int)
        x_max_fid = (x_max + int(W/2)).to(torch.int)
        y_min_fid = (y_min + int(H/2)).to(torch.int)
        y_max_fid = (y_max + int(H/2)).to(torch.int)

        for cav in range(num_cav):
            basic_warp_mat[0,y_min_fid[cav]:y_max_fid[cav],x_min_fid[cav]:x_max_fid[cav]] = cav_warp_mat[cav,y_min_fid[cav]:y_max_fid[cav],x_min_fid[cav]:x_max_fid[cav]]

        # generate mask
        x_min_ori = torch.min(bbox_list[:,:,0],dim=1)[0]
        x_max_ori = torch.max(bbox_list[:,:,0],dim=1)[0]
        y_min_ori = torch.min(bbox_list[:,:,1],dim=1)[0]
        y_max_ori = torch.max(bbox_list[:,:,1],dim=1)[0]
        x_min_fid_ori = (x_min_ori + int(W/2)).to(torch.int)
        x_max_fid_ori = (x_max_ori + int(W/2)).to(torch.int)
        y_min_fid_ori = (y_min_ori + int(H/2)).to(torch.int)
        y_max_fid_ori = (y_max_ori + int(H/2)).to(torch.int)
        # set original location as 0
        for cav in range(num_cav):
            reserved_area[:,y_min_fid_ori[cav]:y_max_fid_ori[cav],x_min_fid_ori[cav]:x_max_fid_ori[cav]] = 0
        # set warped location as 1
        for cav in range(num_cav):
            reserved_area[:,y_min_fid[cav]:y_max_fid[cav],x_min_fid[cav]:x_max_fid[cav]] = 1

        return basic_warp_mat, reserved_area.unsqueeze(0)

    def generate_flow(self, prev_box, current_box):
        # prev_box_projected = box_utils.project_box3d(prev_box, torch.inverse(prev_ego2world))
        # current_box_projected = box_utils.project_box3d(current_box, torch.inverse(prev_ego2world))

        prev_box_projected = prev_box.cpu().numpy()
        current_box_projected = current_box.cpu().numpy()

        prev_box_center = corner_to_center(prev_box_projected, order='hwl')
        current_box_center = corner_to_center(current_box_projected, order='hwl')

        prev_box_id = list(range(len(prev_box_center)))
        current_box_id = list(range(len(current_box_center)))
        prev_object_stack, prev_object_id_stack = OrderedDict(), OrderedDict()
        prev_object_stack[0], prev_object_stack[1] = current_box_center, prev_box_center
        prev_object_id_stack[0], prev_object_id_stack[1] = current_box_id, prev_box_id

        flow = generate_flow_map(prev_object_stack, prev_object_id_stack,
                                 self.cav_lidar_range, self.voxel_size, past_k=1, current_timestamp=0)

        return flow

    @staticmethod
    def late_fusion(box_preds, box_colla, box_preds_score, box_colla_score, score_thre=0.1):
        pred_box3d_tensor = torch.cat([box_preds, box_colla], dim=0)
        scores = torch.cat([box_preds_score, box_colla_score], dim=0)
        keep_index = box_utils.nms_rotated(pred_box3d_tensor,
                                           scores,
                                           0.05
                                           )

        pred_box3d_tensor = pred_box3d_tensor[keep_index]

        # select cooresponding score
        scores = scores[keep_index]

        # filter out the prediction out of the range. with z-dim
        pred_box3d_np = pred_box3d_tensor.cpu().numpy()
        pred_box3d_np, mask = box_utils.mask_boxes_outside_range_numpy(pred_box3d_np,
                                                                       [-140.8, -40, -3, 140.8, 40, 1],
                                                                       order=None,
                                                                       return_mask=True)
        pred_box3d_tensor = torch.from_numpy(pred_box3d_np).to(device=pred_box3d_tensor.device)
        scores = scores[mask]

        assert scores.shape[0] == pred_box3d_tensor.shape[0]

        mask = torch.gt(scores, score_thre)
        keep_index = torch.where(mask)[0]
        scores = scores[keep_index]
        pred_box3d_tensor = pred_box3d_tensor[keep_index]

        return pred_box3d_tensor, scores

    def forward(self, data_dict, dataset=None):

        # self.resetKal()
        if self.process_frame_count % self.skip_scale == 0 or self.process_frame_count < 3:
            colla_flag = True
        else:
            colla_flag = False

        self.lidar = data_dict['origin_lidar'][0]
        trans_mats_track = data_dict['transformation_matrix']
        record_len = data_dict['record_len']
        pairwise_t_matrix = data_dict['pairwise_t_matrix']
        batch_dict = self.get_voxel_feat(data_dict)  # shape sum(agents) * batch

        #################### Single Encode & Decode ##################
        feat_list_single = self.encode(batch_dict['spatial_features'])
        feats_single = self.decode(feat_list_single)  # [sum(cav_num)*num_sweep_frames, 256, 100, 352]
        psm_single = self.cls_head(feats_single)
        rm_single = self.reg_head(feats_single)
        ##############################################################

        ############# Split features from diff timestamps ############
        num_sweep_frames = int(torch.div(feat_list_single[0].shape[0], record_len.sum()).item())
        lidar_pose = data_dict['lidar_pose'].view(-1, num_sweep_frames, 6)
        split_feat_list_single = []
        for layer_id, feat_single in enumerate(feat_list_single):
            BN, C, H, W = feat_single.shape
            feat_single = feat_single.view(-1, num_sweep_frames, C, H, W)
            split_feat_list_single.append(feat_single)
        psm_single = psm_single.view(-1, num_sweep_frames, psm_single.shape[-3], psm_single.shape[-2],
                                     psm_single.shape[-1])
        rm_single = rm_single.view(-1, num_sweep_frames, rm_single.shape[-3], rm_single.shape[-2], rm_single.shape[-1])

        #################### Transformation Matrix ###################
        trans_i_0_list = []  # trans matrix from timestamp i to timestamp 0, [num_sweep_frames, 4, 4], for visualization
        for i in range(num_sweep_frames):
            trans_2ego = x1_to_x2(lidar_pose[0, i].cpu().numpy(), lidar_pose[0, 0].cpu().numpy())
            trans_2ego = torch.from_numpy(trans_2ego).to(device=psm_single.device).float()
            trans_i_0_list.append(trans_2ego)
        ego2world_list = []
        for i in range(num_sweep_frames):
            tfm = pose_to_tfm(lidar_pose[0, i].unsqueeze(0))[0]
            ego2world_list.append(tfm)
        current2prev_list = []  # ego trans matrix from timestamp i to timestamp i+1， i.e. current to previous
        for i in range(num_sweep_frames - 1):
            trans_mat = x1_to_x2(lidar_pose[0, i].cpu().numpy(), lidar_pose[0, i + 1].cpu().numpy())
            trans_mat = torch.from_numpy(trans_mat).to(device=psm_single.device).float()
            current2prev_list.append(trans_mat)
        prev2current_list = []
        for i in range(1, num_sweep_frames):
            trans_mat = x1_to_x2(lidar_pose[0, i].cpu().numpy(), lidar_pose[0, i - 1].cpu().numpy())
            trans_mat = torch.from_numpy(trans_mat).to(device=psm_single.device).float()
            prev2current_list.append(trans_mat)

        trans_mats_dict = {'trans_i_0_list': trans_i_0_list,
                           'trans_mats_track': trans_mats_track,
                           'ego2world_list': ego2world_list,
                           'current2prev_list': current2prev_list,
                           'lidar_pose': lidar_pose,
                           }
        ##############################################################

        #################### Iteration each timestamp ################
        comm_rates = []
        past_i = 0
        cur_feat_list = [x[:, past_i] for x in split_feat_list_single]
        cur_pairwise_t_matrix = get_pairwise_transformation_torch(lidar_pose[:, past_i], 5, record_len)
        fused_box_dict, debug_dict = self.iterateKal(cur_feat_list, record_len, cur_pairwise_t_matrix,
                                                     trans_mats_dict, past_i,
                                                     prev_feat_box=self.prev_feat_box_dict,
                                                    #  prev_feat_box=None,
                                                     colla_flag=colla_flag,
                                                     anchors=data_dict['anchor_box']
                                                     )
        self.prev_feat_box_dict = {'prev_feat': debug_dict['colla_feat'].clone(),
                                   'prev_conf': debug_dict['colla_conf'].clone(),
                                  'prev_box': fused_box_dict['fused_box'].clone(),
                                  'prev_box_score': fused_box_dict['fused_box_score'].clone()}
                                #   'flow_gt': data_dict['flow_gt']
        # import ipdb; ipdb.set_trace()
        output_dict = {
            'box_preds': fused_box_dict['fused_box'],
            'box_preds_score': fused_box_dict['fused_box_score'],
            # 'box_preds': fused_box_dict['fused_feat_box'],
            # 'box_preds_score': fused_box_dict['fused_feat_box_score'].squeeze(),
            # 'box_preds': debug_dict['box_update'],
            # 'box_preds_score': debug_dict['box_update'].squeeze(),
            # 'box_preds': debug_dict['box_colla'],
            # 'box_preds_score': debug_dict['box_colla_score'].squeeze()
        }
        # self.vis_count += 1
        self.process_frame_count += 1

        return output_dict

    def vis_fused_feats(self, hist_fused_feats, fused_feat, flow, cls, save_name='vis_feat'):
        use_mask = False
        mask = ((cls.sigmoid()[0][0] > 0.1) * 1.0).cpu().numpy()
        save_dir = '/remote-home/share/yhu/Co_Flow/opencood/visualization/{}/{}.png'.format(save_name, self.vis_count)
        fig, axes = plt.subplots(len(hist_fused_feats[::-1][:2]) + 2, 1)
        for ax_i, cur_feat in enumerate(hist_fused_feats[::-1][:2]):
            cur_feat = hist_fused_feats[::-1][:2][ax_i].max(dim=1)[0].cpu().numpy()[0]
            if use_mask:
                cur_feat = cur_feat * mask
            sns.heatmap(cur_feat, ax=axes[ax_i], cbar=True)
            axes[ax_i].set_xticks([])
            axes[ax_i].set_yticks([])
            axes[ax_i].set_title('past_{}'.format(ax_i))
        cur_feat = fused_feat.max(dim=1)[0].cpu().numpy()[0]
        if use_mask:
            cur_feat = cur_feat * mask
        sns.heatmap(cur_feat, ax=axes[-2], cbar=True)
        axes[-2].set_xticks([])
        axes[-2].set_yticks([])
        axes[-2].set_title('fused')
        cur_feat = flow[0][0].cpu().numpy()
        sns.heatmap(cur_feat, ax=axes[-1], cbar=True)
        axes[-1].set_xticks([])
        axes[-1].set_yticks([])
        axes[-1].set_title('delta_x')
        fig.tight_layout()
        plt.savefig(save_dir)
        plt.close()

    def vis_CR_maps(self, confidence_maps_list, request_maps_list, save_name='vis_cr_maps'):
        hist_len = 0
        for t_i, cur_feat in enumerate(confidence_maps_list[hist_len:]):
            save_dir = '/remote-home/share/yhu/Co_Flow/opencood/visualization/{}/{}_{}.png'.format(save_name,
                                                                                                   self.vis_count, t_i)
            num_agents = confidence_maps_list[0].shape[0]
            fig, axes = plt.subplots(num_agents, 3, figsize=(20, 9))
            for ax_i in range(num_agents):
                conf_single = (confidence_maps_list[hist_len + t_i].cpu().numpy()[ax_i, 0] > 0.001) * 1.0
                sns.heatmap(conf_single, ax=axes[ax_i][0], cbar=True)
                axes[ax_i][0].set_xticks([])
                axes[ax_i][0].set_yticks([])
                axes[ax_i][0].set_title('t_{}_single'.format(ax_i))

                conf_whist = (request_maps_list[hist_len + t_i].cpu().numpy()[ax_i, 0] > 0.001) * 1.0
                sns.heatmap(conf_whist, ax=axes[ax_i][1], cbar=True)
                axes[ax_i][1].set_xticks([])
                axes[ax_i][1].set_yticks([])
                axes[ax_i][1].set_title('t_{}_whist'.format(ax_i))

                conf_diff = conf_whist - conf_single
                sns.heatmap(conf_diff, ax=axes[ax_i][2], cbar=True)
                axes[ax_i][2].set_xticks([])
                axes[ax_i][2].set_yticks([])
                axes[ax_i][2].set_title('t_{}_diff'.format(ax_i))
            fig.tight_layout()
            plt.savefig(save_dir)
            plt.close()

    
    def generate_predicted_boxes(self, cls_preds, deltas, anchors, dir_preds=None):
        """
        Convert the output delta to 3d bbx.

        Parameters
        ----------
        deltas : torch.Tensor
            (N, 14, H, W)
        anchors : torch.Tensor
            (W, L, 2, 7) -> xyzhwlr

        Returns
        -------
        box3d : torch.Tensor
            (N, W*L*2, 7)
        """
        # batch size
        N = deltas.shape[0]
        deltas = deltas.permute(0, 2, 3, 1).contiguous().view(N, -1, 7)
        boxes3d = torch.zeros_like(deltas)

        if deltas.is_cuda:
            anchors = anchors.cuda()
            boxes3d = boxes3d.cuda()

        # (W*L*2, 7)
        anchors_reshaped = anchors.view(-1, 7).float()
        # the diagonal of the anchor 2d box, (W*L*2)
        anchors_d = torch.sqrt(
            anchors_reshaped[:, 4] ** 2 + anchors_reshaped[:, 5] ** 2)
        anchors_d = anchors_d.repeat(N, 2, 1).transpose(1, 2)
        anchors_reshaped = anchors_reshaped.repeat(N, 1, 1)

        # Inv-normalize to get xyz
        boxes3d[..., [0, 1]] = torch.mul(deltas[..., [0, 1]], anchors_d) + \
                               anchors_reshaped[..., [0, 1]]
        boxes3d[..., [2]] = torch.mul(deltas[..., [2]],
                                      anchors_reshaped[..., [3]]) + \
                            anchors_reshaped[..., [2]]
        # hwl
        boxes3d[..., [3, 4, 5]] = torch.exp(
            deltas[..., [3, 4, 5]]) * anchors_reshaped[..., [3, 4, 5]]
        # yaw angle
        boxes3d[..., 6] = deltas[..., 6] + anchors_reshaped[..., 6]

        # adding dir classifier
        if dir_preds is not None:
            dir_offset = 0.7853
            num_bins = 2

            dm  = dir_preds # [N, H, W, 4]
            dir_cls_preds = dm.permute(0, 2, 3, 1).contiguous().reshape(1, -1, num_bins) # [1, N*H*W*2, 2]
            # if rot_gt > 0, then the label is 1, then the regression target is [0, 1]
            dir_labels = torch.max(dir_cls_preds, dim=-1)[1]  # indices. shape [1, N*H*W*2].  value 0 or 1. If value is 1, then rot_gt > 0
            
            period = (2 * np.pi / num_bins) # pi
            dir_rot = limit_period(
                boxes3d[..., 6] - dir_offset, 0, period
            ) # 限制在0到pi之间
            boxes3d[..., 6] = dir_rot + dir_offset + period * dir_labels.to(dir_cls_preds.dtype) # 转化0.25pi到2.5pi
            boxes3d[..., 6] = limit_period(boxes3d[..., 6], 0.5, 2 * np.pi) # limit to [-pi, pi]

        return boxes3d

    @staticmethod
    def feature2box(cls_preds, reg_preds, transformation_matrix, score_thre=0.2):
        pred_box3d_list = []
        pred_box2d_list = []

        prob = F.sigmoid(cls_preds.permute(0, 2, 3, 1))
        prob = prob.reshape(1, -1)

        batch_box3d = reg_preds.view(1, -1, 7)

        mask = torch.gt(prob, score_thre)
        mask = mask.view(1, -1)
        mask_reg = mask.unsqueeze(2).repeat(1, 1, 7)

        # during validation/testing, the batch size should be 1
        assert batch_box3d.shape[0] == 1
        boxes3d = torch.masked_select(batch_box3d[0], mask_reg[0]).view(-1, 7)
        scores = torch.masked_select(prob[0], mask[0])

        if len(boxes3d) != 0:
            # (N, 8, 3)
            boxes3d_corner = \
                box_utils.boxes_to_corners_3d(boxes3d, order='hwl')

            # STEP 2
            # (N, 8, 3)
            projected_boxes3d = \
                box_utils.project_box3d(boxes3d_corner,
                                        transformation_matrix)
            # convert 3d bbx to 2d, (N,4)
            projected_boxes2d = \
                box_utils.corner_to_standup_box_torch(projected_boxes3d)
            # (N, 5)
            boxes2d_score = \
                torch.cat((projected_boxes2d, scores.unsqueeze(1)), dim=1)

            pred_box2d_list.append(boxes2d_score)
            pred_box3d_list.append(projected_boxes3d)

        if len(pred_box2d_list) == 0 or len(pred_box3d_list) == 0:
            return None, None
        # shape: (N, 5)
        pred_box2d_list = torch.vstack(pred_box2d_list)
        # scores
        scores = pred_box2d_list[:, -1]
        # predicted 3d bbx
        pred_box3d_tensor = torch.vstack(pred_box3d_list)
        # remove large bbx
        keep_index_1 = box_utils.remove_large_pred_bbx(pred_box3d_tensor)
        keep_index_2 = box_utils.remove_bbx_abnormal_z(pred_box3d_tensor)
        keep_index = torch.logical_and(keep_index_1, keep_index_2)

        pred_box3d_tensor = pred_box3d_tensor[keep_index]
        scores = scores[keep_index]

        # STEP3
        # nms
        # pred_box3d_tensor.shape: (N, 8, 3)
        # scores.shape: (N,)
        keep_index = box_utils.nms_rotated(pred_box3d_tensor, scores, 0.15)

        pred_box3d_tensor = pred_box3d_tensor[keep_index]

        # select cooresponding score
        scores = scores[keep_index]

        # filter out the prediction out of the range. with z-dim
        pred_box3d_np = pred_box3d_tensor.cpu().numpy()
        pred_box3d_np, mask = box_utils.mask_boxes_outside_range_numpy(pred_box3d_np,
                                                                       [-140.8, -38.4, -3, 140.8, 38.4, 1],
                                                                       order=None,
                                                                       return_mask=True)
        pred_box3d_tensor = torch.from_numpy(pred_box3d_np).to(device=pred_box3d_tensor.device)
        scores = scores[mask]

        assert scores.shape[0] == pred_box3d_tensor.shape[0]

        return pred_box3d_tensor, scores

    def forward_ss(self, data_dict):
        record_len = data_dict['record_len']
        pairwise_t_matrix = data_dict['pairwise_t_matrix']
        batch_dict = self.get_voxel_feat(data_dict)

        #################### Single Encode & Decode ##################
        feat_list_single = self.encode(batch_dict['spatial_features'])
        feats_single = self.decode(feat_list_single)  # [sum(cav_num)*num_sweep_frames, 256, 100, 352]
        psm_single = self.cls_head(feats_single)
        rm_single = self.reg_head(feats_single)

        ############# Split features from diff timestamps ############
        num_sweep_frames = int(torch.div(feat_list_single[0].shape[0], record_len.sum()).item())
        lidar_pose = data_dict['lidar_pose'].view(-1, num_sweep_frames, 6)
        split_feat_list_single = []
        for layer_id, feat_single in enumerate(feat_list_single):
            BN, C, H, W = feat_single.shape
            feat_single = feat_single.view(-1, num_sweep_frames, C, H, W)
            split_feat_list_single.append(feat_single)
        psm_single = psm_single.view(-1, num_sweep_frames, psm_single.shape[-3], psm_single.shape[-2],
                                     psm_single.shape[-1])
        rm_single = rm_single.view(-1, num_sweep_frames, rm_single.shape[-3], rm_single.shape[-2], rm_single.shape[-1])
        ##############################################################

        curr_psm_single = self.get_ego_feat(psm_single[:, 0], record_len)
        curr_rm_single = self.get_ego_feat(rm_single[:, 0], record_len)
        bbox_temp_single = self.generate_predicted_boxes(curr_psm_single, curr_rm_single)

        output_dict = {'cls_preds': curr_psm_single,
                       'reg_preds': bbox_temp_single,
                       'bbox_preds': curr_rm_single
                       }
        return output_dict
