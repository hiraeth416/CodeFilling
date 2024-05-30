import torch.nn as nn

from opencood.models.sub_modules.pillar_vfe import PillarVFE
from opencood.models.sub_modules.point_pillar_scatter import PointPillarScatter
from opencood.models.sub_modules.base_bev_backbone import BaseBEVBackbone
from opencood.models.sub_modules.base_bev_backbone_resnet import ResNetBEVBackbone
from opencood.models.sub_modules.downsample_conv import DownsampleConv
from opencood.models.sub_modules.naive_compress import NaiveCompressor
from opencood.models.sub_modules.dcn_net import DCNNet
# from opencood.models.fuse_modules.where2comm import Where2comm
from opencood.models.fuse_modules.where2comm_attn import Where2comm
from opencood.models.fuse_modules.temporal_fuse import TemporalFusion
from opencood.utils.transformation_utils import pose_to_tfm, get_pairwise_transformation_torch
import torch

class PointPillarWhere2commMultiSweepFlow(nn.Module):
    def __init__(self, args):
        super(PointPillarWhere2commMultiSweepFlow, self).__init__()

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
            self.naive_compressor = NaiveCompressor(256, args['compression'])
        
        self.dcn = False
        if 'dcn' in args:
            self.dcn = True
            self.dcn_net = DCNNet(args['dcn'])
        
        # self.fusion_net = TransformerFusion(args['fusion_args'])
        self.fusion_net = Where2comm(args['fusion_args'])
        self.temporal_fusion_net = TemporalFusion(args['fusion_args'])
        self.multi_scale = args['fusion_args']['multi_scale']

        self.cls_head = nn.Conv2d(self.out_channel, args['anchor_number'],
                                  kernel_size=1)
        self.reg_head = nn.Conv2d(self.out_channel, 7 * args['anchor_number'],
                                  kernel_size=1)
        self.use_dir = False
        if 'dir_args' in args.keys():
            self.use_dir = True
            self.dir_head = nn.Conv2d(self.out_channel, args['dir_args']['num_bins'] * args['anchor_number'],
                                  kernel_size=1) # BIN_NUM = 2
        if "backbone_fix" in args and args['backbone_fix']:
            self.backbone_fix()

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
    
    def regroup(self, x, record_len):
        cum_sum_len = torch.cumsum(record_len, dim=0)
        split_x = torch.tensor_split(x, cum_sum_len[:-1].cpu())
        return split_x

    def forward(self, data_dict):
        voxel_features = data_dict['processed_lidar']['voxel_features']
        voxel_coords = data_dict['processed_lidar']['voxel_coords']
        voxel_num_points = data_dict['processed_lidar']['voxel_num_points']
        record_len = data_dict['record_len']

        pairwise_t_matrix = data_dict['pairwise_t_matrix']

        batch_dict = {'voxel_features': voxel_features,
                      'voxel_coords': voxel_coords,
                      'voxel_num_points': voxel_num_points,
                      'record_len': record_len}
        # n, 4 -> n, c
        batch_dict = self.pillar_vfe(batch_dict)
        # n, c -> N, C, H, W
        batch_dict = self.scatter(batch_dict)
        batch_dict = self.backbone(batch_dict)
        # N, C, H', W'. [N, 384, 100, 352]
        spatial_features_2d = batch_dict['spatial_features_2d']
        
        # downsample feature to reduce memory
        if self.shrink_flag:
            spatial_features_2d = self.shrink_conv(spatial_features_2d)
        # compressor
        if self.compression:
            spatial_features_2d = self.naive_compressor(spatial_features_2d)
        # dcn
        if self.dcn:
            spatial_features_2d = self.dcn_net(spatial_features_2d)

        # spatial_features_2d is [sum(cav_num), 256, 50, 176]
        # output only contains ego
        # [B, 256, 50, 176]
        psm_single = self.cls_head(spatial_features_2d)
        rm_single = self.reg_head(spatial_features_2d)
        if self.use_dir:
            dir_single = self.dir_head(spatial_features_2d)

        # Split the features from differen timestamps
        BN, C, H, W = spatial_features_2d.shape
        num_sweep_frames = int(torch.div(BN, batch_dict['record_len'].sum()).item())
        spatial_features_2d = spatial_features_2d.view(-1, num_sweep_frames, C, H, W)
        _, C, H, W = batch_dict['spatial_features'].shape
        ori_spatial_features = batch_dict['spatial_features'].view(-1, num_sweep_frames, C, H, W)
        lidar_pose = data_dict['lidar_pose'].view(-1, num_sweep_frames, 6)
        
        psm_single = psm_single.view(-1, num_sweep_frames, psm_single.shape[-3], psm_single.shape[-2], psm_single.shape[-1])
        rm_single = rm_single.view(-1, num_sweep_frames, rm_single.shape[-3], rm_single.shape[-2], rm_single.shape[-1])

        if self.use_dir:
            dir_single = dir_single.view(-1, num_sweep_frames, dir_single.shape[-3], dir_single.shape[-2], dir_single.shape[-1])

        fused_feature_list = []
        # accumulate feature from history frames
        for past_i in range(num_sweep_frames):
            cur_pairwise_t_matrix = get_pairwise_transformation_torch(lidar_pose[:, past_i], 5, record_len)
            # print('spatial_features_2d: ', spatial_features_2d.shape)
            if self.multi_scale:
                fused_feature, communication_rates, result_dict = self.fusion_net(ori_spatial_features[:, past_i],
                                                psm_single[:, past_i],
                                                record_len,
                                                cur_pairwise_t_matrix, 
                                                self.backbone)
                # downsample feature to reduce memory
                if self.shrink_flag:
                    fused_feature = self.shrink_conv(fused_feature).unsqueeze(1)
            else:
                fused_feature, communication_rates, result_dict = self.fusion_net(spatial_features_2d[:, past_i],
                                                psm_single[:, past_i],
                                                record_len,
                                                cur_pairwise_t_matrix)
            fused_feature_list.append(fused_feature.unsqueeze(1))
            
        # fuse history
        lidar_pose_list = self.regroup(lidar_pose, record_len)
        ego_lidar_poses = torch.cat([lidar_pose_list[i][0] for i in range(len(record_len))], dim=0)
        frame_record_len = torch.tensor([num_sweep_frames for _ in range(len(record_len))]).to(record_len.device)
        cur_pairwise_t_matrix = get_pairwise_transformation_torch(ego_lidar_poses, num_sweep_frames, frame_record_len)
        fused_features = torch.cat(fused_feature_list, dim=1).transpose(1,0).flatten(0,1)
        if len(fused_features.shape) > 4:
            fused_features = fused_features.squeeze(1)
        flow, state_class_pred, fused_feature = self.temporal_fusion_net(fused_features, frame_record_len, cur_pairwise_t_matrix, data_dict)
        # fused_feature = fused_features[0:1]

        # fuse current observation with history
        split_feats = self.regroup(spatial_features_2d, record_len)
        curr_ego_feat = [feat[0:1,0] for feat in split_feats]
        curr_ego_feat = torch.cat(curr_ego_feat, dim=0) if len(curr_ego_feat)>1 else curr_ego_feat[0]
        fused_feature = torch.cat([fused_feature.unsqueeze(1), curr_ego_feat.unsqueeze(1)], dim=1).max(dim=1)[0]
            
        # print('fused_feature: ', fused_feature.shape)
        psm = self.cls_head(fused_feature)
        rm = self.reg_head(fused_feature)
        

        output_dict = {'cls_preds': psm,
                       'reg_preds': rm}
        if self.use_dir:
            output_dict.update({'dir_preds': self.dir_head(fused_feature),
                                'dir_preds_single': dir_single[:,0]})

        output_dict.update(result_dict)

        output_dict.update({'cls_preds_single': psm_single[:,0],
                       'reg_preds_single': rm_single[:,0],
                       'comm_rate': communication_rates
                       })
        return output_dict