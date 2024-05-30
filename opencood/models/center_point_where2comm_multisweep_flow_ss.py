import torch.nn as nn
import numpy as np
from opencood.models.sub_modules.pillar_vfe import PillarVFE
from opencood.models.sub_modules.point_pillar_scatter import PointPillarScatter
from opencood.models.sub_modules.base_bev_backbone import BaseBEVBackbone
from opencood.models.sub_modules.base_bev_backbone_resnet import ResNetBEVBackbone
from opencood.models.sub_modules.downsample_conv import DownsampleConv
from opencood.models.sub_modules.naive_compress import NaiveCompressor
from opencood.models.sub_modules.dcn_net import DCNNet
from opencood.models.sub_modules.SyncLSTM import SyncLSTM
# from opencood.models.fuse_modules.where2comm import Where2comm
from opencood.models.fuse_modules.where2comm_attn import Where2comm
from opencood.models.fuse_modules.temporal_fuse import TemporalFusion
from opencood.utils.transformation_utils import pose_to_tfm, get_pairwise_transformation_torch
import torch
from opencood.models.fuse_modules.max_fuse import MaxFusion

class CenterPointWhere2commMultiSweepFlow(nn.Module):
    def __init__(self, args):
        super(CenterPointWhere2commMultiSweepFlow, self).__init__()

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
        self.out_size_factor = args['out_size_factor']
        self.cav_lidar_range  = args['lidar_range']

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
        self.temporal_compensation_net = TemporalFusion(args['fusion_args'])
        self.colla_fusion_net = MaxFusion(args['fusion_args'])
        self.multi_scale = args['fusion_args']['multi_scale']

        self.cls_head = nn.Conv2d(self.out_channel, args['anchor_number'],
                                  kernel_size=1)
        self.reg_head = nn.Conv2d(self.out_channel, 8 * args['anchor_number'],
                                  kernel_size=1)
        if 'backbone_fix' in args.keys() and args['backbone_fix']:
            self.backbone_fix()
        
        self.init_weight()
        self.debug = False # True
        self.vis_count = 0
    
    def init_weight(self):
        pi = 0.01
        nn.init.constant_(self.cls_head.bias, -np.log((1 - pi) / pi) )
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
        curr_ego_feat = torch.cat(curr_ego_feat, dim=0) if len(curr_ego_feat)>1 else curr_ego_feat[0]
        return curr_ego_feat    # [B, C, H, W]
    
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

    def forward(self, data_dict):
        record_len = data_dict['record_len']
        pairwise_t_matrix = data_dict['pairwise_t_matrix']
        batch_dict = self.get_voxel_feat(data_dict)

        #################### Single Encode & Decode ##################
        feat_list_single = self.encode(batch_dict['spatial_features'])
        feats_single = self.decode(feat_list_single)   # [sum(cav_num)*num_sweep_frames, 256, 100, 352]
        psm_single = self.cls_head(feats_single)
        rm_single = self.reg_head(feats_single)
        ##############################################################

        ############# Split features from diff timestamps ############
        num_sweep_frames = int(torch.div(feat_list_single[0].shape[0],record_len.sum()).item())
        lidar_pose = data_dict['lidar_pose'].view(-1, num_sweep_frames, 6)
        split_feat_list_single = []
        for layer_id, feat_single in enumerate(feat_list_single):
            BN, C, H, W = feat_single.shape
            feat_single = feat_single.view(-1, num_sweep_frames, C, H, W)
            split_feat_list_single.append(feat_single)
        psm_single = psm_single.view(-1, num_sweep_frames, psm_single.shape[-3], psm_single.shape[-2], psm_single.shape[-1])
        rm_single = rm_single.view(-1, num_sweep_frames, rm_single.shape[-3], rm_single.shape[-2], rm_single.shape[-1])
        ##############################################################
        
        ################# Fuse features at each timestamp ############
        fuse_layer_id = 0
        spatial_features_2d = split_feat_list_single[fuse_layer_id]
        feat_list_fused = []
        for past_i in range(num_sweep_frames):
            cur_pairwise_t_matrix = get_pairwise_transformation_torch(lidar_pose[:, past_i], 5, record_len)
            fused_feature, communication_rates, result_dict = self.fusion_net(spatial_features_2d[:, past_i],
                                            psm_single[:, past_i],
                                            record_len,
                                            cur_pairwise_t_matrix)
            feat_list_fused.append(fused_feature.unsqueeze(1))
        ##############################################################
            
        ############### Flow-based History Compensation ##############
        lidar_pose_list = self.regroup(lidar_pose, record_len)
        ego_lidar_poses = torch.cat([lidar_pose_list[i][0] for i in range(len(record_len))], dim=0)
        frame_record_len = torch.tensor([num_sweep_frames for _ in range(len(record_len))]).to(record_len.device)
        cur_pairwise_t_matrix = get_pairwise_transformation_torch(ego_lidar_poses, num_sweep_frames, frame_record_len)
        hist_fused_feats = torch.cat(feat_list_fused, dim=1).transpose(1,0).flatten(0,1)  # (B, T, C, H, W)
        if len(hist_fused_feats.shape) > 4:
            hist_fused_feats = hist_fused_feats.squeeze(1)
        flow, state_class_pred, compensated_fused_feat = self.temporal_compensation_net(hist_fused_feats, frame_record_len, cur_pairwise_t_matrix, data_dict)
        ##############################################################

        ############## Fuse history with current observation #########
        curr_ego_feat = self.get_ego_feat(split_feat_list_single[fuse_layer_id][:,0], record_len)
        fused_feat = torch.cat([compensated_fused_feat.unsqueeze(1), curr_ego_feat.unsqueeze(1)], dim=1).max(dim=1)[0]
        ##############################################################

        ################## Decode compensated features ###############
        # feat_list = []
        # for layer_id, feat in enumerate(split_feat_list_single):
        #     if layer_id == fuse_layer_id:
        #         feat_list.append(fused_feat)
        #     else:
        #         feat_list.append(self.get_ego_feat(split_feat_list_single[layer_id][:,0], record_len))
        # feats_compensated = self.decode(feat_list)
        ##############################################################

        ############### Collaboration at current timestamp ###########
        # fuse observations from other agents with ego --> ms-max
        # fused_feature = torch.cat([fused_feature.unsqueeze(1), fused_feature_list[0].squeeze(1)], dim=1).max(dim=1)[0]

        # fuse observations from other agents with ego --> max
        split_feats = self.regroup(split_feat_list_single[fuse_layer_id][:,0], record_len)
        curr_feats = [torch.cat([fused_feat[i:i+1],feat[1:]], dim=0) for i, feat in enumerate(split_feats)]
        curr_feats = torch.cat(curr_feats, dim=0)
        colla_feat = self.colla_fusion_net(curr_feats, record_len, pairwise_t_matrix)
        ##############################################################

        ################## Decode collaborated features ##############
        feat_list_colla = []
        for layer_id, feat in enumerate(split_feat_list_single):
            if layer_id == fuse_layer_id:
                feat_list_colla.append(colla_feat)
            else:
                feat_list_colla.append(self.get_ego_feat(split_feat_list_single[layer_id][:,0], record_len))
        feats_colla = self.decode(feat_list_colla)
        ##############################################################
        
        # print('fused_feature: ', fused_feature.shape)
        cls = self.cls_head(feats_colla)
        bbox = self.reg_head(feats_colla)

        DEBUG = self.debug
        if DEBUG:
            use_mask = True
            mask = ((cls.sigmoid()[0][0] > 0.1)*1.0).cpu().numpy()
            import seaborn as sns
            import matplotlib.pyplot as plt
            save_dir = '/GPFS/data/yhu/code/OpenCOODv2/opencood/visualization/vis_feat/{}.png'.format(self.vis_count)
            fig, axes = plt.subplots(len(fused_feature_list[:2])+2, 1)
            for ax_i, cur_feat in enumerate(fused_feature_list[:2]):
                cur_feat = fused_feature_list[ax_i][0,0].max(dim=1)[0].cpu().numpy()[0]
                if use_mask:
                    cur_feat = cur_feat * mask
                sns.heatmap(cur_feat, ax=axes[ax_i], cbar=True)
                axes[ax_i].set_xticks([])
                axes[ax_i].set_yticks([])
                axes[ax_i].set_title('past_{}'.format(ax_i))
            cur_feat = fused_feature.max(dim=1)[0].cpu().numpy()[0]
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
            self.vis_count += 1

        _, bbox_temp = self.generate_predicted_boxes(cls, bbox)

        output_dict = {'cls_preds': cls,
                       'reg_preds': bbox_temp,
                       'bbox_preds': bbox,
                       'flow_preds': flow,
                       'state_preds': state_class_pred
                       }
        output_dict.update(result_dict)

        _, bbox_temp_single = self.generate_predicted_boxes(psm_single[:,0], rm_single[:,0])

        output_dict.update({'cls_preds_single': psm_single[:,0],
                       'reg_preds_single': bbox_temp_single,
                       'bbox_preds_single': rm_single[:,0],
                       'comm_rate': communication_rates
                       })
        return output_dict

    def generate_predicted_boxes(self, cls_preds, box_preds, dir_cls_preds=None):
        """
        Args:
            batch_size:
            cls_preds: (N, H, W, C1)
            box_preds: (N, H, W, C2)
            dir_cls_preds: (N, H, W, C3)

        Returns:
            batch_cls_preds: (B, num_boxes, num_classes)
            batch_box_preds: (B, num_boxes, 7+C)

        """
        box_preds = box_preds.permute(0, 2, 3, 1).contiguous()
        
        batch, H, W, code_size = box_preds.size()   ## code_size 表示的是预测的尺寸
        
        box_preds = box_preds.reshape(batch, H*W, code_size)

        batch_reg = box_preds[..., 0:2]
        # batch_hei = box_preds[..., 2:3] 
        # batch_dim = torch.exp(box_preds[..., 3:6])
        
        h = box_preds[..., 3:4] * self.out_size_factor * self.voxel_size[0]
        w = box_preds[..., 4:5] * self.out_size_factor * self.voxel_size[1]
        l = box_preds[..., 5:6] * self.out_size_factor * self.voxel_size[2]
        batch_dim = torch.cat([h,w,l], dim=-1)
        batch_hei = box_preds[..., 2:3] * self.out_size_factor * self.voxel_size[2] + self.cav_lidar_range[2]

        batch_rots = box_preds[..., 6:7]
        batch_rotc = box_preds[..., 7:8]

        rot = torch.atan2(batch_rots, batch_rotc)

        ys, xs = torch.meshgrid([torch.arange(0, H), torch.arange(0, W)])
        ys = ys.view(1, H, W).repeat(batch, 1, 1).to(cls_preds.device)
        xs = xs.view(1, H, W).repeat(batch, 1, 1).to(cls_preds.device)

        xs = xs.view(batch, -1, 1) + batch_reg[:, :, 0:1]
        ys = ys.view(batch, -1, 1) + batch_reg[:, :, 1:2]

        xs = xs * self.out_size_factor * self.voxel_size[0] + self.cav_lidar_range[0]   ## 基于feature_map 的size求解真实的坐标
        ys = ys * self.out_size_factor * self.voxel_size[1] + self.cav_lidar_range[1]


        batch_box_preds = torch.cat([xs, ys, batch_hei, batch_dim, rot], dim=2)
        # batch_box_preds = batch_box_preds.reshape(batch, H, W, batch_box_preds.shape[-1])
        # batch_box_preds = batch_box_preds.permute(0, 3, 1, 2).contiguous()

        # batch_box_preds_temp = torch.cat([xs, ys, batch_hei, batch_dim, rot], dim=1)
        # box_preds = box_preds.permute(0, 3, 1, 2).contiguous()

        # batch_cls_preds = cls_preds.view(batch, H*W, -1)
        return cls_preds, batch_box_preds
