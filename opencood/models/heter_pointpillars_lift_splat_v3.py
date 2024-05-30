# Author: Yifan Lu <yifan_lu@sjtu.edu.cn>
# In this heterogeneous version, feature align start before backbone.
# We add self-supervise learning for view-embedding

import torch
import torch.nn as nn
import numpy as np
from icecream import ic
import copy
from opencood.models.sub_modules.point_pillar_scatter import PointPillarScatter
from opencood.models.sub_modules.base_bev_backbone_resnet import ResNetBEVBackbone 
from opencood.models.sub_modules.feature_alignnet import AlignNet, DeformAlignNet
from opencood.models.sub_modules.base_bev_backbone import BaseBEVBackbone
from opencood.models.sub_modules.downsample_conv import DownsampleConv
from opencood.models.sub_modules.naive_compress import NaiveCompressor
from opencood.models.fuse_modules.f_cooper_fuse import SpatialFusion
from opencood.models.fuse_modules.fusion_in_one import MaxFusion, AttFusion, DiscoFusion, V2VNetFusion, V2XViTFusion, When2commFusion
from opencood.models.fuse_modules.modality_aware_fusion import MAttFusion
from opencood.utils.transformation_utils import normalize_pairwise_tfm
from opencood.models.point_pillar_ import PointPillar_
from opencood.models.lift_splat_shoot import LiftSplatShoot
from opencood.data_utils.post_processor.fpvrcnn_postprocessor import FpvrcnnPostprocessor
from opencood.models.sub_modules.view_embedding import PoiExtractor

class HeterPointPillarsLiftSplatV3(nn.Module):
    def __init__(self, args):
        super(HeterPointPillarsLiftSplatV3, self).__init__()
        self.lidar_encoder = PointPillar_(args['lidar_args'])
        self.camera_encoder = LiftSplatShoot(args['camera_args'])
        self.voxel_size = args['lidar_args']['voxel_size']
        self.cav_range = args['lidar_args']['lidar_range']
        self.camera_mask_range = args['camera_mask_args']
        self.mask_ratio_W = min(self.camera_mask_range['grid_conf']['ddiscr'][1] / self.camera_mask_range['cav_lidar_range'][3], 1)
        self.mask_ratio_H = min(self.camera_mask_range['grid_conf']['ddiscr'][1] / self.camera_mask_range['cav_lidar_range'][4], 1)
        
        """light backbone for each modality"""
        self.lidar_light_backbone = ResNetBEVBackbone(args['lidar_light_backbone'])
        self.camera_light_backbone = ResNetBEVBackbone(args['camera_light_backbone'])

        """For transformation"""
        self.H = (self.cav_range[4] - self.cav_range[1])
        self.W = (self.cav_range[3] - self.cav_range[0])
        self.fake_voxel_size = 1

        """
        Fusion, by default multiscale fusion: 
        """
        self.backbone = ResNetBEVBackbone(args['fusion_backbone'])
        self.fusion_net = nn.ModuleList()
        self.modality_aware_fusion = False
        for i in range(len(args['fusion_backbone']['layer_nums'])):
            if args['fusion_method'] == "max":
                self.fusion_net.append(MaxFusion())
            if args['fusion_method'] == "att":
                self.fusion_net.append(AttFusion(args['att']['feat_dim'][i]))
            if args['fusion_method'] == "modality_aware_att":
                self.fusion_net.append(MAttFusion(args['modality_aware_att']['feat_dim'][i]))
                self.modality_aware_fusion = True


        self.shrink_flag = False
        if 'shrink_header' in args:
            self.shrink_flag = True
            self.shrink_conv = DownsampleConv(args['shrink_header'])

        """
        Domain alignment net for lidar and camera
        """
        self.lidar_aligner = AlignNet(args.get("lidar_aligner", None))
        self.camera_aligner = DeformAlignNet(args.get("camera_aligner", None))

        """
        Shared Before fusion Heads
        """
        before_fusion_dim = sum(args["lidar_light_backbone"]['num_filters'])
        self.cls_head_single = nn.Conv2d(before_fusion_dim, args['anchor_number'],
                                  kernel_size=1)
        self.reg_head_single = nn.Conv2d(before_fusion_dim, 7 * args['anchor_number'],
                                  kernel_size=1)
        self.dir_head_single = nn.Conv2d(before_fusion_dim, args['dir_args']['num_bins'] * args['anchor_number'],
                                  kernel_size=1) # BIN_NUM = 2

        """
        Shared Heads
        """
        self.cls_head = nn.Conv2d(args['in_head'], args['anchor_number'],
                                  kernel_size=1)
        self.reg_head = nn.Conv2d(args['in_head'], 7 * args['anchor_number'],
                                  kernel_size=1)
        self.dir_head = nn.Conv2d(args['in_head'], args['dir_args']['num_bins'] * args['anchor_number'],
                                  kernel_size=1) # BIN_NUM = 2

        """
        1 stage postprocessor
        """
        post_processer_args = copy.deepcopy(args['post_processer'])
        post_processer_args['target_args']['score_threshold'] = 0.5 # very confident
        self.post_processor = FpvrcnnPostprocessor(post_processer_args,
                                                   train=self.training)
        self.poi_extractor = PoiExtractor(args['poi_extractor'])

        
        if 'freeze_lidar' in args and args['freeze_lidar']:
            self.freeze_lidar()
        if 'freeze_camera' in args and args['freeze_camera']:
            self.freeze_camera()

    def freeze_lidar(self):
        for p in self.lidar_encoder.parameters():
            p.requires_grad_(False)


    def freeze_camera(self):
        for p in self.camera_encoder.parameters():
            p.requires_grad_(False)


    def forward(self, data_dict):
        output_dict = {}
        lidar_agent_indicator = data_dict['lidar_agent_record'] # [sum(record_len)]
        print(lidar_agent_indicator)
        record_len = data_dict['record_len']

        skip_lidar, skip_camera = False, False
        if sum(lidar_agent_indicator) == sum(record_len):
            skip_camera = True
        if sum(lidar_agent_indicator) == 0:
            skip_lidar = True

        """
        LiDAR Encode
        """
        if not skip_lidar:
            voxel_features = data_dict['processed_lidar']['voxel_features']
            voxel_coords = data_dict['processed_lidar']['voxel_coords']
            voxel_num_points = data_dict['processed_lidar']['voxel_num_points']
        
            batch_dict = {'voxel_features': voxel_features,
                        'voxel_coords': voxel_coords,
                        'voxel_num_points': voxel_num_points,
                        'record_len': record_len,
                        'anchor_box': data_dict['anchor_box']}

            batch_dict = self.lidar_encoder.pillar_vfe(batch_dict)
            batch_dict = self.lidar_encoder.scatter(batch_dict)
            lidar_feature_2d = batch_dict['spatial_features'] # H0, W0
            # only go to first layer , 64->128, downsample 2x
            lidar_feature_2d = self.lidar_light_backbone.get_layer_i_feature(lidar_feature_2d, 0)

            lidar_feature_2d = self.lidar_aligner(lidar_feature_2d)

        """
        Camera Encode
        """
        if not skip_camera:
            image_inputs_dict = data_dict['image_inputs']
            x, rots, trans, intrins, post_rots, post_trans = \
                image_inputs_dict['imgs'], image_inputs_dict['rots'], image_inputs_dict['trans'], image_inputs_dict['intrins'], image_inputs_dict['post_rots'], image_inputs_dict['post_trans']
            camera_feature_2d, depth_items = self.camera_encoder.get_voxels(x, rots, trans, intrins, post_rots, post_trans)  # 将图像转换到BEV下，x: B x C x H x W (4 x 64 x H x W)
            camera_feature_2d = self.camera_light_backbone.get_layer_i_feature(camera_feature_2d, 0)

            camera_feature_2d = self.camera_aligner(camera_feature_2d)

            # mask valid range
            _, _, H, W = camera_feature_2d.shape
            mask = torch.zeros((1, 1, H, W), device = camera_feature_2d.device)
            startH, endH = H/2-H/2*self.mask_ratio_H,  H/2+H/2*self.mask_ratio_H
            startW, endW = W/2-W/2*self.mask_ratio_W,  W/2+W/2*self.mask_ratio_W
            startH = np.clip(int(startH), 0, H)
            endH = np.clip(int(endH), 0, H)
            startW = np.clip(int(startW), 0, W)
            endW = np.clip(int(endW), 0, W)
            mask[:, :, startH:endH, startW:endW] = 1

            camera_feature_2d = mask * camera_feature_2d

        t_matrix = normalize_pairwise_tfm(data_dict['pairwise_t_matrix'], self.H, self.W, self.fake_voxel_size)

        """
        Heterogeneous Agent Selection
        """
        if skip_camera:
            heter_feature_2d = lidar_feature_2d
        elif skip_lidar:
            heter_feature_2d = camera_feature_2d
        else:
            _, C, H, W = camera_feature_2d.shape
            heter_feature_2d = []
            camera_idx = 0
            lidar_idx = 0
            for i in range(sum(record_len)): 
                if lidar_agent_indicator[i]:
                    heter_feature_2d.append(lidar_feature_2d[lidar_idx])
                    lidar_idx += 1
                else:
                    heter_feature_2d.append(camera_feature_2d[camera_idx])
                    camera_idx += 1
            heter_feature_2d = torch.stack(heter_feature_2d)

        cls_preds_before_fusion = self.cls_head_single(heter_feature_2d)
        reg_preds_before_fusion = self.reg_head_single(heter_feature_2d)
        dir_preds_before_fusion = self.dir_head_single(heter_feature_2d)

        if not skip_lidar:
            batch_dict['stage1_out'] = {}
            batch_dict['stage1_out']['cls_preds'] = cls_preds_before_fusion
            batch_dict['stage1_out']['reg_preds'] = reg_preds_before_fusion
            batch_dict['stage1_out']['dir_preds'] = dir_preds_before_fusion

            """
            Decode stage 1 box (only lidar agent)
            """
            data_dict, output_dict = {}, {}
            data_dict['ego'], output_dict['ego'] = batch_dict, batch_dict
            pred_box3d_list, _ = \
                self.post_processor.post_process(data_dict, output_dict,
                                                stage1=True)

            batch_dict.pop('stage1_out')
            if pred_box3d_list is not None:
                heter_feature_2d, poi_feature_pred, poi_feature_gt = \
                    self.poi_extractor(heter_feature_2d, pred_box3d_list, lidar_agent_indicator, inferring=True)
                
                output_dict.update({"poi_feature_pred" : poi_feature_pred,
                                    "poi_feature_gt" : poi_feature_gt})


        """
        Feature Fusion (multiscale).

        Suppose light_backbone only has one layer.

        Then we omit self.backbone's first layer.
        """
        feature_list = [heter_feature_2d]
        for i in range(1, len(self.fusion_net)):
            heter_feature_2d = self.backbone.get_layer_i_feature(heter_feature_2d, layer_i=i)
            feature_list.append(heter_feature_2d)

        fused_feature_list = []
        for i, fuse_module in enumerate(self.fusion_net):
            if self.modality_aware_fusion:
                fused_feature_list.append(fuse_module(feature_list[i], record_len, t_matrix, lidar_agent_indicator))
            else:
                fused_feature_list.append(fuse_module(feature_list[i], record_len, t_matrix))
        fused_feature = self.backbone.decode_multiscale_feature(fused_feature_list)
        
        if self.shrink_flag:
            fused_feature = self.shrink_conv(fused_feature)

        cls_preds = self.cls_head(fused_feature)
        reg_preds = self.reg_head(fused_feature)
        dir_preds = self.dir_head(fused_feature)

        output_dict.update({'cls_preds_single': cls_preds_before_fusion, 
                            'reg_preds_single': reg_preds_before_fusion, 
                            'dir_preds_single': dir_preds_before_fusion, 
                            'cls_preds': cls_preds,
                            'reg_preds': reg_preds,
                            'dir_preds': dir_preds})


        return output_dict
