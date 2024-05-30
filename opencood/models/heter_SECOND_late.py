# Author: Yifan Lu <yifan_lu@sjtu.edu.cn>
# a class that integrate multiple simple fusion methods (Single Scale)
# Support F-Cooper, Self-Att, DiscoNet(wo KD), V2VNet, V2XViT, When2comm

import torch.nn as nn
from icecream import ic
from opencood.models.sub_modules.pillar_vfe import PillarVFE
from opencood.models.sub_modules.point_pillar_scatter import PointPillarScatter
from opencood.models.sub_modules.base_bev_backbone import BaseBEVBackbone
from opencood.models.sub_modules.base_bev_backbone_resnet import ResNetBEVBackbone
from opencood.models.sub_modules.downsample_conv import DownsampleConv
from opencood.models.sub_modules.naive_compress import NaiveCompressor
from opencood.models.fuse_modules.f_cooper_fuse import SpatialFusion
from opencood.models.fuse_modules.fusion_in_one import MaxFusion, AttFusion, DiscoFusion, V2VNetFusion, V2XViTFusion, When2commFusion
from opencood.utils.transformation_utils import normalize_pairwise_tfm
from opencood.models.second_ssfa_ import SecondSSFA_
from opencood.models.lift_splat_shoot import LiftSplatShoot

class HeterSECONDLate(nn.Module):
    """
    F-Cooper implementation with point pillar backbone.
    """
    def __init__(self, args):
        super(HeterSECONDLate, self).__init__()
        self.lidar_encoder_second = SecondSSFA_(args['lidar_args'])


        """backbone for each modality"""
        self.lidar_backbone_second = ResNetBEVBackbone(args['lidar_backbone'])
        
        """
        Shrink head for unify feature channels
        """
        self.shrink_lidar = DownsampleConv(args['shrink_header_lidar'])


        """
        Shared Heads
        """
        self.cls_head_lidar = nn.Conv2d(args['in_head'], args['anchor_number'],
                                  kernel_size=1)
        self.reg_head_lidar = nn.Conv2d(args['in_head'], 7 * args['anchor_number'],
                                  kernel_size=1)
        self.dir_head_lidar = nn.Conv2d(args['in_head'], args['dir_args']['num_bins'] * args['anchor_number'],
                                  kernel_size=1) # BIN_NUM = 2
        

    def forward(self, data_dict):
        if 'image_inputs' in data_dict:
            """
            Camera Encode
            """
            print("Camera encoder.")
            
        elif 'processed_lidar' in data_dict:
            """
            LiDAR Encode
            """
            voxel_features = data_dict['processed_lidar']['voxel_features']
            voxel_coords = data_dict['processed_lidar']['voxel_coords']
            voxel_num_points = data_dict['processed_lidar']['voxel_num_points']
            batch_size = voxel_coords[:,0].max() + 1


            batch_dict = {'voxel_features': voxel_features,
                        'voxel_coords': voxel_coords,
                        'voxel_num_points': voxel_num_points,
                        'batch_size': batch_size}

            batch_dict = self.lidar_encoder_second.vfe(batch_dict)
            batch_dict = self.lidar_encoder_second.spconv_block(batch_dict)
            batch_dict = self.lidar_encoder_second.map_to_bev(batch_dict) 
            lidar_feature_2d = self.lidar_backbone_second(batch_dict)['spatial_features_2d']
            lidar_feature_2d = self.shrink_lidar(lidar_feature_2d) 
            feature = lidar_feature_2d

            cls_preds = self.cls_head_lidar(feature)
            reg_preds = self.reg_head_lidar(feature)
            dir_preds = self.dir_head_lidar(feature)
            print("LiDAR encoder.")


        output_dict = {'cls_preds': cls_preds,
                       'reg_preds': reg_preds,
                       'dir_preds': dir_preds}


        return output_dict
