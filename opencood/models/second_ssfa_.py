# -*- coding: utf-8 -*-
# Author: Runsheng Xu <rxx3386@ucla.edu>, OpenPCDet
# License: TDG-Attribution-NonCommercial-NoDistrib


import torch.nn as nn

from opencood.models.sub_modules.mean_vfe import MeanVFE
from opencood.models.sub_modules.sparse_backbone_3d import VoxelBackBone8x
from opencood.models.sub_modules.height_compression import HeightCompression
from opencood.models.sub_modules.base_bev_backbone import BaseBEVBackbone
from opencood.models.sub_modules.cia_ssd_utils import SSFA, Head
from opencood.models.sub_modules.downsample_conv import DownsampleConv
import numpy as np

class SecondSSFA_(nn.Module):
    def __init__(self, args):
        super(SecondSSFA_, self).__init__()
        lidar_range = np.array(args['lidar_range'])
        grid_size = np.round((lidar_range[3:6] - lidar_range[:3]) /
                             np.array(args['voxel_size'])).astype(np.int64)
        self.vfe = MeanVFE(args['mean_vfe'],
                           args['mean_vfe']['num_point_features'])
        self.spconv_block = VoxelBackBone8x(args['spconv'],
                                            input_channels=args['spconv'][
                                                'num_features_in'],
                                            grid_size=grid_size)
        self.map_to_bev = HeightCompression(args['map2bev'])

    def forward(self, data_dict):
        pass