# Author: Yifan Lu <yifan_lu@sjtu.edu.cn>
# In this heterogeneous version, feature align start before backbone.

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
from icecream import ic
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
from opencood.models.comm_modules.where2comm import Communication
from opencood.models.sub_modules.codebook import ChannelCompressor
from opencood.models.sub_modules.codebook import UMGMQuantizer

class HeterPointPillarsLiftSplatV2withfeature(nn.Module):
    def __init__(self, args):
        super(HeterPointPillarsLiftSplatV2withfeature, self).__init__()        

        """
        Codebook Part
        """
        self.multi_channel_compressor_flag = False
        if 'multi_channel_compressor' in args and args['multi_channel_compressor']:
            self.multi_channel_compressor_flag = True

        """
        Get feature
        """
        self.get_feature_flag = False
        if 'get_feature' in args:
            self.get_feature_flag = True
            print("get featrue: true")


        channel = 64
        p_rate = 0.0
        seg_num = args['codebook']['seg_num']
        dict_size = [args['codebook']['dict_size'], args['codebook']['dict_size'], args['codebook']['dict_size']]
        self.multi_channel_compressor = UMGMQuantizer(channel, seg_num, dict_size, p_rate,
                          {"latentStageEncoder": lambda: nn.Linear(channel, channel), "quantizationHead": lambda: nn.Linear(channel, channel),
                           "latentHead": lambda: nn.Linear(channel, channel), "restoreHead": lambda: nn.Linear(channel, channel),
                           "dequantizationHead": lambda: nn.Linear(channel, channel), "sideHead": lambda: nn.Linear(channel, channel)})
        print(self.multi_channel_compressor_flag)
        print("seg_num: ", seg_num)        
        print("dict_size: ", args['codebook']['dict_size'])

        

    def regroup(self, x, record_len):
        #print(x)
        cum_sum_len = torch.cumsum(record_len, dim=0)
        split_x = torch.tensor_split(x, cum_sum_len[:-1].cpu())
        return split_x
        
    


    def forward(self, heter_feature_2d):
        #save_path = "/GPFS/rhome/sifeiliu/OpenCOODv2/opencood/logs/feature_folder/"
        #heter_feature_2d = torch.load(os.path.join(save_path,'feature%d.pt' % (num)))
        #record_len = torch.load(os.path.join(save_path,'record_len%d.pt' % (num)))


        #cls_preds_before_fusion = self.cls_head_single(heter_feature_2d)
        #reg_preds_before_fusion = self.reg_head_single(heter_feature_2d)
        #dir_preds_before_fusion = self.dir_head_single(heter_feature_2d)


        """
        Codebook Part
        """
        print("------------Codebook Part---------------")
        #print("feature_shape: ", heter_feature_2d.shape)
        _, C, H, W = heter_feature_2d.shape
        parameter_dict = {}
        # import pdb
        # pdb.set_trace()
        if self.multi_channel_compressor_flag:
            heter_feature_2d_gt = heter_feature_2d.clone()
            #print("if equal: ", heter_feature_2d_gt.equal(heter_feature_2d))
            heter_feature_2d = heter_feature_2d.permute(0, 2, 3, 1).contiguous().view(-1, C)
            heter_feature_2d, _, _, codebook_loss = self.multi_channel_compressor(heter_feature_2d)
                
            #print("heter_feature_2d[0]_shape: ", heter_feature_2d[0].shape)
            parameter_dict.update({'codebook_loss': codebook_loss})
            parameter_dict.update({'heter_feature_2d_gt': heter_feature_2d_gt})

        
        print("------------Codebook Part---------------")
        
        """
        Feature Fusion (multiscale).

        Suppose light_backbone only has one layer.

        Then we omit self.backbone's first layer.
        """
        
        """
        feature_list = [heter_feature_2d]
        for i in range(1, len(self.fusion_net)):
            heter_feature_2d = self.backbone.get_layer_i_feature(heter_feature_2d, layer_i=i)
            feature_list.append(heter_feature_2d)
        batch_confidence_maps = self.regroup(cls_preds_before_fusion, record_len)
        #print(batch_confidence_maps)
        _, communication_masks, communication_rates = self.naive_communication(batch_confidence_maps, record_len, pairwise_t_matrix)
        for i in range(len(feature_list)):
              #print(x.size())
              #print("communication_rates")
              #print(communication_rates)
              feature_list[i] = feature_list[i] * communication_masks
              communication_masks = F.max_pool2d(communication_masks, kernel_size=2)
              #print(communication_masks.size())
        fused_feature_list = []
        for i, fuse_module in enumerate(self.fusion_net):
            if self.modality_aware_fusion[i]:
                fused_feature_list.append(fuse_module(feature_list[i], record_len, t_matrix, lidar_agent_indicator))
            else:
                fused_feature_list.append(fuse_module(feature_list[i], record_len, t_matrix))
        fused_feature = self.backbone.decode_multiscale_feature(fused_feature_list)
        
        if self.shrink_flag:
            fused_feature = self.shrink_conv(fused_feature)

        cls_preds = self.cls_head(fused_feature)
        reg_preds = self.reg_head(fused_feature)
        dir_preds = self.dir_head(fused_feature)
        """

        # from matplotlib import pyplot as plt
        # if sum(lidar_agent_indicator) >= 1 and sum(1-lidar_agent_indicator) >= 1:
        #     import torch.nn.functional as F
        #     cls_preds_before_fusion_np = F.sigmoid((cls_preds_before_fusion.max(dim=1)[0] > 0.15).int()).cpu().numpy() 
        #     cls_preds_np = F.sigmoid((cls_preds.max(dim=1)[0] > 0.15).int()).cpu().numpy() 
        #     lidar_feature_2d_np = lidar_feature_2d.cpu().numpy()
        #     camera_feature_2d_np = camera_feature_2d.cpu().numpy()
        #     i = np.random.randint(100)
        #     for ch in range(64):
        #         vmax = 4
        #         vmin = -0.2
        #         fig, axes = plt.subplots(4,2)

        #         im = axes[0,0].imshow(lidar_feature_2d_before_fusion_np[0, ch], vmin=vmin, vmax=vmax)
        #         # for ii in range(lidar_roi_features.shape[-2]):
        #         #     for jj in range(lidar_roi_features.shape[-1]):
        #         #         text = axes[0,0].text(jj, ii, '%.3f' % lidar_roi_features_np[0,ch][ii,jj],
        #         #                             ha="center", va="center", color="w", fontsize=5)
        #         plt.colorbar(im, ax=axes[0,0])

        #         im = axes[0,1].imshow(camera_feature_2d_before_fusion_np[0, ch], vmin=vmin, vmax=vmax)
        #         # for ii in range(camera_roi_features_np.shape[-2]):
        #         #     for jj in range(camera_roi_features_np.shape[-1]):
        #         #         text = axes[0,1].text(jj, ii, '%.3f' % camera_roi_features_np[0,ch][ii,jj],
        #         #                             ha="center", va="center", color="w", fontsize=5)
        #         plt.colorbar(im, ax=axes[0,1])

        #         im = axes[1,0].imshow(lidar_feature_2d_np[0, ch], vmin=vmin, vmax=vmax)
        #         # for ii in range(lidar_roi_features.shape[-2]):
        #         #     for jj in range(lidar_roi_features.shape[-1]):
        #         #         text = axes[1,0].text(jj, ii, '%.3f' % lidar_roi_features_np[-1,ch][ii,jj],
        #         #                             ha="center", va="center", color="w", fontsize=5)
        #         plt.colorbar(im, ax=axes[1,0])

        #         im = axes[1,1].imshow(camera_feature_2d_np[0, ch], vmin=vmin, vmax=vmax)
        #         # for ii in range(camera_roi_features_np.shape[-2]):
        #         #     for jj in range(camera_roi_features_np.shape[-1]):
        #         #         text = axes[1,1].text(jj, ii, '%.3f' % camera_roi_features_np[-1,ch][ii,jj],
        #         #                             ha="center", va="center", color="w", fontsize=5)
        #         plt.colorbar(im, ax=axes[1,1])

        #         im = axes[2,0].imshow(cls_preds_before_fusion_np[0])
        #         # for ii in range(camera_roi_features_np.shape[-2]):
        #         #     for jj in range(camera_roi_features_np.shape[-1]):
        #         #         text = axes[1,1].text(jj, ii, '%.3f' % camera_roi_features_np[-1,ch][ii,jj],
        #         #                             ha="center", va="center", color="w", fontsize=5)
        #         plt.colorbar(im, ax=axes[2,0])

        #         im = axes[2,1].imshow(cls_preds_before_fusion_np[1])
        #         # for ii in range(camera_roi_features_np.shape[-2]):
        #         #     for jj in range(camera_roi_features_np.shape[-1]):
        #         #         text = axes[1,1].text(jj, ii, '%.3f' % camera_roi_features_np[-1,ch][ii,jj],
        #         #                             ha="center", va="center", color="w", fontsize=5)
        #         plt.colorbar(im, ax=axes[2,1])

        #         im = axes[3,0].imshow(cls_preds_np[0])
        #         # for ii in range(camera_roi_features_np.shape[-2]):
        #         #     for jj in range(camera_roi_features_np.shape[-1]):
        #         #         text = axes[1,1].text(jj, ii, '%.3f' % camera_roi_features_np[-1,ch][ii,jj],
        #         #                             ha="center", va="center", color="w", fontsize=5)
        #         plt.colorbar(im, ax=axes[3,1])


        #         plt.savefig(f"/GPFS/rhome/yifanlu/workspace/OpenCOODv2/vis_result/layer1_heter_feature_before_after_align_sdta_6layers_singlesup_confmap/{i}_{ch}.jpg", dpi=400)
        #         plt.clf()
        #         plt.close()
        #         print(ch)
        # raise

        """
        output_dict = {'cls_preds_single': cls_preds_before_fusion, 
                       'reg_preds_single': reg_preds_before_fusion, 
                       'dir_preds_single': dir_preds_before_fusion, 
                       'cls_preds': cls_preds,
                       'reg_preds': reg_preds,
                       'dir_preds': dir_preds,
                       'comm_rates': communication_rates,
                       'codebook_loss': codebook_loss}
        """
        
        output_dict = {'codebook_loss': codebook_loss}

        return output_dict
