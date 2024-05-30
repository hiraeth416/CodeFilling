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
from opencood.models.fuse_modules.where2comm import Where2comm
# from opencood.models.fuse_modules.where2comm_attn import Where2comm
# from opencood.models.fuse_modules.temporal_fuse import TemporalFusion
from opencood.models.sub_modules.temporal_compensation import TemporalCompensation
from opencood.utils.transformation_utils import pose_to_tfm, get_pairwise_transformation_torch
import torch
from opencood.models.fuse_modules.max_fuse import MaxFusion, SMaxFusion
from opencood.models.fuse_modules.pointwise_fuse import PointwiseFusion
import seaborn as sns
import matplotlib.pyplot as plt


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
        # self.temporal_fusion_net = TemporalFusion(args['fusion_args'])
        self.temporal_fusion_net = TemporalCompensation(args['fusion_args'])
        self.colla_fusion_net = MaxFusion(args['fusion_args'])
        self.multi_scale = args['fusion_args']['multi_scale']
        self.hist_len = args['hist_len']
        
        temporal_fusion_mode = args['fusion_args']['temporal_fusion'] if 'temporal_fusion' in args['fusion_args'] else 'max'
        if temporal_fusion_mode == 'attn':
            self.hist_fusion_net = PointwiseFusion(args['fusion_args'])
        else:
            self.hist_fusion_net = SMaxFusion()
        
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
        
        print('only_hist: ', self.only_hist)
        print('with_hist: ', self.with_hist)
        print('with_compensation: ', self.with_compensation)
        print('wo_colla: ', self.wo_colla)

        self.cls_head = nn.Conv2d(self.out_channel, args['anchor_number'],
                                  kernel_size=1)
        self.reg_head = nn.Conv2d(self.out_channel, 8 * args['anchor_number'],
                                  kernel_size=1)
        if 'backbone_fix' in args.keys() and args['backbone_fix']:
            self.backbone_fix()
        
        self.init_weight()
        self.debug = False # True # 
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
    
    def compensate(self, hist_fused_feats, lidar_pose, flow_gt=None):
        '''
        Function: compensate historical features for each agent.
        Input:
            hist_fused_feats: [N, T, C, H, W]
            lidar_pose: [N, T, 6]
            flow_gt: include GT flow
        Output:
            compensated_fused_feats: [N, C, H, W]
        '''
        cur_hist_fused_feats = torch.cat([x.unsqueeze(1) for x in hist_fused_feats[::-1][:5]], dim=1).flatten(0,1)
        cur_lidar_pose = torch.cat([x.unsqueeze(1) for x in lidar_pose[::-1][:5]], dim=1)
        N, T, _ = cur_lidar_pose.shape
        frame_record_len = torch.tensor([T for _ in range(N)]).to(cur_hist_fused_feats.device)
        lidar_pose_T = cur_lidar_pose.flatten(0,1) # (N*T, 6)
        pairwise_t_matrix = get_pairwise_transformation_torch(lidar_pose_T, T, frame_record_len)
        flow, state_class_pred, compensated_fused_feat = self.temporal_fusion_net(cur_hist_fused_feats, frame_record_len, pairwise_t_matrix, flow_gt)
        return flow, state_class_pred, compensated_fused_feat
    
    def temporal_fusion(self, cur_feat, hist_feat):
        '''
        Function: fuse the current observation with historical feature.
        Input:
            hist_feat: [N, C, H, W]
            cur_feat: [N, C, H, W]
        Output:
            fused_feat: [N, C, H, W]
        '''
        # fused_feat = torch.cat([cur_feat.unsqueeze(1), hist_feat.unsqueeze(1)], dim=1).max(dim=1)[0]
        fused_feat = self.hist_fusion_net(cur_feat, hist_feat)
        return fused_feat

    def fusion(self, feat, conf_map, record_len, pairwise_t_matrix, request_map=None):
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
                                            request_map)
        return fused_feature, comm_rates, result_dict

    def get_colla_feat_list(self, feat_list_single, fused_feat, fuse_layer_id=0):
        feat_list_colla = []
        for layer_id, feat in enumerate(feat_list_single):
            if layer_id == fuse_layer_id:
                feat_list_colla.append(fused_feat)
            else:
                feat_list_colla.append(feat)
        return feat_list_colla
        
    def iterate(self, cur_feat_list, lidar_pose, record_len, pairwise_t_matrix, wo_colla=False, prev_confidence_maps=None, hist_fused_feats=None, flow_gt=None, fuse_layer_id=0):
        # 1. temporal compensation & fusion
        if hist_fused_feats is not None:
            # temporal compensation
            flow, state_class_pred, compensated_fused_feat = self.compensate(hist_fused_feats, lidar_pose, flow_gt)
            
            hist_feats = compensated_fused_feat if self.with_compensation else hist_fused_feats[-1]            
            curr_feats = cur_feat_list[fuse_layer_id]

            if self.only_hist:
                feats = hist_feats
            elif self.with_hist:
                feats = self.temporal_fusion(curr_feats, hist_feats)
            else:
                feats = curr_feats
            
            feats_list = [hist_feats, feats]
        else:
            feats = cur_feat_list[fuse_layer_id]
            flow = None
            state_class_pred = None
            feats_list = []

        # 2. generate confidence maps # [sum(cav_num)*num_sweep_frames, 256, 100, 352]
        psm_single = self.cls_head(self.decode(cur_feat_list))
        feat_list = self.get_colla_feat_list(cur_feat_list, feats)
        feats_single = self.decode(feat_list)   # [sum(cav_num)*num_sweep_frames, 256, 100, 352]
        psm_single_whist = self.cls_head(feats_single)
        # rm_single = self.reg_head(feats_single)

        # 3. communication and fusion
        confidence_maps = psm_single.sigmoid().max(dim=1)[0].unsqueeze(1) # dim1=2 represents the confidence of two anchors
        confidence_maps_whist = psm_single_whist.sigmoid().max(dim=1)[0].unsqueeze(1) # dim1=2 represents the confidence of two anchors
        # request_maps = 1 - prev_confidence_maps if prev_confidence_maps is not None else None
        if hist_fused_feats is not None:
            # confidence_maps = torch.zeros_like(psm_single.max(dim=1)[0].unsqueeze(1)).to(psm_single.device)
            # request_maps = torch.zeros_like(confidence_maps).to(confidence_maps.device)
            # request_maps = 1 - prev_confidence_maps if prev_confidence_maps is not None else None
            request_maps = 1 - confidence_maps_whist
        else:
            request_maps = None
        
        if (hist_fused_feats is not None) and wo_colla:
            fused_feat = feats
            comm_rate = torch.tensor(0.0).to(feats.device)
        else:
            fused_feat, comm_rate, _ = self.fusion(feats, confidence_maps, record_len, pairwise_t_matrix, request_maps)
        # return fused_feat, comm_rate, confidence_maps, flow, state_class_pred
        if len(feats_list) == 0:
            return fused_feat, comm_rate, confidence_maps, confidence_maps_whist, flow, state_class_pred
        else:
            return [fused_feat]+feats_list, comm_rate, confidence_maps, confidence_maps_whist, flow, state_class_pred

    
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

        #################### Iteration each timestamp ################
        prev_confidence_maps = None
        hist_fused_feats = []
        lidar_pose_list = []
        hist_len = self.hist_len
        comm_rates = []
        confidence_maps_list = []
        request_maps_list = []
        for past_i in range(num_sweep_frames)[::-1]:
            # print('timestamp: ', past_i)
            cur_feat_list = [x[:,past_i] for x in split_feat_list_single]
            cur_pairwise_t_matrix = get_pairwise_transformation_torch(lidar_pose[:, past_i], 5, record_len)
            if len(hist_fused_feats) >= hist_len:
                wo_colla = self.wo_colla if len(hist_fused_feats)%self.sampling_gap == 0 else True
                # print('past_i: ', past_i, 'wo_colla: ', wo_colla)
                fused_feat, comm_rate, confidence_maps_single, confidence_maps, flow, state_class_pred = self.iterate(cur_feat_list, lidar_pose_list, record_len, cur_pairwise_t_matrix, wo_colla, prev_confidence_maps, hist_fused_feats, flow_gt=data_dict['flow_gt'])
                ego_flow = self.get_ego_feat(flow, record_len) if flow is not None else flow
                ego_state_class_pred = self.get_ego_feat(state_class_pred, record_len)  if state_class_pred is not None else state_class_pred
            else:
                fused_feat, comm_rate, confidence_maps_single, confidence_maps, flow, state_class_pred = self.iterate(cur_feat_list, lidar_pose_list, record_len, cur_pairwise_t_matrix, wo_colla=False)
                ego_flow = None
                ego_state_class_pred = None
            if isinstance(fused_feat, list):
                hist_fused_feats.append(fused_feat[0])
            else:
                hist_fused_feats.append(fused_feat)
            lidar_pose_list.append(lidar_pose[:, past_i])
            prev_confidence_maps = confidence_maps
            # if len(hist_fused_feats) >= hist_len:
            comm_rates.append(comm_rate)
                # print('comm_rate: ', past_i, comm_rate)
            confidence_maps_list.append(confidence_maps_single)
            request_maps_list.append(confidence_maps)
        comm_rates = sum(comm_rates)
        ##############################################################

        ################## Decode collaborated features ##############
        output_dict = {
            'flow_preds': ego_flow, # flow,
            'state_preds': ego_state_class_pred # state_class_pred
        }
        fuse_layer_id = 0
        if isinstance(fused_feat, list):
            prefixs = ['', '_onlyhist', '_withhist']
        else:
            prefixs = ['']
            fused_feat = [fused_feat]
        
        # print('diff: ', torch.abs(fused_feat[-1]-fused_feat[-2]).mean())
        for _i, _fused_feat in enumerate(fused_feat):
            feat_list_colla = []
            for layer_id, feat in enumerate(split_feat_list_single):
                if layer_id == fuse_layer_id:
                    feat_list_colla.append(self.get_ego_feat(_fused_feat, record_len))
                    # feat_list_colla.append(self.get_ego_feat(split_feat_list_single[layer_id][:,0], record_len))
                else:
                    feat_list_colla.append(self.get_ego_feat(split_feat_list_single[layer_id][:,0], record_len))
            feats_colla = self.decode(feat_list_colla)
            # print('fused_feature: ', fused_feature.shape)
            cls = self.cls_head(feats_colla)
            bbox = self.reg_head(feats_colla)
            _, bbox_temp = self.generate_predicted_boxes(cls, bbox)
            result_dict = {'cls_preds{}'.format(prefixs[_i]): cls,
                       'reg_preds{}'.format(prefixs[_i]): bbox_temp,
                       'bbox_preds{}'.format(prefixs[_i]): bbox
                       }
            output_dict.update(result_dict)
        ##############################################################

        DEBUG = self.debug
        if DEBUG:
            self.vis_CR_maps(confidence_maps_list, request_maps_list, save_name='vis_cr_maps')
            self.vis_count += 1

        _, bbox_temp_single = self.generate_predicted_boxes(psm_single[:,0], rm_single[:,0])

        output_dict.update({'cls_preds_single': psm_single[:,0],
                       'reg_preds_single': bbox_temp_single,
                       'bbox_preds_single': rm_single[:,0],
                       'comm_rate': comm_rates
                       })
        return output_dict

    def vis_fused_feats(self, hist_fused_feats, fused_feat, flow, cls, save_name='vis_feat'):
        use_mask = False
        mask = ((cls.sigmoid()[0][0] > 0.1)*1.0).cpu().numpy()
        save_dir = '/remote-home/share/yhu/Co_Flow/opencood/visualization/{}/{}.png'.format(save_name, self.vis_count)
        fig, axes = plt.subplots(len(hist_fused_feats[::-1][:2])+2, 1)
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
            save_dir = '/remote-home/share/yhu/Co_Flow/opencood/visualization/{}/{}_{}.png'.format(save_name, self.vis_count, t_i)
            num_agents = confidence_maps_list[0].shape[0]
            fig, axes = plt.subplots(num_agents, 3, figsize=(20, 9))
            for ax_i in range(num_agents):
                conf_single = (confidence_maps_list[hist_len+t_i].cpu().numpy()[ax_i,0] > 0.001)*1.0
                sns.heatmap(conf_single, ax=axes[ax_i][0], cbar=True)
                axes[ax_i][0].set_xticks([])
                axes[ax_i][0].set_yticks([])
                axes[ax_i][0].set_title('t_{}_single'.format(ax_i))

                conf_whist = (request_maps_list[hist_len+t_i].cpu().numpy()[ax_i,0] > 0.001)*1.0
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

    def forward_ss(self, data_dict):
        record_len = data_dict['record_len']
        pairwise_t_matrix = data_dict['pairwise_t_matrix']
        batch_dict = self.get_voxel_feat(data_dict)

        #################### Single Encode & Decode ##################
        feat_list_single = self.encode(batch_dict['spatial_features'])
        feats_single = self.decode(feat_list_single)   # [sum(cav_num)*num_sweep_frames, 256, 100, 352]
        psm_single = self.cls_head(feats_single)
        rm_single = self.reg_head(feats_single)

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

        curr_psm_single = self.get_ego_feat(psm_single[:,0],record_len)
        curr_rm_single = self.get_ego_feat(rm_single[:,0],record_len)
        _, bbox_temp_single = self.generate_predicted_boxes(curr_psm_single, curr_rm_single)

        output_dict = {'cls_preds': curr_psm_single,
                       'reg_preds': bbox_temp_single,
                       'bbox_preds': curr_rm_single
                       }
        return output_dict
