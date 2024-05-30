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
# from opencood.models.fuse_modules.where2comm import Where2comm
from opencood.models.sub_modules.torch_transformation_utils import warp_affine_simple
from opencood.models.fuse_modules.where2comm_attn import Where2comm
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


def measure_time(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        execution_time = end_time - start_time
        print(f"函数 {func.__name__} 的执行时间：{execution_time} 秒")
        return result

    return wrapper


class CenterPointWhere2commMultiSweepKalmanTrack(nn.Module):
    def __init__(self, args):
        super(CenterPointWhere2commMultiSweepKalmanTrack, self).__init__()

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
        # self.hist_len = args['hist_len']
        self.skip_scale = args['skip_scale']

        temporal_fusion_mode = args['fusion_args']['temporal_fusion'] if 'temporal_fusion' in args[
            'fusion_args'] else 'max'
        if temporal_fusion_mode == 'attn':
            self.hist_fusion_net = PointwiseFusion(args['fusion_args'])
        else:
            self.hist_fusion_net = SMaxFusion()

        self.process_frame_count = 0

        self.wo_colla = False
        if 'temporal_args' in args and 'wo_colla' in args['temporal_args']:
            self.wo_colla = args['temporal_args']['wo_colla']
        self.sampling_gap = 1
        if 'temporal_args' in args and 'sampling_gap' in args['temporal_args']:
            self.sampling_gap = args['temporal_args']['sampling_gap']
        self.temporal_thre = 0.001
        if 'temporal_args' in args and 'temporal_thre' in args['temporal_args']:
            self.temporal_thre = args['temporal_args']['temporal_thre']
        print('wo_colla: ', self.wo_colla)

        self.cls_head = nn.Conv2d(self.out_channel, args['anchor_number'],
                                  kernel_size=1)
        self.reg_head = nn.Conv2d(self.out_channel, 8 * args['anchor_number'],
                                  kernel_size=1)
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

        self.MOTtracker = AB3DMOT(
            covariance_id=0,
            use_angular_velocity=True,
            max_age=2
        )
        # self.matcher = Matcher(fusion='flow')

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

    # @measure_time
    # 0.4s each frame on 3090
    def updateKal(self, dets, dets_score, trks):
        dets = corner_to_center(dets)  # [xyz, l, w, h, theta]
        dets_score = dets_score.unsqueeze(1)
        trks = corner_to_center(trks)  # --> [xyz, l, w, h, theta]
        trks = trks[:, self.MOTtracker.reorder]  # -->[x, y, z, theta, l, w, h]

        # dets: [N, 7]
        dets = dets[:, self.MOTtracker.reorder]  # before [x, y, z, l, w, h, theta] after: [x, y, z, theta, l, w, h]
        info = dets_score
        ids = np.zeros((dets.shape[0], 1))  ## 生成一个记载 id 的矩阵

        dets_8corner = [convert_3dbox_to_8corner(det_tmp) for det_tmp in dets]
        if len(dets_8corner) > 0:
            dets_8corner = np.stack(dets_8corner, axis=0)
        else:
            dets_8corner = []

        trks_8corner = [convert_3dbox_to_8corner(trk_tmp) for trk_tmp in trks]
        if len(trks_8corner) > 0:
            trks_8corner = np.stack(trks_8corner, axis=0)

        # st = time.time()
        # slow
        matched, unmatched_dets, unmatched_trks = associate_detections_to_trackers(dets_8corner, trks_8corner,
                                                                                   iou_threshold=0.1,
                                                                                   print_debug=False,
                                                                                   match_algorithm='hungar',
                                                                                   doreid=False)
        # asso_time = time.time()
        # print('asso time: ', asso_time - st)

        # update matched trackers with assigned detections
        for t, trk in enumerate(self.MOTtracker.trackers):
            if t not in unmatched_trks:
                d = matched[np.where(matched[:, 1] == t)[0], 0]  # a list of index
                trk.update(dets[d, :][0], info[d, :][0], reid=None)
                detection_score = info[d, :][0][-1]
                trk.track_score = detection_score
                ids[d, 0] = trk.id
        # update_time = time.time()
        # print('update time: ', update_time - asso_time)

        # create and initialise new trackers for unmatched detections
        for i in unmatched_dets:  # a scalar of index
            detection_score = info[i, :]
            track_score = detection_score[0]
            trk = KalmanBoxTracker(dets[i, :], info[i, :], dets_8corner[i, :], self.MOTtracker.covariance_id,
                                   track_score,
                                   self.MOTtracker.tracking_name, self.MOTtracker.use_angular_velocity)
            self.MOTtracker.trackers.append(trk)
            ids[i, 0] = trk.id
        # create_time = time.time()
        # print('create time: ', create_time - update_time)

        i = len(self.MOTtracker.trackers)
        for trk in reversed(self.MOTtracker.trackers):

            d = trk.get_state()  # bbox location
            d = d[self.MOTtracker.reorder_back]

            # if ((trk.time_since_update < self.max_age) and (
            #         trk.hits >= self.min_hits or self.frame_count <= self.min_hits)):
            #     ret.append(np.concatenate((d, [trk.id + 1], trk.info[:-1], [trk.track_score])).reshape(1, -1))
            i -= 1
            # remove dead tracklet
            if trk.time_since_update >= self.MOTtracker.max_age:
                self.MOTtracker.trackers.pop(i)
        # remove_time = time.time()
        # print('remove time: ', remove_time - create_time)

    def resetKal(self):
        self.MOTtracker.trackers = []
        self.MOTtracker.frame_count = 0
        self.process_frame_count = 0

    def predictKal(self, predict_decay=0.7):
        trk_previous = np.zeros((len(self.MOTtracker.trackers), 7))  # N x 7 , #get previous locations from trackers.
        for t, trk in enumerate(trk_previous):
            pos = self.MOTtracker.trackers[t].get_state().reshape((-1, 1))
            trk_previous[t] = [pos[0], pos[1], pos[2], pos[3], pos[4], pos[5], pos[6]]
        trk_previous = trk_previous[:, self.MOTtracker.reorder_back]

        trks = np.zeros((len(self.MOTtracker.trackers), 7))  # N x 7 , #get predicted locations from existing trackers.
        score = np.zeros((len(self.MOTtracker.trackers), 1))
        to_del = []
        for t, trk in enumerate(trks):
            pos = self.MOTtracker.trackers[t].predict().reshape((-1, 1))
            trk[:] = [pos[0], pos[1], pos[2], pos[3], pos[4], pos[5], pos[6]]

            self.MOTtracker.trackers[t].track_score *= predict_decay

            score[t] = self.MOTtracker.trackers[t].track_score.cpu().numpy()
            if (pd.isnull(np.isnan(pos.all()))):  # ly
                to_del.append(t)
        trks = np.ma.compress_rows(np.ma.masked_invalid(trks))
        score = np.ma.compress_rows(np.ma.masked_invalid(score))
        for t in reversed(to_del):
            self.MOTtracker.trackers.pop(t)

        # trks: x,y,z,l,w,h,theta
        trks = trks[:, self.MOTtracker.reorder_back]

        trks = box_utils.boxes_to_corners_3d(trks, order='lwh')
        trk_previous = box_utils.boxes_to_corners_3d(trk_previous, order='lwh')

        return trks, score, trk_previous

    # @measure_time
    # about 0.3-0.8s except first frame
    def iterateKal(self, cur_feat_list, record_len, pairwise_t_matrix, trans_mats_dict, frame_id, colla_flag=False,
                   fuse_layer_id=0, predict_decay=0.7, prev_feat_box=None, box_multi=None):

        # 1. current ego / collaborative results
        feats = cur_feat_list[fuse_layer_id]
        # psm_single = self.cls_head(self.decode(cur_feat_list))
        # confidence_maps = psm_single.sigmoid().max(dim=1)[0].unsqueeze(
        #     1)  # dim1=2 represents the confidence of two anchors
        # request_maps = None
        # if colla_flag:
        #     detect_feat, comm_rate, fusion_result_dict = self.fusion(feats, confidence_maps, record_len,
        #                                                              pairwise_t_matrix, request_maps,
        #                                                              channel_compression=False, config_thre=True)
        # else:
        #     detect_feat, comm_rate, fusion_result_dict = self.fusion(feats, confidence_maps, record_len,
        #                                                              pairwise_t_matrix, request_maps,
        #                                                              channel_compression=False, config_thre=False)
        #
        # feat_list = self.get_colla_feat_list(cur_feat_list, detect_feat)
        # feat_list_colla = []
        # for layer_id, feat in enumerate(feat_list):
        #     feat_list_colla.append(self.get_ego_feat(feat, record_len))
        #
        # feats_colla = self.decode(feat_list_colla)
        # cls = self.cls_head(feats_colla)
        # bbox = self.reg_head(feats_colla)
        # _, bbox_temp = self.generate_predicted_boxes(cls, bbox)
        #
        # box_colla, box_colla_score = self.feature2box(cls, bbox_temp,
        #                                               trans_mats_dict['trans_mats_track'])  # shape [N, 8, 3]

        box_colla, box_colla_score, comm_rate = box_multi['box_colla'], box_multi['box_colla_score'], box_multi['comm_rate']

        # 2. kalman filter prediction result
        box_preds, box_preds_score, box_previous = self.predictKal()  # shape [N, 8, 3]

        box_preds = torch.from_numpy(box_preds).to(feats.device)
        box_previous = torch.from_numpy(box_previous).to(feats.device)
        box_preds_score = torch.from_numpy(box_preds_score).to(feats.device).squeeze()

        box_preds = box_utils.project_box3d(box_preds, torch.inverse(trans_mats_dict['ego2world_list'][frame_id]))
        box_previous = box_utils.project_box3d(box_previous, torch.inverse(trans_mats_dict['ego2world_list'][frame_id]))

        flow = None
        if colla_flag is False:
            # import pdb
            # pdb.set_trace()
            flow = self.generate_flow(box_previous, box_preds)
            flow = torch.from_numpy(flow).to(feats.device)[None, ...]  # shape [1, 2, H, W]

        box_preds_save = box_preds.clone()

        # 3. fusion of kalman prediction with ego/collaborative results

        if box_preds_score.dim() == 0:
            box_preds_score = box_preds_score.unsqueeze(0)
        box_update, box_update_score = self.late_fusion(box_preds, box_colla, box_preds_score, box_colla_score)

        box_update_save = box_update.clone()  # shape = [N, 8, 3]

        # 4. kalman filter update
        box_update = box_utils.project_box3d(box_update, trans_mats_dict['ego2world_list'][frame_id]).cpu().numpy()
        box_preds = box_utils.project_box3d(box_preds, trans_mats_dict['ego2world_list'][frame_id]).cpu().numpy()
        self.updateKal(box_update, box_update_score, box_preds)

        debug_dict = {
            'box_kal': box_preds_save,  # kalman prediction
            'box_kal_score': box_preds_score,
            'box_colla': box_colla,  # current frame detection
            'box_colla_score': box_colla_score,
            'box_update': box_update_save,  # fused result
            'box_update_score': box_update_score,
            # 'colla_feat': detect_feat.clone(),
        }

        # 5. generate feature flow
        fused_feat_box, fused_feat_box_score = None, None
        if colla_flag is False and prev_feat_box is not None:
            # import pdb; pdb.set_trace()
            # box_flow, box_mask = self.generate_box_flow(box_update_save, box_update_score, prev_feat_box['prev_box'],
            #                                             prev_feat_box['prev_box_score'],
            #                                             trans_mats_dict['current2prev_list'][frame_id],
            #                                             # trans_mats_dict['prev2current_list'][frame_id],
            #                                             )
            lidar_pose = torch.cat([trans_mats_dict['lidar_pose'][0][frame_id][None, ...],
                                    trans_mats_dict['lidar_pose'][0][frame_id + 1][None, ...]], dim=0)
            # import pdb; pdb.set_trace()
            # vis_flow_sns(flow, path='/remote-home/share/xhpang/OpenCOODv2/opencood/logs/kalman/Predict ')
            flow = prev_feat_box['flow_gt'].float()
            prev_feat = self.warp_feat(prev_feat_box['prev_feat'][0][None, ...], lidar_pose)
            x_coord = torch.arange(feats.shape[-1]).float()
            y_coord = torch.arange(feats.shape[-2]).float()
            y, x = torch.meshgrid(y_coord, x_coord)
            ori_grid = torch.cat([x.unsqueeze(0), y.unsqueeze(0)], dim=0).unsqueeze(0).expand(flow.shape[0], -1, -1,
                                                                                              -1).to(flow.device)
            updated_grid = ori_grid - flow
            updated_grid[:, 0, :, :] = updated_grid[:, 0, :, :] / (feats.shape[-1] / 2.0) - 1.0
            updated_grid[:, 1, :, :] = updated_grid[:, 1, :, :] / (feats.shape[-2] / 2.0) - 1.0
            updated_grid = updated_grid.permute(0, 2, 3, 1)
            out = F.grid_sample(prev_feat, grid=updated_grid, mode='bilinear')
            # out = self.warp_feat(out, lidar_pose)

            # warped_feat = self.warp_w_box_flow(prev_feat_box['prev_feat'][0].unsqueeze(0), box_flow, box_mask)
            # fused_feat = self.hist_fusion_net(warped_feat, feats[0].unsqueeze(0))
            # fused_feat = warped_feat
            debug_dict.update({'colla_feat': out.clone()})
            fused_feat_list = self.get_colla_feat_list(cur_feat_list, out)
            fused_feat_list_colla = []
            for layer_id, feat in enumerate(fused_feat_list):
                fused_feat_list_colla.append(self.get_ego_feat(feat, record_len))
            fused_feats = self.decode(fused_feat_list_colla)
            cls = self.cls_head(fused_feats)
            bbox = self.reg_head(fused_feats)
            _, bbox_temp = self.generate_predicted_boxes(cls, bbox)
            fused_feat_box, fused_feat_box_score = self.feature2box(cls, bbox_temp, trans_mats_dict['trans_mats_track'])

            debug_dict.update({'fused_feat_box': fused_feat_box.clone(),
                               'fused_feat_box_score': fused_feat_box_score.clone()})

        output_box_dict = {
                           'comm_rate': comm_rate,
                           'fused_box': box_update_save,
                           'fused_box_score': box_update_score,
                           'fused_feat_box': fused_feat_box,
                           'fused_feat_box_score': fused_feat_box_score
                           }
        return output_box_dict, debug_dict

    def warp_feat(self, feat, pose):
        _, _, H, W = feat.shape
        downsample_rate = 2
        discrete_ratio = 0.4
        frame_record_len = torch.tensor([2]).to(feat.device)
        pairwise_t_matrix = get_pairwise_transformation_torch(pose, 2, frame_record_len, 6)

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

    def generate_box_flow(self, current_box, current_score, previous_box, previous_score, curr2prev_mat):

        # 1. transform current box to previous coordinate
        current_box = box_utils.project_box3d(current_box, curr2prev_mat)
        # previous_box = box_utils.project_box3d(previous_box, curr2prev_mat)
        current_box_center = corner_to_center(current_box.cpu().numpy())
        prev_box_center = corner_to_center(previous_box.cpu().numpy())
        ego_box_curr_results = {'pred_box_3dcorner_tensor': current_box,
                                'pred_box_center_tensor': torch.from_numpy(current_box_center).to(current_score.device),
                                'scores': current_score}
        ego_box_prev_results = {'pred_box_3dcorner_tensor': previous_box,
                                'pred_box_center_tensor': torch.from_numpy(prev_box_center).to(current_score.device),
                                'scores': previous_score}
        cav_idx = 1
        past_k_time_diff = [-1, -2]

        cav_content = OrderedDict()
        cav_content.update({'past_k_time_diff': past_k_time_diff})
        cav_content[0] = OrderedDict()
        cav_content[0].update(ego_box_curr_results)
        cav_content[1] = OrderedDict()
        cav_content[1].update(ego_box_prev_results)

        cav_dict = OrderedDict()
        cav_dict.update({cav_idx: cav_content})
        # import pdb; pdb.set_trace()

        box_flow_map, mask = self.matcher(cav_dict, shape_list=torch.tensor([64, 96, 352]).to(current_score.device))
        return box_flow_map, mask

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
    def late_fusion(box_preds, box_colla, box_preds_score, box_colla_score):
        # import pdb; pdb.set_trace()
        if box_preds is None:
            pred_box3d_tensor = box_colla
            scores = box_colla_score
        elif box_colla is None:
            pred_box3d_tensor = box_preds
            scores = box_preds_score
        else:
            pred_box3d_tensor = torch.cat([box_preds, box_colla], dim=0)
            scores = torch.cat([box_preds_score, box_colla_score], dim=0)

        # filter out the prediction with low score
        score_thre = 0.2
        mask = torch.gt(scores, score_thre)
        mask_reg = mask.unsqueeze(1).unsqueeze(2).repeat(1, 8, 3)

        pred_box3d_tensor = torch.masked_select(pred_box3d_tensor, mask_reg).view(-1, 8, 3)
        scores = torch.masked_select(scores, mask)

        keep_index = box_utils.nms_rotated(pred_box3d_tensor,
                                           scores,
                                           0.15
                                           )

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

    # @measure_time
    def forward(self, data_dict, dataset=None):

        if self.process_frame_count % self.skip_scale == 0:
            colla_flag = True
        else:
            colla_flag = False

        if self.process_frame_count <= 5:
            colla_flag = True

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
        ################################################ for multi scale fusion test
        psm_single = psm_single.view(-1, psm_single.shape[-3], psm_single.shape[-2],
                                     psm_single.shape[-1])

        # import pdb
        # pdb.set_trace()
        fused_feature, communication_rates, result_dict = self.fusion_net(batch_dict['spatial_features'],
                                                                          psm_single,
                                                                          record_len,
                                                                          pairwise_t_matrix,
                                                                          self.backbone,
                                                                          config_thre=colla_flag)
        # print('communication_rates', communication_rates)
        # import pdb
        # pdb.set_trace()
        fused_feature = self.shrink_conv(fused_feature)
        cls = self.cls_head(fused_feature)
        bbox = self.reg_head(fused_feature)
        _, bbox_temp = self.generate_predicted_boxes(cls, bbox)
        box_colla, box_colla_score = self.feature2box(cls, bbox_temp, trans_mats_track)
        box_multi_dict = {'box_colla': box_colla,
                          'box_colla_score': box_colla_score,
                          'comm_rate': communication_rates}
        ################################################

        ################### Iteration each timestamp ################
        comm_rates = []
        past_i = 0
        cur_feat_list = [x[:, past_i] for x in split_feat_list_single]
        psm_single = psm_single.view(-1, psm_single.shape[-3], psm_single.shape[-2],
                                     psm_single.shape[-1])
        cur_pairwise_t_matrix = get_pairwise_transformation_torch(lidar_pose[:, past_i], 5, record_len, 6)
        fused_box_dict, debug_dict = self.iterateKal(cur_feat_list, record_len, cur_pairwise_t_matrix,
                                                     trans_mats_dict, past_i,
                                                     # prev_feat_box=prev_feat_box_dict,
                                                     prev_feat_box=None,
                                                     colla_flag=colla_flag,
                                                     box_multi=box_multi_dict
                                                     )

        output_dict = {
            'comm_rate': fused_box_dict['comm_rate'],
            'box_preds': fused_box_dict['fused_box'],
            'box_preds_score': fused_box_dict['fused_box_score'],
            # 'box_preds': fused_box_dict['fused_feat_box'],
            # 'box_preds_score': fused_box_dict['fused_feat_box_score'].squeeze(),
            # 'box_preds': debug_dict['box_update'],
            # 'box_preds_score': debug_dict['box_update'].squeeze(),
            # 'box_preds': debug_dict['box_colla'],
            # 'box_preds_score': debug_dict['box_colla_score'],
            # 'box_preds': box_colla,
            # 'box_preds_score': box_colla_score,
        }

        ori_output_dict = {'cls_preds': cls,
                       'reg_preds': bbox_temp,
                       'bbox_preds': bbox
                       }
        output_dict.update(ori_output_dict)
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

        batch, H, W, code_size = box_preds.size()  ## code_size 表示的是预测的尺寸

        box_preds = box_preds.reshape(batch, H * W, code_size)

        batch_reg = box_preds[..., 0:2]
        # batch_hei = box_preds[..., 2:3]
        # batch_dim = torch.exp(box_preds[..., 3:6])

        h = box_preds[..., 3:4] * self.out_size_factor * self.voxel_size[0]
        w = box_preds[..., 4:5] * self.out_size_factor * self.voxel_size[1]
        l = box_preds[..., 5:6] * self.out_size_factor * self.voxel_size[2]
        batch_dim = torch.cat([h, w, l], dim=-1)
        batch_hei = box_preds[..., 2:3] * self.out_size_factor * self.voxel_size[2] + self.cav_lidar_range[2]

        batch_rots = box_preds[..., 6:7]
        batch_rotc = box_preds[..., 7:8]

        rot = torch.atan2(batch_rots, batch_rotc)

        ys, xs = torch.meshgrid([torch.arange(0, H), torch.arange(0, W)])
        ys = ys.view(1, H, W).repeat(batch, 1, 1).to(cls_preds.device)
        xs = xs.view(1, H, W).repeat(batch, 1, 1).to(cls_preds.device)

        xs = xs.view(batch, -1, 1) + batch_reg[:, :, 0:1]
        ys = ys.view(batch, -1, 1) + batch_reg[:, :, 1:2]

        xs = xs * self.out_size_factor * self.voxel_size[0] + self.cav_lidar_range[0]  ## 基于feature_map 的size求解真实的坐标
        ys = ys * self.out_size_factor * self.voxel_size[1] + self.cav_lidar_range[1]

        batch_box_preds = torch.cat([xs, ys, batch_hei, batch_dim, rot], dim=2)
        # batch_box_preds = batch_box_preds.reshape(batch, H, W, batch_box_preds.shape[-1])
        # batch_box_preds = batch_box_preds.permute(0, 3, 1, 2).contiguous()

        # batch_box_preds_temp = torch.cat([xs, ys, batch_hei, batch_dim, rot], dim=1)
        # box_preds = box_preds.permute(0, 3, 1, 2).contiguous()

        # batch_cls_preds = cls_preds.view(batch, H*W, -1)
        return cls_preds, batch_box_preds

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
        _, bbox_temp_single = self.generate_predicted_boxes(curr_psm_single, curr_rm_single)

        output_dict = {'cls_preds': curr_psm_single,
                       'reg_preds': bbox_temp_single,
                       'bbox_preds': curr_rm_single
                       }
        return output_dict