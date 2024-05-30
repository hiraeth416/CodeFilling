# -*- coding: utf-8 -*-
# Author: Runsheng Xu <rxx3386@ucla.edu>, Hao Xiang <haxiang@g.ucla.edu>,
# License: TDG-Attribution-NonCommercial-NoDistrib

import cv2
import argparse
import os
import time
from typing import OrderedDict
import importlib
import torch
import open3d as o3d
from torch.utils.data import DataLoader, Subset
import numpy as np
import opencood.hypes_yaml.yaml_utils as yaml_utils
from opencood.tools import train_utils, inference_utils
from opencood.data_utils.datasets import build_dataset
from opencood.utils import eval_utils
from opencood.visualization import vis_utils, my_vis, simple_vis

torch.multiprocessing.set_sharing_strategy('file_system')
from opencood.data_utils.datasets.basedataset.uav_dataset.processing import UAVProcessing
from progress.bar import Bar


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        if self.count > 0:
            self.avg = self.sum / self.count


class PrefetchDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, pre_process_func):
        self.samples = dataset.samples
        self.img_dir = dataset.img_dir
        self.pre_process_func = pre_process_func
        self.map_scale = 1.0
        self.real = False
        self.noise = 1.0
        self.trans_layer = [0]
        self.message_mode = 'QualityMap'
        self.test_scales = '1'
        self.test_scales = [float(i) for i in self.test_scales.split(',')]

    # def load_image_func(self, index):
    #     sample_id = index // 25
    #     cam_id = index % 25
    #     images_key = 'image'
    #     images = []
    #     trans_mat_list = []
    #     image_idx = []
    #     cams_list = [x for x in self.samples[sample_id].keys() if not x.startswith('vehicles')]
    #     cam_list = random.sample([x for x in cams_list if not x.startswith(cams_list[cam_id])], self.opt.num_agents-1) + [cams_list[cam_id]]
    #     for cam, info in self.samples[sample_id].items():
    #         if cam.startswith('vehicles'):
    #             continue
    #         # else:
    #         # elif cam.startswith('F'):
    #         # elif cam.endswith(str(cam_id)):
    #         if cam in cam_list:
    #             images.append(cv2.imread(os.path.join(self.img_dir, info[images_key])))
    #             image_idx.append(info['image_id'])
    #             trans_mat_list.append(np.array(info['trans_mat'], dtype=np.float32))
    #     trans_mats = np.concatenate([x[None,:,:] for x in trans_mat_list], axis=0)
    #     return images, image_idx, trans_mats

    def load_image_func(self, index):
        # sample_id = 54
        # cam_id = 0
        sample_id = index // 5
        cam_id = index % 5
        img_dir = self.img_dir[sample_id] if len(self.img_dir) == len(self.samples) else self.img_dir
        images_key = 'image'
        images = []
        trans_mat_list = []
        trans_mats_n010_list = []
        trans_mats_n005_list = []
        trans_mats_p005_list = []
        trans_mats_p007_list = []
        trans_mats_p010_list = []
        trans_mats_p015_list = []
        trans_mats_p020_list = []
        trans_mats_p080_list = []
        shift_mats_1_list = []
        shift_mats_2_list = []
        shift_mats_4_list = []
        shift_mats_8_list = []
        trans_mats_list_withnoise = []
        shift_mats_list_withnoise = []
        image_idx = []
        cams_list = [x for x in self.samples[sample_id].keys() if not x.startswith('vehicles')]
        # cam_list = random.sample([x for x in cams_list if not x.startswith(cams_list[cam_id])], self.opt.num_agents-1) + [cams_list[cam_id]]
        # sensor = cams_list[cam_id].split('_')[1]
        sensors = ['FRONT', 'BOTTOM', 'LEFT', 'RIGHT', "BACK"]
        sensor = sensors[cam_id]
        cam_list = [x for x in cams_list if sensor in x]
        # print(cam_list)

        for cam, info in self.samples[sample_id].items():
            if cam.startswith('vehicles'):
                continue
            # else:
            # elif cam.startswith('F'):
            # elif cam.endswith(str(cam_id)):
            if cam in cam_list:
                image = cv2.imread(os.path.join(img_dir, info[images_key]))
                if self.real:
                    image = cv2.resize(image, (720, 480))
                images.append(image)
                image_idx.append(info['image_id'])
                trans_mat_list.append(np.array(info['trans_mat'], dtype=np.float32))
                trans_mats_n010_list.append(np.array(info['trans_mat_n010'], dtype=np.float32))
                trans_mats_n005_list.append(np.array(info['trans_mat_n005'], dtype=np.float32))
                trans_mats_p005_list.append(np.array(info['trans_mat_p005'], dtype=np.float32))
                trans_mats_p007_list.append(np.array(info['trans_mat_p007'], dtype=np.float32))
                trans_mats_p010_list.append(np.array(info['trans_mat_p010'], dtype=np.float32))
                trans_mats_p015_list.append(np.array(info['trans_mat_p015'], dtype=np.float32))
                trans_mats_p020_list.append(np.array(info['trans_mat_p020'], dtype=np.float32))
                trans_mats_p080_list.append(np.array(info['trans_mat_p080'], dtype=np.float32))
                shift_mats_1_list.append(np.array(info['shift_mats'][1 * self.map_scale], dtype=np.float32))
                shift_mats_2_list.append(np.array(info['shift_mats'][2 * self.map_scale], dtype=np.float32))
                shift_mats_4_list.append(np.array(info['shift_mats'][4 * self.map_scale], dtype=np.float32))
                shift_mats_8_list.append(np.array(info['shift_mats'][8 * self.map_scale], dtype=np.float32))
                if self.noise > 0:
                    trans_mats_list_withnoise.append(
                        np.array(info['trans_mat_withnoise{:01d}'.format(int(self.noise * 10))], dtype=np.float32))
                    shift_mats_list_withnoise.append(
                        np.array(info['shift_mats_withnoise'][self.noise][2 ** self.trans_layer[-1]],
                                 dtype=np.float32))
                else:
                    trans_mats_list_withnoise.append(np.array(info['trans_mat'], dtype=np.float32))
                    shift_mats_list_withnoise.append(
                        np.array(info['shift_mats'][1 * self.map_scale], dtype=np.float32))
        trans_mats = np.concatenate([x[None, :, :] for x in trans_mat_list], axis=0)
        trans_mats_n010 = np.concatenate([x[None, :, :] for x in trans_mats_n010_list], axis=0)
        trans_mats_n005 = np.concatenate([x[None, :, :] for x in trans_mats_n005_list], axis=0)
        trans_mats_p005 = np.concatenate([x[None, :, :] for x in trans_mats_p005_list], axis=0)
        trans_mats_p007 = np.concatenate([x[None, :, :] for x in trans_mats_p007_list], axis=0)
        trans_mats_p010 = np.concatenate([x[None, :, :] for x in trans_mats_p010_list], axis=0)
        trans_mats_p015 = np.concatenate([x[None, :, :] for x in trans_mats_p015_list], axis=0)
        trans_mats_p020 = np.concatenate([x[None, :, :] for x in trans_mats_p020_list], axis=0)
        trans_mats_p080 = np.concatenate([x[None, :, :] for x in trans_mats_p080_list], axis=0)
        shift_mats_1 = np.concatenate([x[None, :, :] for x in shift_mats_1_list], axis=0)
        shift_mats_2 = np.concatenate([x[None, :, :] for x in shift_mats_2_list], axis=0)
        shift_mats_4 = np.concatenate([x[None, :, :] for x in shift_mats_4_list], axis=0)
        shift_mats_8 = np.concatenate([x[None, :, :] for x in shift_mats_8_list], axis=0)

        trans_mats_withnoise = np.concatenate([x[None, :, :] for x in trans_mats_list_withnoise], axis=0)
        shift_mats_withnoise = np.concatenate([x[None, :, :] for x in shift_mats_list_withnoise], axis=0)

        # if self.opt.noise > 0:
        #     trans_mats = trans_mats_withnoise
        #     if self.opt.trans_layer in [0,-2]:
        #         shift_mats_1 = shift_mats_withnoise
        #     elif self.opt.trans_layer == 1:
        #         shift_mats_2 = shift_mats_withnoise
        #     elif self.opt.trans_layer == 2:
        #         shift_mats_4 = shift_mats_withnoise
        #     elif self.opt.trans_layer == 3:
        #         shift_mats_8 = shift_mats_withnoise

        return images, image_idx, [trans_mats, trans_mats_n010, trans_mats_n005, trans_mats_p005, trans_mats_p007,
                                   trans_mats_p010, trans_mats_p015, trans_mats_p020, trans_mats_p080,
                                   trans_mats_withnoise], \
               [shift_mats_1, shift_mats_2, shift_mats_4, shift_mats_8, shift_mats_withnoise]

    def load_sample_func(self, index):
        info = self.samples[index]
        img_dir = self.img_dir[index] if len(self.img_dir) == len(self.samples) else self.img_dir
        images_key = 'image'
        images = []
        image_idx = []
        image = cv2.imread(os.path.join(img_dir, info[images_key]))
        if self.real:
            image = cv2.resize(image, (720, 480))
        images.append(image)
        image_idx.append(info['image_id'])
        trans_mats = np.array(info['trans_mat'], dtype=np.float32)[None, :, :]
        trans_mats_n010 = np.array(info['trans_mat_n010'], dtype=np.float32)[None, :, :]
        trans_mats_n005 = np.array(info['trans_mat_n005'], dtype=np.float32)[None, :, :]
        trans_mats_p005 = np.array(info['trans_mat_p005'], dtype=np.float32)[None, :, :]
        trans_mats_p007 = np.array(info['trans_mat_p007'], dtype=np.float32)[None, :, :]
        trans_mats_p010 = np.array(info['trans_mat_p010'], dtype=np.float32)[None, :, :]
        trans_mats_p015 = np.array(info['trans_mat_p015'], dtype=np.float32)[None, :, :]
        trans_mats_p020 = np.array(info['trans_mat_p020'], dtype=np.float32)[None, :, :]
        trans_mats_p080 = np.array(info['trans_mat_p080'], dtype=np.float32)[None, :, :]
        shift_mats_1 = np.array(info['shift_mats'][1 * self.map_scale], dtype=np.float32)[None, :, :]
        shift_mats_2 = np.array(info['shift_mats'][2 * self.map_scale], dtype=np.float32)[None, :, :]
        shift_mats_4 = np.array(info['shift_mats'][4 * self.map_scale], dtype=np.float32)[None, :, :]
        shift_mats_8 = np.array(info['shift_mats'][8 * self.map_scale], dtype=np.float32)[None, :, :]
        return images, image_idx, [trans_mats, trans_mats_n010, trans_mats_n005, trans_mats_p005, trans_mats_p007,
                                   trans_mats_p010, trans_mats_p015, trans_mats_p020, trans_mats_p080], \
               [shift_mats_1, shift_mats_2, shift_mats_4, shift_mats_8]

    def __getitem__(self, index):
        if 'NO_MESSAGE' in self.message_mode:
            images, image_idx, trans_mats, shift_mats = self.load_sample_func(index)
        else:
            images, image_idx, trans_mats, shift_mats = self.load_image_func(index)

        scaled_images, meta = {}, {}
        for scale in self.test_scales:
            cur_images = []
            for image in images:
                cur_image, cur_meta = self.pre_process_func(image, scale)
                cur_images.append(cur_image)
            scaled_images[scale] = np.concatenate(cur_images, axis=0)
            meta[scale] = cur_meta
        return image_idx, {'images': scaled_images, 'image': images, 'meta': meta, \
                           'trans_mats': trans_mats[0], 'trans_mats_n010': trans_mats[1],
                           'trans_mats_n005': trans_mats[2], 'trans_mats_p005': trans_mats[3], \
                           'trans_mats_p007': trans_mats[4], 'trans_mats_p010': trans_mats[5],
                           'trans_mats_p015': trans_mats[6], 'trans_mats_p020': trans_mats[7], \
                           'trans_mats_p080': trans_mats[8], \
                           'shift_mats_1': shift_mats[0], 'shift_mats_2': shift_mats[1], 'shift_mats_4': shift_mats[2],
                           'shift_mats_8': shift_mats[3],
                           'trans_mats_withnoise': trans_mats[-1], 'shift_mats_withnoise': shift_mats[-1]}

    def __len__(self):
        if 'NO_MESSAGE' in self.message_mode:
            return len(self.samples)
        else:
            # return len(self.samples)*5
            return len(self.samples)


def test_parser():
    parser = argparse.ArgumentParser(description="synthetic data generation")
    parser.add_argument('--model_dir', type=str, required=True,
                        help='Continued training path')
    parser.add_argument('--fusion_method', type=str,
                        default='intermediate',
                        help='no, no_w_uncertainty, late, early or intermediate')
    parser.add_argument('--save_vis_interval', type=int, default=40,
                        help='interval of saving visualization')
    parser.add_argument('--save_npy', action='store_true',
                        help='whether to save prediction and gt result'
                             'in npy file')
    parser.add_argument('--range', type=str, default="140.8,40",
                        help="detection range is [-140.8,+140.8m, -40m, +40m]")
    parser.add_argument('--modal', type=int, default=0,
                        help='used in heterogeneous setting, 0 lidaronly, 1 camonly, 2 ego_lidar_other_cam, 3 ego_cam_other_lidar')
    parser.add_argument('--no_score', action='store_true',
                        help="whether print the score of prediction")
    parser.add_argument('--note', default="", type=str, help="any other thing?")
    opt = parser.parse_args()
    return opt


def main():
    opt = test_parser()

    assert opt.fusion_method in ['late', 'early', 'intermediate', 'no', 'no_w_uncertainty', 'single']

    hypes = yaml_utils.load_yaml(None, opt)

    if 'heter' in hypes:
        if opt.modal == 0:
            hypes['heter']['lidar_ratio'] = 1
            hypes['heter']['ego_modality'] = 'lidar'
            opt.note += '_lidaronly'

        if opt.modal == 1:
            hypes['heter']['lidar_ratio'] = 0
            hypes['heter']['ego_modality'] = 'camera'
            opt.note += '_camonly'

        if opt.modal == 2:
            hypes['heter']['lidar_ratio'] = 0
            hypes['heter']['ego_modality'] = 'lidar'
            opt.note += 'ego_lidar_other_cam'

        if opt.modal == 3:
            hypes['heter']['lidar_ratio'] = 1
            hypes['heter']['ego_modality'] = 'camera'
            opt.note += '_ego_cam_other_lidar'

        x_min, x_max = -140.8, 140.8
        y_min, y_max = -40, 40
        opt.note += f"_{x_max}_{y_max}"
        hypes['fusion']['args']['grid_conf']['xbound'] = [x_min, x_max,
                                                          hypes['fusion']['args']['grid_conf']['xbound'][2]]
        hypes['fusion']['args']['grid_conf']['ybound'] = [y_min, y_max,
                                                          hypes['fusion']['args']['grid_conf']['ybound'][2]]
        hypes['model']['args']['grid_conf'] = hypes['fusion']['args']['grid_conf']

        new_cav_range = [x_min, y_min, hypes['postprocess']['anchor_args']['cav_lidar_range'][2], \
                         x_max, y_max, hypes['postprocess']['anchor_args']['cav_lidar_range'][5]]

        hypes['preprocess']['cav_lidar_range'] = new_cav_range
        hypes['postprocess']['anchor_args']['cav_lidar_range'] = new_cav_range
        hypes['postprocess']['gt_range'] = new_cav_range
        hypes['model']['args']['lidar_args']['lidar_range'] = new_cav_range
        if 'camera_mask_args' in hypes['model']['args']:
            hypes['model']['args']['camera_mask_args']['cav_lidar_range'] = new_cav_range

        # reload anchor
        yaml_utils_lib = importlib.import_module("opencood.hypes_yaml.yaml_utils")
        for name, func in yaml_utils_lib.__dict__.items():
            if name == hypes["yaml_parser"]:
                parser_func = func
        hypes = parser_func(hypes)

    hypes['validate_dir'] = hypes['test_dir']
    if "OPV2V" in hypes['test_dir'] or "v2xsim" in hypes['test_dir']:
        assert "test" in hypes['validate_dir']

    # This is used in visualization
    # left hand: OPV2V, V2XSet
    # right hand: V2X-Sim 2.0 and DAIR-V2X
    left_hand = True if ("OPV2V" in hypes['test_dir'] or "V2XSET" in hypes['test_dir']) else False

    print(f"Left hand visualizing: {left_hand}")

    if 'box_align' in hypes.keys():
        hypes['box_align']['val_result'] = hypes['box_align']['test_result']

    print('Creating Model')
    model = train_utils.create_model(hypes)
    # we assume gpu is necessary
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print('Loading Model from checkpoint')
    saved_path = opt.model_dir
    resume_epoch, model = train_utils.load_saved_model(saved_path, model)
    print(f"resume from {resume_epoch} epoch.")
    opt.note += f"_epoch{resume_epoch}"

    if torch.cuda.is_available():
        model.cuda()
    model.eval()

    # setting noise
    np.random.seed(303)

    # build dataset for each noise setting
    print('Dataset Building')
    uavDetector = UAVProcessing(hypes['Processing'], model)
    opencood_dataset = build_dataset(hypes, visualize=True, train=False)
    # opencood_dataset_subset = Subset(opencood_dataset, range(640,2100))
    # data_loader = DataLoader(opencood_dataset_subset,
    data_loader = DataLoader(PrefetchDataset(opencood_dataset, uavDetector.pre_process),
                             batch_size=1,
                             num_workers=4,
                             # collate_fn=opencood_dataset.collate_batch_test,
                             shuffle=False,
                             pin_memory=False,
                             drop_last=False)

    # Create the dictionary for evaluation
    result_stat = {0.3: {'tp': [], 'fp': [], 'gt': 0, 'score': []},
                   0.5: {'tp': [], 'fp': [], 'gt': 0, 'score': []},
                   0.7: {'tp': [], 'fp': [], 'gt': 0, 'score': []}}

    infer_info = opt.fusion_method + opt.note


    num_iters = len(opencood_dataset)
    results = {}

    bar = Bar('{}'.format('default'), max=num_iters)
    time_stats = ['tot', 'load', 'pre', 'net', 'dec', 'post', 'merge', 'comm_rate', 'quant_error', 'code_len']
    avg_time_stats = {t: AverageMeter() for t in time_stats}

    for ind, (img_idx, pre_processed_images) in enumerate(data_loader):

        # if ind == 0 or ind == 1:
        #     continue
        ret = uavDetector.run(pre_processed_images)
        rets = ret['results']
        for i in range(len(rets)):
            img_id = img_idx[i]
            results[img_id.numpy().astype(np.int32)[0]] = rets[i]

        Bar.suffix = '[{0}/{1}]|Tot: {total:} |ETA: {eta:} '.format(
            ind, num_iters, total=bar.elapsed_td, eta=bar.eta_td)
        for t in avg_time_stats:
            avg_time_stats[t].update(ret[t])
            Bar.suffix = Bar.suffix + '|{} {tm.val:.4f}s ({tm.avg:.4f}s) '.format(
                t, tm=avg_time_stats[t])
        bar.next()

    bar.finish()
    print('#################### comm_rate: {:.6f}s) #################### '.format(avg_time_stats['comm_rate'].avg))
    print('#################### quant_error: {:.6f}s) #################### '.format(avg_time_stats['quant_error'].avg))
    print('#################### code_len: {:.6f}bytes) #################### '.format(avg_time_stats['code_len'].avg))

    print('######################################################')
    print('################# BEV Object Detection ###############')
    print('######################################################')
    save_dir = '/GPFS/data/xhpang/OpenCOODv2/opencood/logs/uav_centerpoint_where2comm_2023_02_28_17_45_59/'
    if True:
        opencood_dataset.run_polygon_eval(results, save_dir, 'Global')
    else:
        dataset.run_eval(results, opt.save_dir, 'Global')

    # for i, batch_data in enumerate(data_loader):
    #     print(f"{infer_info}_{i}")
    #     if batch_data is None:
    #         continue
    #     with torch.no_grad():
    #         batch_data = train_utils.to_device(batch_data, device)
    #
    #         if opt.fusion_method == 'late':
    #             infer_result = inference_utils.inference_late_fusion(batch_data,
    #                                                     model,
    #                                                     opencood_dataset)
    #         elif opt.fusion_method == 'early':
    #             infer_result = inference_utils.inference_early_fusion(batch_data,
    #                                                     model,
    #                                                     opencood_dataset)
    #         elif opt.fusion_method == 'intermediate':
    #             infer_result = inference_utils.inference_intermediate_fusion(batch_data,
    #                                                             model,
    #                                                             opencood_dataset)
    #         elif opt.fusion_method == 'no':
    #             infer_result = inference_utils.inference_no_fusion(batch_data,
    #                                                             model,
    #                                                             opencood_dataset)
    #         elif opt.fusion_method == 'no_w_uncertainty':
    #             infer_result = inference_utils.inference_no_fusion_w_uncertainty(batch_data,
    #                                                             model,
    #                                                             opencood_dataset)
    #         elif opt.fusion_method == 'single':
    #             infer_result = inference_utils.inference_no_fusion(batch_data,
    #                                                             model,
    #                                                             opencood_dataset,
    #                                                             single_gt=True)
    #         else:
    #             raise NotImplementedError('Only single, no, no_w_uncertainty, early, late and intermediate'
    #                                     'fusion is supported.')
    #
    #         pred_box_tensor = infer_result['pred_box_tensor']
    #         gt_box_tensor = infer_result['gt_box_tensor']
    #         pred_score = infer_result['pred_score']
    #
    #         eval_utils.caluclate_tp_fp(pred_box_tensor,
    #                                 pred_score,
    #                                 gt_box_tensor,
    #                                 result_stat,
    #                                 0.3)
    #         eval_utils.caluclate_tp_fp(pred_box_tensor,
    #                                 pred_score,
    #                                 gt_box_tensor,
    #                                 result_stat,
    #                                 0.5)
    #         eval_utils.caluclate_tp_fp(pred_box_tensor,
    #                                 pred_score,
    #                                 gt_box_tensor,
    #                                 result_stat,
    #                                 0.7)
    #         if opt.save_npy:
    #             npy_save_path = os.path.join(opt.model_dir, 'npy')
    #             if not os.path.exists(npy_save_path):
    #                 os.makedirs(npy_save_path)
    #             inference_utils.save_prediction_gt(pred_box_tensor,
    #                                             gt_box_tensor,
    #                                             batch_data['ego'][
    #                                                 'origin_lidar'][0],
    #                                             i,
    #                                             npy_save_path)
    #
    #         if not opt.no_score:
    #             infer_result.update({'score_tensor': pred_score})
    #
    #         if getattr(opencood_dataset, "heterogeneous", False):
    #             cav_box_np, lidar_agent_record = inference_utils.get_cav_box(batch_data)
    #             infer_result.update({"cav_box_np": cav_box_np, \
    #                                  "lidar_agent_record": lidar_agent_record})
    #
    #         if (i % opt.save_vis_interval == 0) and (pred_box_tensor is not None):
    #             vis_save_path_root = os.path.join(opt.model_dir, f'vis_{infer_info}')
    #             if not os.path.exists(vis_save_path_root):
    #                 os.makedirs(vis_save_path_root)
    #
    #             # vis_save_path = os.path.join(vis_save_path_root, '3d_%05d.png' % i)
    #             # simple_vis.visualize(infer_result,
    #             #                     batch_data['ego'][
    #             #                         'origin_lidar'][0],
    #             #                     hypes['postprocess']['gt_range'],
    #             #                     vis_save_path,
    #             #                     method='3d',
    #             #                     left_hand=left_hand)
    #
    #             vis_save_path = os.path.join(vis_save_path_root, 'bev_%05d.png' % i)
    #             simple_vis.visualize(infer_result,
    #                                 batch_data['ego'][
    #                                     'origin_lidar'][0],
    #                                 hypes['postprocess']['gt_range'],
    #                                 vis_save_path,
    #                                 method='bev',
    #                                 left_hand=left_hand)
    #     torch.cuda.empty_cache()
    #
    # _, ap50, ap70 = eval_utils.eval_final_results(result_stat,
    #                             opt.model_dir, infer_info)


if __name__ == '__main__':
    main()
