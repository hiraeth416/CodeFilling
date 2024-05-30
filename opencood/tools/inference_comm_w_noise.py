# -*- coding: utf-8 -*-
# Author: Runsheng Xu <rxx3386@ucla.edu>, Hao Xiang <haxiang@g.ucla.edu>,
# License: TDG-Attribution-NonCommercial-NoDistrib


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
from opencood.utils.common_utils import update_dict
from opencood.visualization import vis_utils, my_vis, simple_vis
torch.multiprocessing.set_sharing_strategy('file_system')

def test_parser():
    parser = argparse.ArgumentParser(description="synthetic data generation")
    parser.add_argument('--model_dir', type=str, required=True,
                        help='Continued training path')
    parser.add_argument('--hypes_yaml', type=str, default=None,
                        help='hypes yaml path')
    parser.add_argument('--model_name', type=str,
                        default=None,
                        help='model name')
    parser.add_argument('--also_laplace', action='store_true',
                        help="whether to use laplace to simulate noise. Otherwise Gaussian")
    parser.add_argument('--fusion_method', type=str,
                        default='intermediate',
                        help='no, no_w_uncertainty, late, early or intermediate')
    parser.add_argument('--save_vis_interval', type=int, default=40,
                        help='interval of saving visualization')
    parser.add_argument('--save_npy', action='store_true',
                        help='whether to save prediction and gt result'
                             'in npy file')
    parser.add_argument('--save_track', action='store_true',
                        help='whether to save prediction and gt result for track'
                             'in npy file')
    parser.add_argument('--range', type=str, default="140.8,40.0",
                        help="detection range is [-140.8,+140.8m, -40m, +40m]")
    parser.add_argument('--modal', type=int, default=0,
                        help='used in heterogeneous setting, 0 lidaronly, 1 camonly, 2 ego_lidar_other_cam, 3 ego_cam_other_lidar')
    parser.add_argument('--no_score', action='store_true',
                        help="whether print the score of prediction")
    parser.add_argument('--note', default="", type=str, help="any other thing?")
    parser.add_argument('--comm_thre', default=0.0, type=float, help="communication threhold")
    parser.add_argument('--hist_len', default=5, type=int, help="history timestamps")
    parser.add_argument('--num_sweep_frames', default=6, type=int, help="total timestamps")
    parser.add_argument('--sampling_gap', default=1, type=int, help="temporal sampling gap")
    parser.add_argument('--only_hist', action='store_true', help="whether only use historical data")
    parser.add_argument('--wo_hist', action='store_true', help="whether use historical data")
    parser.add_argument('--wo_compensation', action='store_true', help="whether compensate historical data")
    parser.add_argument('--wo_colla', action='store_true', help="whether collaborate")
    parser.add_argument('--temporal_thre', default=0.0, type=float, help="history threshold")
    parser.add_argument('--R_thre', default=1.0, type=float, help="history threshold")
    parser.add_argument('--result_name', default="", type=str, help="result txt name")
    parser.add_argument('--noise', default=0.0, type=float, help="pose error")
    parser.add_argument('--delay_time', default="-1", type=int, help="latency time")
    opt = parser.parse_args()
    return opt


def main():
    opt = test_parser()
    opt.result_name = opt.model_name
    assert opt.fusion_method in ['late', 'early', 'intermediate', 'no', 'no_w_uncertainty', 'single'] 

    # hypes = yaml_utils.load_yaml(None, opt)
    # hypes = yaml_utils.load_yaml(None, opt)
    if opt.hypes_yaml is None:
        hypes = yaml_utils.load_yaml(None, opt)
    else:
        hypes = yaml_utils.load_yaml(opt.hypes_yaml, None)
    if opt.delay_time != -1:
        hypes['time_delay'] = opt.delay_time
    else:
        pass

    if "OPV2V" in hypes['test_dir'] or "v2xsim" in hypes['test_dir']:
        if opt.modal == 0:
            hypes['heter']['mapping_dict']['m1'] = 'm1'
            hypes['heter']['mapping_dict']['m2'] = 'm1'
            hypes['heter']['mapping_dict']['m3'] = 'm1'
            hypes['heter']['mapping_dict']['m4'] = 'm1'
            hypes['heter']['ego_modality'] = 'm1'
            hypes['model']['args']['ego_modality'] = 'm1'
            opt.note += '_lidaronly' 

        if opt.modal == 1:
            hypes['heter']['mapping_dict']['m1'] = 'm2'
            hypes['heter']['mapping_dict']['m2'] = 'm2'
            hypes['heter']['mapping_dict']['m3'] = 'm2'
            hypes['heter']['mapping_dict']['m4'] = 'm2'
            hypes['heter']['ego_modality'] = 'm2'
            hypes['model']['args']['ego_modality'] = 'm2'
            opt.note += '_camonly' 

        if opt.modal == 2:
            hypes['heter']['mapping_dict']['m1'] = 'm1'
            hypes['heter']['mapping_dict']['m2'] = 'm2'
            hypes['heter']['mapping_dict']['m3'] = 'm2'
            hypes['heter']['mapping_dict']['m4'] = 'm2'
            hypes['heter']['ego_modality'] = 'm1'
            hypes['model']['args']['ego_modality'] = 'm1'
            opt.note += 'ego_lidar_other_cam'

        if opt.modal == 3:
            hypes['heter']['mapping_dict']['m1'] = 'm2'
            hypes['heter']['mapping_dict']['m2'] = 'm1'
            hypes['heter']['mapping_dict']['m3'] = 'm1'
            hypes['heter']['mapping_dict']['m4'] = 'm1'
            hypes['heter']['ego_modality'] = 'm2'
            hypes['model']['args']['ego_modality'] = 'm2'
            opt.note += '_ego_cam_other_lidar'

        if opt.modal == 4:
            hypes['heter']['mapping_dict']['m1'] = 'm1'
            hypes['heter']['mapping_dict']['m2'] = 'm1'
            hypes['heter']['mapping_dict']['m3'] = 'm2'
            hypes['heter']['mapping_dict']['m4'] = 'm2'
            hypes['heter']['ego_modality'] = 'm1&m2'
            opt.note += 'ego_random_ratio0.5'
    else:
        if opt.modal == 0:
            hypes['heter']['mapping_dict']['m1'] = 'm1'
            hypes['heter']['mapping_dict']['m2'] = 'm1'
            hypes['heter']['ego_modality'] = 'm1'
            opt.note += '_lidaronly' 

        if opt.modal == 1:
            hypes['heter']['mapping_dict']['m1'] = 'm2'
            hypes['heter']['mapping_dict']['m2'] = 'm2'
            hypes['heter']['ego_modality'] = 'm2'
            opt.note += '_camonly' 

        if opt.modal == 2:
            hypes['heter']['mapping_dict']['m1'] = 'm1'
            hypes['heter']['mapping_dict']['m2'] = 'm2'
            hypes['heter']['ego_modality'] = 'm1'
            opt.note += 'ego_lidar_other_cam'

        if opt.modal == 3:
            hypes['heter']['mapping_dict']['m1'] = 'm2'
            hypes['heter']['mapping_dict']['m2'] = 'm1'
            hypes['heter']['ego_modality'] = 'm2'
            opt.note += '_ego_cam_other_lidar'

        if opt.modal == 4:
            hypes['heter']['mapping_dict']['m1'] = 'm1'
            hypes['heter']['mapping_dict']['m2'] = 'm2'
            hypes['heter']['ego_modality'] = 'm1&m2'
            opt.note += 'ego_random_ratio0.5'

        x_min, x_max = -eval(opt.range.split(',')[0]), eval(opt.range.split(',')[0])
        y_min, y_max = -eval(opt.range.split(',')[1]), eval(opt.range.split(',')[1])
        opt.note += f"_{x_max}_{y_max}"
        #hypes['fusion']['args']['grid_conf']['xbound'][0] = x_min
        #hypes['fusion']['args']['grid_conf']['xbound'] = [x_min, x_max, hypes['fusion']['args']['grid_conf']['xbound'][2]]
        #hypes['fusion']['args']['grid_conf']['ybound'] = [y_min, y_max, hypes['fusion']['args']['grid_conf']['ybound'][2]]
        #hypes['model']['args']['grid_conf'] = hypes['fusion']['args']['grid_conf']

        new_cav_range = [x_min, y_min, hypes['postprocess']['anchor_args']['cav_lidar_range'][2], \
                            x_max, y_max, hypes['postprocess']['anchor_args']['cav_lidar_range'][5]]
        
        hypes['preprocess']['cav_lidar_range'] =  new_cav_range
        hypes['postprocess']['anchor_args']['cav_lidar_range'] = new_cav_range
        hypes['postprocess']['gt_range'] = new_cav_range
        hypes = update_dict(hypes, {
            "cav_lidar_range": new_cav_range,
            "lidar_range": new_cav_range,
            "gt_range": new_cav_range
        })
        if 'camera_mask_args' in hypes['model']['args']:
            hypes['model']['args']['camera_mask_args']['cav_lidar_range'] = new_cav_range

        # reload anchor
        yaml_utils_lib = importlib.import_module("opencood.hypes_yaml.yaml_utils")
        for name, func in yaml_utils_lib.__dict__.items():
            if name == hypes["yaml_parser"]:
                parser_func = func
        hypes = parser_func(hypes)

    # hypes['model']['args']['fusion_args']['communication']['thre'] = opt.comm_thre
    # hypes['model']['args']['fusion_args']['communication']['R_thre'] = opt.R_thre
    hypes['model']['args']['hist_len'] = opt.hist_len
    hypes['model']['args']['skip_scale'] = opt.sampling_gap
    hypes['num_sweep_frames'] = opt.num_sweep_frames
    
    hypes['model']['args']['temporal_args'] = {}
    hypes['model']['args']['temporal_args']['only_hist'] = opt.only_hist
    hypes['model']['args']['temporal_args']['with_hist'] = not opt.wo_hist 
    hypes['model']['args']['temporal_args']['with_compensation'] = not opt.wo_compensation
    hypes['model']['args']['temporal_args']['wo_colla'] = opt.wo_colla
    hypes['model']['args']['temporal_args']['sampling_gap'] = opt.sampling_gap
    hypes['model']['args']['temporal_args']['temporal_thre'] = opt.temporal_thre
    
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
    

    # add noise to pose.
    pos_std_list = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
    rot_std_list = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
    pos_mean_list = [0, 0, 0, 0, 0, 0, 0]
    rot_mean_list = [0, 0, 0, 0, 0, 0, 0]
    # pos_std_list = [opt.noise]
    # rot_std_list = [opt.noise]
    # pos_std_list = [0.8, 1.0, 1.5]
    # rot_std_list = [0.8, 1.0, 1.5]
    # pos_mean_list = [0]
    # rot_mean_list = [0]

    
    if opt.also_laplace:
        use_laplace_options = [False, True]
    else:
        use_laplace_options = [False]

    for use_laplace in use_laplace_options:
        AP30 = []
        AP50 = []
        AP70 = []
        for (pos_mean, pos_std, rot_mean, rot_std) in zip(pos_mean_list, pos_std_list, rot_mean_list, rot_std_list):
            # setting noise
            np.random.seed(303)
            noise_setting = OrderedDict()
            noise_args = {'pos_std': pos_std,
                          'rot_std': rot_std,
                          'pos_mean': pos_mean,
                          'rot_mean': rot_mean}

            noise_setting['add_noise'] = True
            noise_setting['args'] = noise_args

            suffix = ""
            if use_laplace:
                noise_setting['args']['laplace'] = True
                suffix = "_laplace"
    
            # build dataset for each noise setting
            print('Dataset Building')
            print(f"Noise Added: {pos_std}/{rot_std}/{pos_mean}/{rot_mean}.")
            hypes.update({"noise_setting": noise_setting})
            opencood_dataset = build_dataset(hypes, visualize=True, train=False)
            # opencood_dataset_subset = Subset(opencood_dataset, range(160,1230))
            # data_loader = DataLoader(opencood_dataset_subset,
            data_loader = DataLoader(opencood_dataset,
                                    batch_size=1,
                                    num_workers=4,
                                    collate_fn=opencood_dataset.collate_batch_test,
                                    shuffle=False,
                                    pin_memory=False,
                                    drop_last=False)
            
            # Create the dictionary for evaluation
            result_stat = {0.3: {'tp': [], 'fp': [], 'gt': 0, 'score': []},                
                        0.5: {'tp': [], 'fp': [], 'gt': 0, 'score': []},                
                        0.7: {'tp': [], 'fp': [], 'gt': 0, 'score': []}}

            
            infer_info = opt.model_name + '_' + opt.fusion_method + '_' + opt.note + '_noise_' +str(pos_std)

            scene_idxs = []
            comm_rates = []
            for i, batch_data in enumerate(data_loader):
                print(f"{infer_info}_{i}")
                if opt.save_track:
                    scene_idx = data_loader.dataset.retrieve_scene_idx(i) + 1
                    if scene_idx in scene_idxs:
                        timestamp += 1
                    else:
                        timestamp = 1
                        scene_idxs.append(scene_idx)
                        npy_save_path = os.path.join(opt.model_dir, infer_info, 'npy', str(scene_idx))
                        # model.resetKal()
                    if not os.path.exists(npy_save_path):
                        os.makedirs(npy_save_path)
                if batch_data is None:
                    continue
                with torch.no_grad():
                    batch_data = train_utils.to_device(batch_data, device)

                    if opt.fusion_method == 'late':
                        infer_func = inference_utils.inference_late_fusion_w_idx if opt.save_track else inference_utils.inference_late_fusion
                        infer_result = infer_func(batch_data,
                                                    model,
                                                    opencood_dataset)
                    elif opt.fusion_method == 'early':
                        infer_func = inference_utils.inference_early_fusion_w_idx if opt.save_track else inference_utils.inference_early_fusion
                        infer_result = infer_func(batch_data,
                                                                model,
                                                                opencood_dataset)
                    elif opt.fusion_method == 'intermediate':
                        infer_func = inference_utils.inference_intermediate_fusion_w_idx if opt.save_track else inference_utils.inference_intermediate_fusion
                        infer_result = infer_func(batch_data,
                                                    model,
                                                    opencood_dataset)
                    elif opt.fusion_method == 'no':
                        infer_func = inference_utils.inference_no_fusion_w_idx if opt.save_track else inference_utils.inference_no_fusion
                        infer_result = infer_func(batch_data,
                                                    model,
                                                    opencood_dataset)
                    elif opt.fusion_method == 'no_w_uncertainty':
                        infer_result = inference_utils.inference_no_fusion_w_uncertainty(batch_data,
                                                                        model,
                                                                        opencood_dataset)
                    elif opt.fusion_method == 'single':
                        infer_func = inference_utils.inference_no_fusion_w_idx if opt.save_track else inference_utils.inference_no_fusion
                        infer_result = infer_func(batch_data,
                                                    model,
                                                    opencood_dataset,
                                                    single_gt=True)
                    else:
                        raise NotImplementedError('Only single, no, no_w_uncertainty, early, late and intermediate'
                                                'fusion is supported.')

                    pred_box_tensor = infer_result['pred_box_tensor']
                    gt_box_tensor = infer_result['gt_box_tensor']
                    pred_score = infer_result['pred_score']
                    if "comm_rate" in infer_result:
                        comm_rates.append(infer_result["comm_rate"].cpu().numpy())
                    
                    eval_utils.caluclate_tp_fp(pred_box_tensor,
                                            pred_score,
                                            gt_box_tensor,
                                            result_stat,
                                            0.3)
                    eval_utils.caluclate_tp_fp(pred_box_tensor,
                                            pred_score,
                                            gt_box_tensor,
                                            result_stat,
                                            0.5)
                    eval_utils.caluclate_tp_fp(pred_box_tensor,
                                            pred_score,
                                            gt_box_tensor,
                                            result_stat,
                                            0.7)

                    if opt.save_npy:
                        npy_save_path = os.path.join(opt.model_dir, 'npy')
                        if not os.path.exists(npy_save_path):
                            os.makedirs(npy_save_path)
                        inference_utils.save_prediction_gt(pred_box_tensor,
                                                        gt_box_tensor,
                                                        batch_data['ego'][
                                                            'origin_lidar'][0],
                                                        i,
                                                        npy_save_path)
                    if opt.save_track:
                        # scene_idx = data_loader.dataset.retrieve_scene_idx(i) + 1
                        # if scene_idx in scene_idxs:
                        #     timestamp += 1
                        # else:
                        #     timestamp = 1
                        #     scene_idxs.append(scene_idx)
                        #     npy_save_path = os.path.join(opt.model_dir, infer_info, 'npy', str(scene_idx))
                        # if not os.path.exists(npy_save_path):
                        #     os.makedirs(npy_save_path)
                        pred_2d, gt_2d = inference_utils.save_mot(pred_box_tensor,
                                                        pred_score, 
                                                        gt_box_tensor,
                                                        infer_result['gt_idx'],
                                                        batch_data['ego'][
                                                            'origin_lidar'][0],
                                                        batch_data['ego'][
                                                            'lidar_pose'][0:1],    
                                                        timestamp,
                                                        npy_save_path)
                        

                    if not opt.no_score:
                        infer_result.update({'score_tensor': pred_score})

                    if getattr(opencood_dataset, "heterogeneous", False):
                        cav_box_np, lidar_agent_record = inference_utils.get_cav_box(batch_data)
                        infer_result.update({"cav_box_np": cav_box_np, \
                                            "lidar_agent_record": lidar_agent_record})

                    if (i % opt.save_vis_interval == 0) and (pred_box_tensor is not None):
                        vis_save_path_root = os.path.join(opt.model_dir, f'vis_{infer_info}')
                        if not os.path.exists(vis_save_path_root):
                            os.makedirs(vis_save_path_root)

                        # vis_save_path = os.path.join(vis_save_path_root, '3d_%05d.png' % i)
                        # simple_vis.visualize(infer_result,
                        #                     batch_data['ego'][
                        #                         'origin_lidar'][0],
                        #                     hypes['postprocess']['gt_range'],
                        #                     vis_save_path,
                        #                     method='3d',
                        #                     left_hand=left_hand)
                        
                        vis_save_path = os.path.join(vis_save_path_root, 'bev_%05d.png' % i)
                        simple_vis.visualize(infer_result,
                                            batch_data['ego'][
                                                'origin_lidar'][0],
                                            hypes['postprocess']['gt_range'],
                                            vis_save_path,
                                            method='bev',
                                            left_hand=left_hand)
                torch.cuda.empty_cache()
            
            if opt.save_track:
                seqmap_save_path = os.path.join(opt.model_dir, infer_info, 'track', 'gt', 'seqmaps', 'OPV2V-test.txt')
                if not os.path.exists(os.path.dirname(seqmap_save_path)):
                    os.makedirs(os.path.dirname(seqmap_save_path))
                with open(seqmap_save_path, 'w') as f:
                    f.write('name'+'\n')
                    for scene_idx in list(set(scene_idxs)):
                        f.write(str(scene_idx)+'\n')

            ap30, ap50, ap70 = eval_utils.eval_final_results(result_stat,
                                        opt.model_dir, infer_info)
            
            if len(comm_rates) > 0:
                comm_rates = sum(comm_rates) / len(comm_rates)
            else:
                comm_rates = 0.0

            detection_path = os.path.join(opt.model_dir, 'detection_noise')
            if not os.path.exists(detection_path):
                os.makedirs(detection_path)
                
            with open(os.path.join(opt.model_dir,'detection_noise', '{}_noise.txt'.format(opt.model_name)), 'a+') as f:
                f.write('ap30: {:.04f} ap50: {:.04f} ap70: {:.04f} comm_thre: {:.04f} comm_rate: {:.06f} pos_std: {:.04f} rot_std: {:.04f}\n'.format(ap30, ap50, ap70, opt.comm_thre, comm_rates, pos_std, rot_std))
            
            AP30.append(ap30)
            AP50.append(ap50)
            AP70.append(ap70)

            dump_dict = {'ap30': AP30 ,'ap50': AP50, 'ap70': AP70}
            yaml_utils.save_yaml(dump_dict, os.path.join(opt.model_dir, f'AP030507{suffix}.yaml'))

    
if __name__ == '__main__':
    main()
