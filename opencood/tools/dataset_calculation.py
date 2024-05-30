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
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))
from torch.utils.data import DataLoader, Subset
import numpy as np
import opencood.hypes_yaml.yaml_utils as yaml_utils
from opencood.tools import train_utils, inference_utils
from opencood.data_utils.datasets import build_dataset
from opencood.utils import eval_utils
from opencood.visualization import vis_utils, my_vis, simple_vis
from opencood.utils.common_utils import update_dict
import matplotlib.pyplot as plt
import json
torch.multiprocessing.set_sharing_strategy('file_system')

def read_json(json_path:str)->dict:
    """
    Read json file
    Args:
        json_path: path of json file

    Return:
        json_data: json data
    """
    with open(json_path, 'r') as f:
        json_data = json.load(f)
    return json_data

def write_json(json_path:str, json_data:dict):
    """
    Write json file
    Args:
        json_path: path of json file
        json_data: json data

    Return:
        None
    """
    with open(json_path, 'w') as f:
        json.dump(json_data, f, indent=4)

def test_parser():
    parser = argparse.ArgumentParser(description="synthetic data generation")
    parser.add_argument('--model_dir', type=str, required=True,
                        help='Continued training path')
    parser.add_argument('--hypes_yaml', type=str, default=None,
                        help='hypes yaml path')
    parser.add_argument('--fusion_method', type=str,
                        default='intermediate',
                        help='no, no_w_uncertainty, late, early or intermediate')
    parser.add_argument('--model_name', type=str,
                        default=None,
                        help='model name')
    parser.add_argument('--save_vis_interval', type=int, default=40,
                        help='interval of saving visualization')
    parser.add_argument('--save_npy', action='store_true',
                        help='whether to save prediction and gt result'
                             'in npy file')
    parser.add_argument('--save_track', action='store_true',
                        help='whether to save prediction and gt result for track'
                             'in npy file')
    parser.add_argument('--range', type=str, default="140.8,40",
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
    parser.add_argument('--sample_method', type=str, default="none",
                        help="the method to downsample the point cloud")
    parser.add_argument('--store_boxes', default=True,action='store_true',
                        help= "store detection boxes and gt boxes")
    parser.add_argument('--sampling_rate', type=float, default=1.0)
    parser.add_argument('--sampler_path', type=str, default=None)
    parser.add_argument('--box_ratio',type=float,default=1.0)
    parser.add_argument('--background_ratio',type=float,default=1.0)
    parser.add_argument('--expansion_ratio',type=float,default=20.0)
    parser.add_argument('--w2cthreshold',type=float,default=0.01)
    parser.add_argument('--vis_score',type=str,default='confidence',
                        help='confidence or uncertainty')

    parser.add_argument('--w_solver', action='store_true', help="whether use solver")
    parser.add_argument('--solver_thre', default=1.0, type=float, help="solver threhold")
    parser.add_argument('--solver_method', default='sum', type=str, help="solver method: max/sum")
    
    parser.add_argument('--min_cav_num', default = 10, type = int)
    parser.add_argument('--max_cav_num', default = 10, type = int)
    opt = parser.parse_args()
    return opt


def main():
    opt = test_parser()
    data_len = 0
    data_type = 'train'
    gt_box_num = 0
    assert opt.fusion_method in ['late', 'early', 'intermediate', 'no', 'no_w_uncertainty', 'single']
    if opt.hypes_yaml is None:
        hypes = yaml_utils.load_yaml(None, opt)
    else:
        hypes = yaml_utils.load_yaml(opt.hypes_yaml, None)
    if opt.delay_time != -1:
        hypes['time_delay'] = opt.delay_time
    else:
        pass

    hypes.update({'sample_method': opt.sample_method})
    hypes.update({'sampling_rate': opt.sampling_rate})
    hypes.update({'store_boxes': opt.store_boxes})
    hypes.update({'model_dir':opt.model_dir})
    hypes.update({'sampler_path':opt.sampler_path})
    hypes.update({'expansion_ratio':opt.expansion_ratio})
    hypes.update({'box_ratio':opt.box_ratio})
    hypes.update({'background_ratio':opt.background_ratio})
    hypes.update({'late_choose_method':False})
    hypes.update({'min_cav_num': opt.min_cav_num})
    hypes.update({'max_cav_num': opt.max_cav_num})
    hypes['validate_dir'] = hypes['test_dir']
    # build dataset for each noise setting
    print('Dataset Building')
    opencood_dataset = build_dataset(hypes, visualize=True, train=True)
    # opencood_dataset_subset = Subset(opencood_dataset, range(1,200))
    # data_loader = DataLoader(opencood_dataset_subset,
    data_loader = DataLoader(opencood_dataset,
                            batch_size=1,
                            num_workers=8,
                            collate_fn=opencood_dataset.collate_batch_test,
                            shuffle=False,
                            pin_memory=False,
                            drop_last=False,
                            prefetch_factor=8)
    data_len= len(data_loader)
    print(data_len)
    vis_save_path_root = "/dssg/home/acct-umjpyb/umjpyb/junhaoge/OpenCOODv2/vis_file/"
    with open(os.path.join(vis_save_path_root, 'dataset_len.txt'), 'a+') as f:
        f.write('len of {} dataset: {}\n'.format(data_type, data_len))
    print("Dataset Building Ends")
    for i, batch_data in enumerate(data_loader):
        print("cur idx: ", i)
        if batch_data is None:
            # print("continue")
            continue
        # json_path = os.path.join(vis_save_path_root, 'data_batch.json')
        # write_json(json_path,batch_data['ego'])
        # print(batch_data['ego'].keys())
        # scene_id = batch_data['ego']['sample_idx']
        # cav_id_list = batch_data['ego']['cav_id_list']
        # print(scene_id)
        # if i == 1:
        #     for j, _ in enumerate(batch_data['ego']['origin_lidar']):
        #         vis_save_path = os.path.join(vis_save_path_root, '{}_000068_lidar.png'.format(cav_id_list[j]))
        #         simple_vis.visualize({},
        #                             batch_data['ego']['origin_lidar'][j],
        #                             hypes['postprocess']['gt_range'],
        #                             vis_save_path,
        #                             method='bev',
        #                             left_hand=True)
        gt_box_num += len(batch_data['ego']['object_ids'])
        print('gt_box_num: ', gt_box_num)
    with open(os.path.join(vis_save_path_root, 'gt_box_num.txt'), 'a+') as f:
        f.write('len of {} dataset: {}\n'.format(data_type, gt_box_num))
        # torch.cuda.empty_cache()
    # with open(os.path.join(vis_save_path_root, 'dataset_len.txt'), 'a+') as f:
    #     f.write('len of {} dataset: {}\n'.format(data_type, data_len))
    
if __name__ == '__main__':
    main()
