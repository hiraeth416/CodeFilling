# -*- coding: utf-8 -*-
# Author: Yangheng Zhao <zhaoyangheng-sjtu@sjtu.edu.cn>

import os
from torch.utils.data import DataLoader, Subset
import torch
import sys
import time
retval = os.getcwd()
# print("1")
# print(retval)
# print(os.path.abspath(os.path.dirname(__file__)))
sys.path.append('../..')
sys.path.append('.')
from opencood.data_utils import datasets
from opencood.tools import train_utils, inference_utils
from opencood.hypes_yaml.yaml_utils import load_yaml
from opencood.visualization import simple_vis
from opencood.data_utils.datasets import build_dataset
import numpy as np

time_now = time.strftime("%Y%m%d-%H%M", time.localtime())
if __name__ == '__main__':
    sys.path.append('../..')
    sys.path.append('.')
    current_path = os.path.dirname(os.path.realpath(__file__))
    yaml_path = os.path.abspath(os.path.join(current_path, ".."))
    params = load_yaml(os.path.abspath(os.path.join(current_path,
                        "..","hypes_yaml/v2xsim2/visualization.yaml")))
    output_path = "/remote-home/share/junhaoge/data/v2xsim_vis" + "/" + time_now + "_delay_{}".format(params["time_delay"])
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    opencda_dataset = build_dataset(params, visualize=True, train=False)
    len = len(opencda_dataset)
    # sampled_indices = np.random.permutation(len)[:100]
    sampled_indices = np.arange(0, 200, 1)
    subset = Subset(opencda_dataset, sampled_indices)
    
    data_loader = DataLoader(subset, batch_size=1, num_workers=2,
                             collate_fn=opencda_dataset.collate_batch_test,
                             shuffle=False,
                             pin_memory=False)
    vis_gt_box = True
    vis_pred_box = False
    hypes = params

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    for i, batch_data in enumerate(data_loader):
        # print(i)
        batch_data = train_utils.to_device(batch_data, device)
        gt_box_tensor = opencda_dataset.post_processor.generate_gt_bbx(batch_data)

        # vis_save_path = os.path.join(output_path, '3d_%05d.png' % i)
        # simple_vis.visualize({},
        #                     batch_data['ego']['origin_lidar'][0],
        #                     hypes['postprocess']['gt_range'],
        #                     vis_save_path,
        #                     method='3d',
        #                     left_hand=False)
        infer_result = dict()
        infer_result["gt_box_tensor"] = gt_box_tensor
        vis_save_path = os.path.join(output_path, 'bev_%05d.png' % i)
        simple_vis.visualize(infer_result,
                            batch_data['ego']['origin_lidar'][0],
                            hypes['postprocess']['gt_range'],
                            vis_save_path,
                            method='bev',
                            left_hand=False)