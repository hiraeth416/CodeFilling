# -*- coding: utf-8 -*-
# Author: Runsheng Xu <rxx3386@ucla.edu>
# License: TDG-Attribution-NonCommercial-NoDistrib


import os
from torch.utils.data import DataLoader, Subset
from opencood.data_utils import datasets
import torch
from opencood.tools import train_utils, inference_utils
from opencood.hypes_yaml.yaml_utils import load_yaml
from opencood.visualization import vis_utils, simple_vis
import opencood.utils.pcd_utils as pcd_utils
import numpy as np
import opencood.hypes_yaml.yaml_utils as yaml_utils
from opencood.tools import train_utils
from opencood.data_utils.datasets import build_dataset
from opencood.utils.camera_utils import coord_3d_to_2d, denormalize_img
from matplotlib import pyplot as plt
import imageio

history_len = 5
future_len = 10
color_list = [(21/255,101/255,192/255),(164/255,19/255,60/255),(216/255,154/255,158/255)]

def vis_img(image, gt_box2d, gt_box2d_mask, output_path, idx):
    plt.imshow(image)
    N = gt_box2d.shape[0]
    for i in range(N):
        if gt_box2d_mask[i]:
            coord2d = gt_box2d[i]
            for start, end in [(0, 1), (1, 2), (2, 3), (3, 0),
                            (0, 4), (1, 5), (2, 6), (3, 7),
                            (4, 5), (5, 6), (6, 7), (7, 4)]:
                plt.plot(coord2d[[start,end]][:,0], coord2d[[start,end]][:,1], marker="o", c='g')
    plt.savefig(f"{output_path}/{idx}.png", dpi=300)
    plt.clf()

def vis_seq(image, record_len, gt_box2d, gt_box2d_mask, output_path, idx):
    plt.imshow(image)
    plt.xticks([])
    plt.yticks([])
    assert gt_box2d.shape[0] == sum(record_len)
    N = gt_box2d.shape[0]
    alpha = np.arange(0.1, 1.0, 0.9/history_len)
    alphas = []
    colors = []
    for i in range(len(record_len)):
        if i < history_len:
            alphas.extend([alpha[i] for _ in range(record_len[i])])
            colors.extend([color_list[0] for _ in range(record_len[i])])
        else:
            alphas.extend([alpha[-1] for _ in range(record_len[i])])
            colors.extend([color_list[1] for _ in range(record_len[i])])
    frames = np.cumsum(record_len)
    for i in range(N):
        if gt_box2d_mask[i]:
            coord2d = gt_box2d[i]
            c_x = coord2d[:,0].mean()
            c_y = coord2d[:,1].mean()
            # print('c_x: {} c_y: {}'.format(c_x, c_y))
            # if (30 < c_x < 100) and (280 < c_y < 310):
            plt.plot(c_x, c_y, marker="o", c=colors[i], markersize=3, alpha=alphas[i])
            # plt.text(c_x, c_y, s='{}_{}'.format(c_x, c_y))
    plt.savefig(f"{output_path}/{idx}.png", dpi=300)
    plt.clf()

def save_video(output_path):
    img_array = []
    for idx in range(8, 40):
        image_path = f"{output_path}/{idx}.png"
        if os.path.exists(image_path):
            image = imageio.v2.imread(image_path)
            img_array.append(image)

    gif_out = os.path.join(output_path, "result_GIF.gif")
    imageio.mimsave(gif_out, [x for x in img_array], 'GIF', duration=0.15)


if __name__ == '__main__':
    current_path = os.path.dirname(os.path.realpath(__file__))
    hypes = load_yaml(os.path.join(current_path,
                                    '../hypes_yaml/opv2v/visualization_opv2v.yaml'))
    # output_path = "/GPFS/rhome/yifanlu/OpenCOOD/data_vis/opv2v_ego_view_others_pc"
    output_path = "/dssg/home/acct-umjpyb/umjpyb/junhaoge/OpenCOODv2/vis_file"
    lidar_root_dir = "/dssg/home/acct-umjpyb/umjpyb/jtpeng/OPV2V_MoreAgents/test_64_half/2021_08_18_19_48_05"
    cav_id_list = ["1045", "1054", "1067", "1068", "1070", "1071", "1072", "1073", "1077", "1080"]
    for cav_id in cav_id_list:
        lidar_path = os.path.join(lidar_root_dir, cav_id, "000068.pcd")
        vis_save_path = os.path.join(output_path, f'{cav_id}_000068_lidar0.png')
        lidar_np= torch.Tensor(pcd_utils.pcd_to_np(lidar_path))
        simple_vis.visualize({},
                            lidar_np,
                            hypes['postprocess']['gt_range'],
                            vis_save_path,
                            method='bev',
                            left_hand=True)
        
        # projected_lidar_list = batch_data['ego']['projected_lidar_list']

        # for idx, lidar in enumerate(projected_lidar_list):
        #     lidar = torch.from_numpy(lidar)
        #     vis_save_path = os.path.join(output_path, 'bev_%05d_%01d.png' % (i, idx))
        #     print(vis_save_path)
        #     simple_vis.visualize(None,
        #                         gt_box_tensor,
        #                         lidar,
        #                         hypes['postprocess']['gt_range'],
        #                         vis_save_path,
        #                         method='bev',
        #                         vis_gt_box = vis_gt_box,
        #                         vis_pred_box = vis_pred_box,
        #                         left_hand=True)
    # save_video(output_path)