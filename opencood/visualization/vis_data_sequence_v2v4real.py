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
import numpy as np
import opencood.hypes_yaml.yaml_utils as yaml_utils
from opencood.tools import train_utils
from opencood.data_utils.datasets import build_dataset
from opencood.utils.camera_utils import coord_3d_to_2d, denormalize_img
from matplotlib import pyplot as plt
import imageio
import tensorboard

history_len = 5
future_len = 10
color_list = [(21/255,101/255,192/255),(164/255,19/255,60/255),(216/255,154/255,158/255)]
RGBs = [(135, 206, 235), (255, 198, 20),  (206,105,231), (147, 224, 255), (199, 237, 233)]

from matplotlib import pyplot as plt
import numpy as np
import copy

from opencood.tools.inference_utils import get_cav_box
import opencood.visualization.simple_plot3d.canvas_3d as canvas_3d
import opencood.visualization.simple_plot3d.canvas_bev as canvas_bev

def visualize_multiagent(infer_result, pcd, pc_range, save_path, method='3d', vis_gt_box=True, vis_pred_box=True, left_hand=False):
        """
        Visualize the prediction, ground truth with point cloud together.
        They may be flipped in y axis. Since carla is left hand coordinate, while kitti is right hand.

        Parameters
        ----------
        infer_result:
            pred_box_tensor : torch.Tensor
                (N, 8, 3) prediction.

            gt_tensor : torch.Tensor
                (N, 8, 3) groundtruth bbx
            
            uncertainty_tensor : optional, torch.Tensor
                (N, ?)

            lidar_agent_record: optional, torch.Tensor
                (N_agnet, )


        pcd : list of torch.Tensor
            PointCloud, (N, 4).

        pc_range : list
            [xmin, ymin, zmin, xmax, ymax, zmax]

        save_path : str
            Save the visualization results to given path.

        dataset : BaseDataset
            opencood dataset object.

        method: str, 'bev' or '3d'

        """
        plt.figure(figsize=[(pc_range[3]-pc_range[0])/40, (pc_range[4]-pc_range[1])/40])
        pc_range = [int(i) for i in pc_range]
        # pcd_np = pcd.cpu().numpy()

        pred_box_tensor = infer_result.get("pred_box_tensor", None)
        gt_box_tensor = infer_result.get("gt_box_tensor", None)

        if gt_box_tensor is not None:
            try:
                gt_box_np = gt_box_tensor.cpu().numpy()
            except AttributeError:
                gt_box_np = gt_box_tensor
            gt_name = ['gt'] * gt_box_np.shape[0]

        if method == 'bev':
            canvas = canvas_bev.Canvas_BEV_heading_right(canvas_shape=((pc_range[4]-pc_range[1])*10, (pc_range[3]-pc_range[0])*10),
                                            canvas_x_range=(pc_range[0], pc_range[3]), 
                                            canvas_y_range=(pc_range[1], pc_range[4]),
                                            canvas_bg_color=(255, 255, 255),
                                            left_hand=left_hand) 
            for i, lidar in enumerate(pcd):
                # lidar = lidar.cpu().numpy()
                canvas_xy, valid_mask = canvas.get_canvas_coords(lidar) # Get Canvas Coords
                canvas.draw_canvas_points(canvas_xy[valid_mask], colors=RGBs[i], radius=1) # Only draw valid points
            if gt_box_tensor is not None:
                canvas.draw_boxes(gt_box_np,colors=(0,180,0), texts=gt_name)
            if pred_box_tensor is not None:
                canvas.draw_boxes(pred_box_np, colors=(255,0,0), texts=pred_name)

            # heterogeneous
            lidar_agent_record = infer_result.get("lidar_agent_record", None)
            cav_box_np = infer_result.get("cav_box_np", None)
            if lidar_agent_record is not None:
                cav_box_np = copy.deepcopy(cav_box_np)
                for i, islidar in enumerate(lidar_agent_record):
                    text = ['lidar'] if islidar else ['camera']
                    color = (0,191,255) if islidar else (255,185,15)
                    canvas.draw_boxes(cav_box_np[i:i+1], colors=color, texts=text)



        elif method == '3d':
            canvas = canvas_3d.Canvas_3D(left_hand=left_hand)
            canvas_xy, valid_mask = canvas.get_canvas_coords(pcd_np)
            canvas.draw_canvas_points(canvas_xy[valid_mask])
            if gt_box_tensor is not None:
                canvas.draw_boxes(gt_box_np,colors=(0,255,0), texts=gt_name)
            if pred_box_tensor is not None:
                canvas.draw_boxes(pred_box_np, colors=(255,0,0), texts=pred_name)

            # heterogeneous
            lidar_agent_record = infer_result.get("lidar_agent_record", None)
            cav_box_np = infer_result.get("cav_box_np", None)
            if lidar_agent_record is not None:
                cav_box_np = copy.deepcopy(cav_box_np)
                for i, islidar in enumerate(lidar_agent_record):
                    text = ['lidar'] if islidar else ['camera']
                    color = (0,191,255) if islidar else (255,185,15)
                    canvas.draw_boxes(cav_box_np[i:i+1], colors=color, texts=text)

        else:
            raise(f"Not Completed for f{method} visualization.")

        plt.axis("off")

        plt.imshow(canvas.canvas)
        plt.tight_layout()
        plt.savefig(save_path, transparent=False, dpi=500)
        plt.clf()
        plt.close()

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

def save_video(output_path, sampled_indices, viz_3d_flag, viz_bev_flag):
    if viz_3d_flag:
        img_array = []
        for idx in sampled_indices:
            image_path = f"{output_path}/3d_%05d.png" % idx
            if os.path.exists(image_path):
                image = imageio.v2.imread(image_path)
                img_array.append(image)

        gif_out = os.path.join(output_path, f"result_GIF_3d_{sampled_indices[0]}_{sampled_indices[-1]}.gif")
        imageio.mimsave(gif_out, [x for x in img_array], 'GIF', duration=0.15)

    if viz_bev_flag:
        img_array = []
        for idx in sampled_indices:
            image_path = f"{output_path}/bev_%05d.png" % idx
            if os.path.exists(image_path):
                image = imageio.v2.imread(image_path)
                img_array.append(image)

        gif_out = os.path.join(output_path, f"result_GIF_bev_{sampled_indices[0]}_{sampled_indices[-1]}.gif")
        imageio.mimsave(gif_out, [x for x in img_array], 'GIF', duration=0.15)


if __name__ == '__main__':
    viz_3d_flag = False
    viz_bev_flag = True
    current_path = os.path.dirname(os.path.realpath(__file__))
    hypes = load_yaml(os.path.join(current_path,
                                    '../hypes_yaml/v2v4real/visualization_v2v4real.yaml'))
    output_path = "/root/percp/OpenCOODv2/OpenCOODv2/data_vis/v2v4real"
    
    print('Dataset Building')
    opencda_dataset = build_dataset(hypes, visualize=True, train=False)

    stard_idx = 610
    duration = 30
    sampled_indices = range(stard_idx, stard_idx+duration)
    subset = Subset(opencda_dataset, sampled_indices)
    
    data_loader = DataLoader(subset, batch_size=1, num_workers=2,
                             collate_fn=opencda_dataset.collate_batch_test,
                             shuffle=False,
                             pin_memory=False)
    vis_gt_box = True
    vis_pred_box = False

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    gt_boxes = []
    record_len = []
    images = []
    int_matrixs = []
    ext_matrixs = []
    for i, batch_data in enumerate(data_loader):
        # batch_data = train_utils.to_device(batch_data, device)
        gt_box_tensor = opencda_dataset.post_processor.generate_gt_bbx(batch_data).cpu().numpy()
        # image = np.array(denormalize_img(batch_data['ego']['image_inputs']['imgs'][0,0]))  # 3, 480, 640
        # int_matrix = batch_data['ego']['image_inputs']['intrins'].cpu().numpy()[0,0]  # 1, 4, 3, 3
        # ext_matrix = batch_data['ego']['image_inputs']['extrinsics'].cpu().numpy()[0,0]  # 1, 4, 3, 3
        
        # gt_boxes.append(gt_box_tensor)
        # images.append(image)
        # int_matrixs.append(int_matrix)
        # ext_matrixs.append(ext_matrix)
        # record_len.append(gt_box_tensor.shape[0])

        # if i >= (history_len + future_len):
        #     cur_gt_box = np.concatenate(gt_boxes[-(history_len + future_len):])
        #     cur_int_matrix = int_matrixs[-future_len]
        #     cur_ext_matrix = ext_matrixs[-future_len]
        #     gt_box2d, gt_box2d_mask, _ = coord_3d_to_2d(cur_gt_box, cur_int_matrix, cur_ext_matrix, image_H=600, image_W=800, image=None, idx=None)
        #     cur_image = images[-future_len]
        #     cur_record_len = record_len[-(history_len + future_len):]
        #     vis_seq(cur_image, cur_record_len, gt_box2d, gt_box2d_mask, output_path, i)
        
        infer_result = {}
        infer_result['gt_box_tensor'] = gt_box_tensor

        if viz_3d_flag:
            vis_save_path = os.path.join(output_path, '3d_%05d.png' % (i+sampled_indices[0]))
            simple_vis.visualize(infer_result,
                                batch_data['ego']['origin_lidar'][0],
                                hypes['postprocess']['gt_range'],
                                vis_save_path,
                                method='3d',
                                vis_gt_box = vis_gt_box,
                                vis_pred_box = vis_pred_box,
                                left_hand=False)

        if viz_bev_flag:
            projected_lidar_list = batch_data['ego']['projected_lidar_list']
            vis_save_path = os.path.join(output_path, 'bev_%05d.png' % (i+sampled_indices[0]))
            print(vis_save_path)
            visualize_multiagent(infer_result,
                                projected_lidar_list,
                                hypes['postprocess']['gt_range'],
                                vis_save_path,
                                method='bev',
                                vis_gt_box = vis_gt_box,
                                vis_pred_box = vis_pred_box,
                                left_hand=False)

    save_video(output_path, sampled_indices, viz_3d_flag, viz_bev_flag)