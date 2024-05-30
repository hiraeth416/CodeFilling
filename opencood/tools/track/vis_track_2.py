import opencood.utils.pcd_utils as pcd_utils
import os
import argparse
from collections import OrderedDict
from matplotlib import pyplot as plt
import numpy as np
import copy
import imageio
from matplotlib.pyplot import cm
from tqdm import tqdm
import opencood.visualization.simple_plot3d.canvas_3d as canvas_3d
import opencood.visualization.simple_plot3d.canvas_bev as canvas_bev
from opencood.utils.box_utils import boxes_to_corners_3d


# Set color for each sequence
# color_list = np.linspace(0,1,100)
# np.random.shuffle(color_list)
# _color = iter(cm.gist_ncar(color_list))
# box_idx = np.unique(det_box[:,1])
# for box_id in box_idx:
#     if box_id not in color_dict:
#         color_dict[box_id] = np.uint(next(_color).reshape(1,-1)[:,:3] * 255)[:,::-1]
#         while np.abs((color_dict[box_id] - np.array([0,0,255]))).sum() < 20:
#             color_dict[box_id] = np.uint(next(_color).reshape(1,-1)[:,:3] * 255)[:,::-1]
#         print(color_dict[box_id])


def box_xywh_2_xyxy(box):
    xy0 = box[:, :2].reshape(-1, 1, 2)
    wh = box[:, 2:4].reshape(-1, 1, 2)

    xy1 = copy.deepcopy(xy0)
    xy1[:, :, 0] += wh[:, :, 0]

    xy2 = copy.deepcopy(xy0)
    xy2 += wh

    xy3 = copy.deepcopy(xy0)
    xy3[:, :, 1] += wh[:, :, 1]

    corners = np.concatenate([xy0, xy1, xy2, xy3], axis=1)
    return corners


def visualize(pred_box_np1, pred_box_np2, gt_box_np, pcd_np, save_path, method='bev', left_hand=False):
    """
    Visualize the prediction, ground truth with point cloud together.
    They may be flipped in y axis. Since carla is left hand coordinate, while kitti is right hand.

    Parameters
    ----------
    pred_box_np : numpy.array
        (N, 10) prediction. [frame_id,box_id,x,y,h,w,-1,-1,-1,-1]

    gt_box_np : numpy.array
        (N, 10) groundtruth bbx. [frame_id,box_id,x,y,h,w,-1,-1,-1,-1]

    pcd_np : numpy.array
        PointCloud, (N, 4).

    pc_range : list
        [xmin, ymin, zmin, xmax, ymax, zmax]

    save_path : str
        Save the visualization results to given path.

    method: str, 'bev' or '3d'

    """
    pc_range = [-140.8, -40, -3, 140.8, 40, 1]
    plt.figure(figsize=[(pc_range[3] - pc_range[0]) / 40, (pc_range[4] - pc_range[1]) / 40])
    pc_range = [int(i) for i in pc_range]

    if pred_box_np1 is not None:
        pred_name1 = [str(int(x)) for x in pred_box_np1[:, 1]]

    if pred_box_np2 is not None:
        pred_name2 = [str(int(x)) for x in pred_box_np2[:, 1]]

    if gt_box_np is not None:
        gt_name = [str(int(x)) for x in gt_box_np[:, 1]]

    if method == 'bev':
        canvas = canvas_bev.Canvas_BEV_heading_right(
            canvas_shape=((pc_range[4] - pc_range[1]) * 10, (pc_range[3] - pc_range[0]) * 10),
            canvas_x_range=(pc_range[0], pc_range[3]),
            canvas_y_range=(pc_range[1], pc_range[4]),
            left_hand=left_hand)
    elif method == '3d':
        canvas = canvas_3d.Canvas_3D(left_hand=left_hand)

    canvas_xy, valid_mask = canvas.get_canvas_coords(pcd_np)  # Get Canvas Coords
    canvas.draw_canvas_points(canvas_xy[valid_mask])  # Only draw valid points
    if gt_box_np is not None:
        # gt_corners = box_xywh_2_xyxy(gt_box_np[:,2:])
        gt_corners = boxes_to_corners_3d(gt_box_np[:, 2:], order='lwh')
        canvas.draw_boxes(gt_corners, colors=(0, 255, 0), texts=gt_name)
    if pred_box_np1 is not None:
        # pred_corners = box_xywh_2_xyxy(pred_box_np[:,2:])
        pred_corners = boxes_to_corners_3d(pred_box_np1[:, 2:], order='lwh')
        # colors = np.concatenate([color_dict[box_id].reshape(1,3) for box_id in pred_box_np[:,1]], axis=0)
        # canvas.draw_boxes(pred_corners, colors=colors, texts=pred_name)
        canvas.draw_boxes(pred_corners, colors=(0, 0, 255), texts=pred_name1)
        # canvas.draw_boxes(pred_corners, colors=(255, 0, 0), texts=pred_name)
    if pred_box_np2 is not None:
        # pred_corners = box_xywh_2_xyxy(pred_box_np[:,2:])
        pred_corners = boxes_to_corners_3d(pred_box_np2[:, 2:], order='lwh')
        # colors = np.concatenate([color_dict[box_id].reshape(1,3) for box_id in pred_box_np[:,1]], axis=0)
        # canvas.draw_boxes(pred_corners, colors=colors, texts=pred_name)
        # canvas.draw_boxes(pred_corners, colors=(0, 0, 255), texts=pred_name1)
        canvas.draw_boxes(pred_corners, colors=(255, 0, 0), texts=pred_name2)

    plt.axis("off")

    plt.imshow(canvas.canvas)
    plt.tight_layout()
    plt.savefig(save_path, transparent=False, dpi=500)
    plt.clf()
    plt.close()


def save_gif(image_paths, save_path):
    img_array = []
    for image_path in image_paths:
        if os.path.exists(image_path):
            image = imageio.v2.imread(image_path)
            img_array.append(image)

    imageio.mimsave(save_path, [x for x in img_array], 'GIF', duration=0.15)


def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description="Visualization demo")
    parser.add_argument(
        "--det_logs_path", default='', type=str, help="Det logs path (to get the tracking input)"
    )
    parser.add_argument(
        "--det_logs_path_s", default='', type=str, help="Det logs path (to get the tracking input)"
    )
    args = parser.parse_args()
    return args


def read_sequence(seq_dir):
    with open(seq_dir, 'r') as f:
        seqs = f.readlines()

    seq_dict = OrderedDict()
    for seq in seqs:
        seq = seq.strip('\n').split(',')  # [frame_id, obj_id, x, y, w, h, -1, -1, -1, -1]
        seq = [float(x) for x in seq]
        frame_id = seq[0]
        if frame_id in seq_dict:
            seq_dict[frame_id].append(np.array(seq)[None, ...])
        else:
            seq_dict[frame_id] = [np.array(seq)[None, ...]]

    updated_seq_dict = OrderedDict()
    for frame_id, seq in seq_dict.items():
        if len(seq) > 1:
            seq = np.concatenate(seq, axis=0)
            updated_seq_dict[frame_id] = seq
        else:
            updated_seq_dict[frame_id] = seq[0]
    return updated_seq_dict


if __name__ == "__main__":
    # 第一个参数红色 二个参数蓝色
    args = parse_args()
    root_dir = '/remote-home/share/xhpang/OpenCOODv2/'
    # args.det_logs_path = 'opencood/logs/opv2v_centerpoint_where2comm_multisweep_withshrinkhead_2023_02_18_15_19_52'

    pcd_folder = os.path.join(root_dir, args.det_logs_path, 'npy')
    gt_folder = os.path.join(root_dir, args.det_logs_path, 'track_3D', 'gt', 'OPV2V-test')
    det_folder = os.path.join(root_dir, args.det_logs_path, 'track_3D', 'data')
    det_folder_s = os.path.join(root_dir, args.det_logs_path_s, 'track_3D', 'data')

    image_save_path_root = os.path.join(root_dir, args.det_logs_path, 'track_3D', 'vis_images')
    if not os.path.exists(image_save_path_root):
        os.makedirs(image_save_path_root)

    gif_save_path_root = os.path.join(root_dir, args.det_logs_path, 'track_3D', 'vis_gifs')
    if not os.path.exists(gif_save_path_root):
        os.makedirs(gif_save_path_root)

    # Load GT
    scenes = [x for x in os.listdir(gt_folder) if os.path.exists(os.path.join(gt_folder, x))]
    for scene in scenes:
        print('################# Processing scene {} ###############'.format(scene))
        gt_dir = os.path.join(gt_folder, scene, 'gt', 'gt.txt')
        det_dir = os.path.join(det_folder, '{}.txt'.format(scene))
        det_dir_s = os.path.join(det_folder_s, '{}.txt'.format(scene))
        gt_seq = read_sequence(gt_dir)
        det_seq = read_sequence(det_dir)
        det_seq_s = read_sequence(det_dir_s)

        vis_save_path = os.path.join(image_save_path_root, '{}'.format(scene))
        if not os.path.exists(vis_save_path):
            os.makedirs(vis_save_path)

        image_save_paths = []
        color_dict = OrderedDict()
        for frame in tqdm(det_seq):
            det_box = det_seq[frame]
            det_box_s = det_seq_s[frame]
            gt_box = gt_seq[frame]

            # pcd_np = pcd_utils.pcd_to_np(os.path.join(pcd_folder, scene, '%04d_pcd.npy' % int(frame)))
            pcd_np = np.load(os.path.join(pcd_folder, scene, '%04d_pcd.npy' % int(frame)))

            image_save_path = os.path.join(vis_save_path, 'bev_%04d.png' % int(frame))
            visualize(det_box_s, det_box, gt_box, pcd_np, image_save_path, method='bev', left_hand=False)
            image_save_paths.append(image_save_path)

        save_gif(image_save_paths, os.path.join(gif_save_path_root, '{}.gif'.format(scene)))