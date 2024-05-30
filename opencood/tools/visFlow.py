import numpy as np
import matplotlib.pyplot as plt
import torch
import copy


def vis_flow(flow_map, path='/remote-home/share/xhpang/OpenCOODv2/opencood/logs/kalman/'):
    N, _, H, W = flow_map.shape
    fig, ax = plt.subplots()

    flow_map[..., 0] *= W / 2
    flow_map[..., 1] *= H / 2

    x = np.arange(W)
    y = np.arange(H)
    xv, yv = np.meshgrid(x, y)
    grid = np.stack((xv, yv))
    grid = np.expand_dims(grid, axis=0)

    flow_map = flow_map - grid

    for n in range(N):
        for i in range(H):
            for j in range(W):
                # 取出当前位置的x和y方向的flow分量
                dx = flow_map[n, 0, i, j]
                dy = flow_map[n, 1, i, j]

                if dx == 0 and dy == 0:
                    continue

                # 计算当前位置的起点和终点坐标
                x1 = j
                y1 = i

                # 绘制箭头
                ax.arrow(x1, y1, dx, dy, head_width=0.3, head_length=0.3, fc='k', ec='k')

    # 显示图像
    # plt.show()

    # save figure
    plt.savefig(path + 'flow.png', dpi=300)
    plt.close()


# visualize flow

# # 从 flow 中提取 x 和 y 分量
# H, W = pre.shape[2:]
# N = 1
# flow = pre
#
# # 绘制flow
# fig, ax = plt.subplots()
#
# for n in range(N):
#     for i in range(H):
#         for j in range(W):
#             # 取出当前位置的x和y方向的flow分量
#             dx = flow[n, 0, i, j]
#             dy = flow[n, 1, i, j]
#
#             if dx == 0 and dy == 0:
#                 continue
#
#             # 计算当前位置的起点和终点坐标
#             x1 = j
#             y1 = i
#             x2 = j + dx
#             y2 = i + dy
#
#             # 绘制箭头
#             ax.arrow(x1, y1, dx, dy, head_width=0.5, head_length=0.5, fc='k', ec='k')
#
# # 显示图像
# plt.show()
#
# # save figure
# fig.savefig("flow0429.png")

import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import cv2
import os

def vis_feat(feat, path='/remote-home/share/xhpang/OpenCOODv2/opencood/logs/kalman/', name=''):
    feat = torch.sum(feat, dim=1)[0]  # shape: (H, W)
    sns.heatmap(feat.cpu().numpy(), cmap='coolwarm')
    plt.savefig(path + name + 'feat.png')
    plt.close()


def vis_flow_sns(flow_map, path='/remote-home/share/yhu/Co_Flow/opencood/visualization/flow/'):
    flow_x = flow_map[:, 0, :, :]
    flow_y = flow_map[:, 1, :, :]
    sns.heatmap(flow_x.mean(axis=0), cmap='coolwarm')
    plt.title('x')
    plt.savefig(path + 'x.png')
    sns.heatmap(flow_y.mean(axis=0), cmap='coolwarm')
    plt.title('y')
    plt.savefig(path + 'y.png')
    plt.close()

def vis_flow_box(flow_mask, prev_boxes, curr_boxes, count=0, path='/remote-home/share/yhu/Co_Flow/opencood/visualization/flow_my/'):
    def coord2grid(boxes, scale=1):
        # boxes = np.array(boxes[:,:4,:2].cpu().numpy(), dtype=np.int32)
        boxes = boxes[:,:4,:2].cpu().numpy() * scale
        boxes[:,:,0] += W//2 * scale
        boxes[:,:,1] += H//2 * scale
        boxes = np.array(boxes, dtype=np.int32)
        return boxes

    def vis_box(image, boxes, color=(0,255,0)):
        for box in boxes:
            idx_draw_pairs = [(0, 1), (1, 2), (2, 3), (3, 0)]
            for start, end in idx_draw_pairs:
                cv2.line(image, tuple(box[start].tolist()), tuple(box[end].tolist()),color,thickness=2)
        return image
    
    if len(flow_mask.shape) > 3:
        flow_mask = flow_mask[0]
    image = flow_mask.max(dim=0)[0].cpu().numpy() * 255
    H, W = image.shape[-2:]
    scale = 4
    image = cv2.resize(image, (W*scale, H*scale), interpolation = cv2.INTER_AREA)
    image = image[...,None].repeat(3,axis=-1)
    
    prev_boxes = coord2grid(prev_boxes, scale)
    curr_boxes = coord2grid(curr_boxes, scale)
    
    prev_image = vis_box(copy.deepcopy(image), prev_boxes, (0,255,0))
    curr_image = vis_box(copy.deepcopy(prev_image), curr_boxes, (0,0,255))
    
    # cv2.imwrite(os.path.join(path, '{}_prev.png'.format(count)), prev_image)
    cv2.imwrite(os.path.join(path, '{}_curr.png'.format(count)), curr_image)


"""

pre_path = "/dssg/home/acct-seecsh/seecsh/xhpang/opencood/logs/flow_iter_01_wo_dcn/flow/pre.npy"
gt_path = "/dssg/home/acct-seecsh/seecsh/xhpang/opencood/logs/flow_iter_01_wo_dcn/flow/gt.npy"

pre_path = "/dssg/home/acct-seecsh/seecsh/xhpang/opencood/tools/flow_vis/pre.npy"
gt_path = "/dssg/home/acct-seecsh/seecsh/xhpang/opencood/tools/flow_vis/gt.npy"

# load data
pre = np.load(pre_path)[0][np.newaxis, ...]
gt = np.load(gt_path)[0][np.newaxis, ...]

# 创建一个形状为(N, 2, H, W)的示例数据
H, W = pre.shape[2:]
N = 1
flow = pre

# 将流场沿着通道的方向拆分为x和y方向的张量
flow_x = flow[:, 0, :, :]
flow_y = flow[:, 1, :, :]

# 绘制x方向的热力图
sns.heatmap(flow_x.mean(axis=0), cmap='coolwarm')
plt.title('predict x')
plt.savefig('/dssg/home/acct-seecsh/seecsh/xhpang/opencood/tools/flow_vis/prex.png')

# 绘制y方向的热力图
sns.heatmap(flow_y.mean(axis=0), cmap='coolwarm')
plt.title('predict y')
plt.savefig('/dssg/home/acct-seecsh/seecsh/xhpang/opencood/tools/flow_vis/prey.png')
"""
