from PIL import Image, ImageDraw
import numpy as np


def draw_large_red_point(image_path, point, radius, save_path):
    # 打开图片
    img = Image.open(image_path)

    # 创建一个ImageDraw对象
    draw = ImageDraw.Draw(img)

    # 绘制红色实心圆
    red_color = (255, 255, 255)  # 红色，RGB值
    x, y = point
    left_top = (x - radius, y - radius)
    right_bottom = (x + radius, y + radius)
    draw.ellipse([left_top, right_bottom], fill=red_color)

    # 保存修改后的图片
    img.save(save_path)


def crop_image_with_points(image_path, points, save_path):
    # 打开图片
    img = Image.open(image_path)
    #import pdb; pdb.set_trace()

    # 将四个点转换为NumPy数组
    points = np.array(points, dtype=int)

    # 计算剪切区域的边界框
    left = min(points[:, 0])
    top = min(points[:, 1])
    right = max(points[:, 0])
    bottom = max(points[:, 1])

    # 剪切图片
    cropped_img = img.crop((left, top, right, bottom))

    # 保存剪切后的图片
    cropped_img.save(save_path)


def cccv(src_path, det_path, points=None, path='opencood/logs/vis_pcd/pipe/padded.png'):
    src = Image.open(src_path)
    det = Image.open(det_path)

    src = np.array(src)
    det = np.array(det)

    left = 810
    right = 870

    top = 220
    bottom = 280
    # import pdb;
    # pdb.set_trace()
    det[top:bottom, left:right, :] = src[top:bottom, left:right, :]
    det = Image.fromarray(det)
    det.save(path)


def load_box(prev_path, curr_path):
    prev_box = np.load(prev_path)
    curr_box = np.load(curr_path)  # shape (N, 8, 3)


#path = 'opencood/logs/vis_pcd/pipe/'
path = "/GPFS/rhome/sifeiliu/OpenCOODv2_new/opencood/logs_HEAL/ego_camera_other_lidar/codebook_size256_r1_segnum2/after_codebook/"
# fig_name_re = ['no_02.png', 'no_03.png',
#             'v2xvit_02.png', 'v2xvit_03.png',
#             's_02.png', 's_03.png',
#             'stc_02.png', 'stc_03.png']
# fig_name_re = ['single_det.png', 'input.png', 'final_det.png']
# fig_name_re = ['fused_feat_box.png', 'feat_single.png', 'feat_recon.png', 'feat_single_warped.png', 'fused_feat.png']
# fig_name_re = ['cmap.png']
# fig_name_re = ['prev_curr.png', 'prev_feat_box.png']
fig_name_re = ['i0_agent1.png']
# fig_name = ['no_1.png', 'no_2.png',
#             'v2xvit_0.png','v2xvit_1.png',
#             's_0.png','s_1.png',
#             'stc_0.png','stc_1.png']

# draw_large_red_point(path + 'no_02.png', (4000, 600), 100, path + 'draw_white_point.png')
# cropx = 0.3 *
point = [(1520, 90), (2490, 90), (2490, 700), (1520, 700)]
# point = [(1620, 140), (2490, 140), (2490, 680), (1620, 680)]
for fig in fig_name_re:
    crop_image_with_points(path + fig, point, path + 'crop_' + fig)
# src_path = path+'crop_fused_feat.png'
# det_path = path+'crop_feat_single_warped.png'
# cccv(src_path, det_path)
# crop_image_with_points(path + 'no_02.png', point, path + 'crop_no_02.png')
#
# opencood/visualization/crop.py
# for fig in fig_name:
#     img = Image.open(path + fig)
#     print(img.size)
