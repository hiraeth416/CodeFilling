'''
Author: yhu
Contact: phyllis1sjtu@outlook.com
LastEditTime: 2021-07-26 09:42:05
Description:
'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from math import inf
from typing_extensions import OrderedDict
from numpy.core.fromnumeric import put

import torch
import numpy as np
import torch.nn as nn

from .uav_base_loss import FocalLoss, RateDistortionLoss
# from ..models.networks.sub_modules.cmap.cmap_loss import PixelwiseRateDistortionLoss
from .uav_base_loss import RegL1Loss, RegLoss, NormRegL1Loss, RegWeightedL1Loss, ZFocalLoss
import time
import os
from torch import stack
import torch.nn as nn


def _sigmoid(x):
    y = torch.clamp(x.sigmoid_(), min=1e-4, max=1 - 1e-4)
    return y


def _reorganize_batch(batch):
    reorg_batch = OrderedDict()
    for k, info in batch.items():
        if 'meta' in k:
            reorg_batch[k] = info
        else:
            reorg_batch[k] = info.flatten(0, 1)
    return reorg_batch


def _gather_feat(feat, ind, mask=None):
    dim = feat.size(2)
    ind = ind.unsqueeze(2).expand(ind.size(0), ind.size(1), dim)
    feat = feat.gather(1, ind)
    if mask is not None:
        mask = mask.unsqueeze(2).expand_as(feat)
        feat = feat[mask]
        feat = feat.view(-1, dim)
    return feat


def _transpose_and_gather_feat(feat, ind):
    feat = feat.permute(0, 2, 3, 1).contiguous()
    feat = feat.view(feat.size(0), -1, feat.size(3))
    feat = _gather_feat(feat, ind)
    return feat


def rotation_2d_torch(points, angles):
    """rotation 2d points based on origin point clockwise when angle positive.

    Args:
        points (float array, shape=[N, point_size, 2]): points to be rotated.
        angles (float array, shape=[N]): rotation angle.

    Returns:
        float array: same shape as points
    """
    rot_sin = angles[:, 0]
    rot_cos = angles[:, 1]
    rot_mat_T = torch.stack(
        [stack([rot_cos, -rot_sin]),
         stack([rot_sin, rot_cos])])
    return torch.einsum('aij,jka->aik', (points, rot_mat_T))


def _reg_loss(regr, gt_regr, mask):
    ''' L1 regression loss
      Arguments:
        regr (batch x max_objects x dim)
        gt_regr (batch x max_objects x dim)
        mask (batch x max_objects)
    '''
    b, k, _, _ = regr.shape
    regr = regr.view(b, k, -1)
    gt_regr = gt_regr.view(b, k, -1)
    num = mask.float().sum()
    mask = mask.unsqueeze(2).expand_as(gt_regr).float()

    regr = regr * mask
    gt_regr = gt_regr * mask

    regr_loss = nn.functional.smooth_l1_loss(regr, gt_regr, size_average=False)
    regr_loss = regr_loss / (num + 1e-4)
    return regr_loss


class CtdetLoss(torch.nn.Module):
    def __init__(self, opt):
        super(CtdetLoss, self).__init__()
        self.crit = FocalLoss()
        self.crit_reg = RegL1Loss()
        self.crit_wh = self.crit_reg
        self.crit_z = ZFocalLoss()
        # self.crit_comp = PixelwiseRateDistortionLoss()
        self.opt = opt
        self.acc_z = []
        self.count = 0

    def _acc_z(self, output_z, z, ind, reg_mask):
        z_pred = _transpose_and_gather_feat(output_z, ind)
        z_pred_cls = z_pred.argmax(-1)
        z_cls = z.argmax(-1)
        correct = ((z_pred_cls == z_cls) * 1 * reg_mask).sum()
        amount = reg_mask.sum()
        return correct / (amount + 1e-6)

    def _angle_loss(self, output_angle, output_wh, output_reg, wh, reg, angle, reg_mask, ind):
        weights = torch.Tensor([-0.5, -0.5, -0.5, 0.5, 0.5, 0.5, 0.5, -0.5]).to(output_angle.device)
        weights = weights.reshape([4, 2])

        def _get_corners(wh, reg, angle):
            corners = wh.detach().unsqueeze(2).expand(-1, -1, 4, -1)
            cur_weights = weights.unsqueeze(0).expand(corners.shape[1], -1, -1).unsqueeze(0).expand(corners.shape[0],
                                                                                                    -1, -1, -1)
            corners = corners * cur_weights
            corners = corners + reg.detach().unsqueeze(2)  # (b, k, 4, 2)
            b, k, _ = angle.shape
            corners = corners.view(b * k, 4, 2)
            angle = angle.view(b * k, 2)
            corners = rotation_2d_torch(corners, angle)
            return corners.view(b, k, 4, 2).contiguous()

        pred_angle = _transpose_and_gather_feat(output_angle, ind)
        pred_wh = _transpose_and_gather_feat(output_wh, ind)
        pred_reg = _transpose_and_gather_feat(output_reg, ind)

        # get the corners
        pred_corners = _get_corners(pred_wh, pred_reg, pred_angle)
        gt_corners = _get_corners(wh, reg, angle)

        angle_loss = _reg_loss(pred_corners, gt_corners, reg_mask)
        return angle_loss

    def forward(self, outputs, ori_batch):
        opt = self.opt
        batch = _reorganize_batch(ori_batch)
        hm_loss, wh_loss, off_loss, angle_loss = 0, 0, 0, 0
        hm_loss_early, wh_loss_early, off_loss_early, angle_loss_early = 0, 0, 0, 0
        hm_loss_fused, wh_loss_fused, off_loss_fused, angle_loss_fused = 0, 0, 0, 0
        hm_loss_i, wh_loss_i, off_loss_i = 0, 0, 0
        z_loss = 0
        acc_z = 0
        comp_loss = 0
        comp_aux_loss = 0
        single_loss = {}
        if opt['message_mode'] == 'QualityMap':
            for i in range(opt['round']):
                single_loss['hm_single_r{}_loss'.format(i)] = 0
                single_loss['wh_single_r{}_loss'.format(i)] = 0
                single_loss['off_single_r{}_loss'.format(i)] = 0
                single_loss['angle_single_r{}_loss'.format(i)] = 0

        for s in range(1):  # opt.num_stacks
            output = outputs[s]
            if True:  # not opt.mse_loss
                output['hm'] = _sigmoid(output['hm'])
                if (opt['coord'] in ['Global', 'Joint']) and (opt['feat_mode'] == 'fused'):
                    output['hm_early'] = _sigmoid(output['hm_early'])
                    output['hm_fused'] = _sigmoid(output['hm_fused'])
                if opt['coord'] == 'Joint':
                    output['hm_i'] = _sigmoid(output['hm_i'])

                if 'QualityMap' in opt['message_mode']:
                    for i in range(opt['round']):
                        output['hm_single_r{}'.format(i)] = _sigmoid(output['hm_single_r{}'.format(i)])
                        single_loss['hm_single_r{}_loss'.format(i)] += self.crit(output['hm_single_r{}'.format(i)],
                                                                                 batch['hm']) / 1  # opt.num_stacks

            hm_loss += self.crit(output['hm'], batch['hm']) / 1  # opt.num_stacks

            # import ipdb; ipdb.set_trace()
            # if not os.path.exists('train_vis'):
            #     os.makedirs('train_vis')
            # cur_time = time.time()
            # c_b = output['hm'][0,0].detach().cpu().numpy()
            # c_b = (c_b / (c_b.max()-c_b.min()) * 255.0).astype('uint8')[:,:,None]
            # c_r = (batch['hm'][0,0]*255).detach().cpu().numpy().astype('uint8')[:,:,None]
            # c_g = np.zeros_like(c_b)
            # heatmap = np.concatenate([c_b, c_g, c_r], axis=-1)
            # cv2.imwrite('train_vis/{}_pred_hm.png'.format(int(cur_time)), heatmap)

            # save_img = (batch['input'][0].detach().cpu().numpy()*255).astype('uint8').transpose(1,2,0)
            # cv2.imwrite('train_vis/{}_ori.png'.format(int(cur_time)), save_img)
            # worldgrid2worldcoord_mat = np.array([[500/800.0, 0, -200], [0, 500/448.0, -250], [0, 0, 1]])
            # cur_trans_mats = np.linalg.inv(batch['trans_mats'][0].detach().cpu().numpy() @ worldgrid2worldcoord_mat)
            # data = kornia.image_to_tensor(save_img, keepdim=False)
            # data_warp = kornia.warp_perspective(data.float(),
            #                                     torch.tensor(cur_trans_mats).repeat([1, 1, 1]).float(),
            #                                     dsize=(448, 800))
            # # convert back to numpy
            # img_warp = kornia.tensor_to_image(data_warp.byte())
            # img_warp = cv2.resize(img_warp, dsize=(400, 224))
            # img_warp = cv2.addWeighted(heatmap, 0.5, img_warp, 1-0.5, 0)
            # cv2.imwrite('train_vis/{}_ori_warp.png'.format(int(cur_time)), img_warp)

            if opt['wh_weight'] > 0:
                wh_loss += self.crit_reg(
                    output['wh'], batch['reg_mask'],
                    batch['ind'], batch['wh']) / 1  # num_stacks
                if opt['message_mode'] == 'QualityMap':
                    for i in range(opt['round']):
                        single_loss['wh_single_r{}_loss'.format(i)] += self.crit_reg(
                            output['wh_single_r{}'.format(i)], batch['reg_mask'],
                            batch['ind'], batch['wh'])
                if (opt['coord'] in ['Global', 'Joint']) and (opt['feat_mode'] == 'fused'):
                    wh_loss_early += self.crit_reg(
                        output['wh_early'], batch['reg_mask'],
                        batch['ind'], batch['wh'])
                    wh_loss_fused += self.crit_reg(
                        output['wh_fused'], batch['reg_mask'],
                        batch['ind'], batch['wh'])
                if opt['coord'] == 'Joint':
                    wh_loss_i += self.crit_reg(
                        output['wh_i'], batch['reg_mask_i'],
                        batch['ind_i'], batch['wh_i'])
                if ('z' in output) and (opt['coord'] in ['Global', 'Joint']):
                    z_loss += self.crit_z(
                        output['z'], batch['reg_mask'],
                        batch['ind'], batch['cat_depth'])
                    acc_z += self._acc_z(output['z'], batch['cat_depth'], batch['ind'], batch['reg_mask'])

            if opt['polygon'] is True and (opt['angle_weight'] > 0):
                if opt['coord'] in ['Global', 'Joint']:
                    angle_loss += self.crit_reg(
                        output['angle'], batch['reg_mask'],
                        batch['ind'], batch['angle'])
                    if opt['message_mode'] == 'QualityMap':
                        for i in range(opt['round']):
                            single_loss['angle_single_r{}_loss'.format(i)] += self.crit_reg(
                                output['angle_single_r{}'.format(i)], batch['reg_mask'],
                                batch['ind'], batch['angle'])
                    # angle_loss += self._angle_loss(output['angle'], output['wh'], output['reg'],
                    #                                  batch['wh'], batch['reg'], batch['angle'], batch['reg_mask'], batch['ind'])
                    if opt['feat_mode'] == 'fused':
                        angle_loss_early += self.crit_reg(
                            output['angle_early'], batch['reg_mask'],
                            batch['ind'], batch['angle'])
                        angle_loss_fused += self.crit_reg(
                            output['angle_fused'], batch['reg_mask'],
                            batch['ind'], batch['angle'])

            if opt['reg_offset'] is True and opt['off_weight'] > 0:
                off_loss += self.crit_reg(output['reg'], batch['reg_mask'],
                                          batch['ind'], batch['reg'])
                if opt['message_mode'] == 'QualityMap':
                    for i in range(opt['round']):
                        single_loss['off_single_r{}_loss'.format(i)] += self.crit_reg(
                            output['reg_single_r{}'.format(i)], batch['reg_mask'],
                            batch['ind'], batch['reg'])
                if (opt['coord'] in ['Global', 'Joint']) and (opt['feat_mode'] == 'fused'):
                    off_loss_early += self.crit_reg(output['reg_early'], batch['reg_mask'],
                                                    batch['ind'], batch['reg'])
                    off_loss_fused += self.crit_reg(output['reg_fused'], batch['reg_mask'],
                                                    batch['ind'], batch['reg'])
                if opt['coord'] == 'Joint':
                    off_loss_i += self.crit_reg(output['reg_i'], batch['reg_mask_i'],
                                                batch['ind_i'], batch['reg_i'])

        loss = opt['hm_weight'] * hm_loss + opt['wh_weight'] * wh_loss + opt['off_weight'] * off_loss
        loss_stats = {'loss': loss, 'hm_loss': hm_loss, 'wh_loss': wh_loss, 'off_loss': off_loss}
        if opt['polygon'] is True and (opt['angle_weight'] > 0):
            loss = loss + opt['angle_weight'] * angle_loss
            loss_stats.update({'angle_loss': angle_loss})
        if opt['feat_mode'] in ['fused']:
            loss = loss + opt.hm_weight * (hm_loss_early + hm_loss_fused) + \
                   opt.wh_weight * (wh_loss_early + wh_loss_fused) + \
                   opt.off_weight * (off_loss_early + off_loss_fused)
            loss_stats.update({'loss': loss,
                               'hm_loss_early': hm_loss_early, 'wh_loss_early': wh_loss_early,
                               'off_loss_early': off_loss_early,
                               'hm_loss_fused': hm_loss_fused, 'wh_loss_fused': wh_loss_fused,
                               'off_loss_fused': off_loss_fused})
            if opt.polygon and (opt.angle_weight > 0):
                loss = loss + opt.angle_weight * (angle_loss_early + angle_loss_fused)
                loss_stats.update({'angle_loss_early': angle_loss_early, 'angle_loss_fused': angle_loss_fused})
        if opt['coord'] == 'Joint':
            loss = loss + opt['hm_weight'] * hm_loss_i + \
                   opt['wh_weight'] * wh_loss_i + \
                   opt['off_weight'] * off_loss_i
            loss_stats.update({'loss': loss, 'hm_loss_i': hm_loss_i, 'wh_loss_i': wh_loss_i, 'off_loss_i': off_loss_i})
        if ('z' in output) and (opt['depth_mode'] == 'Weighted'):
            loss = loss + opt['wh_weight'] * z_loss
            loss_stats.update({'loss': loss, 'z_loss': z_loss, 'acc_z': acc_z})
            self.acc_z.append(acc_z)

        if opt['message_mode'] == 'QualityMap':
            if len(single_loss) > 0:
                for i in range(opt['round']):
                    loss = loss + opt['hm_weight'] * single_loss['hm_single_r{}_loss'.format(i)] \
                           + opt['wh_weight'] * single_loss['wh_single_r{}_loss'.format(i)] \
                           + opt['off_weight'] * single_loss['off_single_r{}_loss'.format(i)]
                    if opt['polygon'] and (opt['angle_weight'] > 0):
                        loss = loss + opt['angle_weight'] * single_loss['angle_single_r{}_loss'.format(i)]
                loss_stats.update(single_loss)

        if opt['train_mode'] in ['compressor', 'all']:
            if opt.train_mode == 'compressor':
                comp_loss = output['code_loss']
            else:
                comp_loss = output['logits_loss']

            #
            # comp_loss += output['comp_all_loss']['comp_loss']['loss']
            # comp_aux_loss += output['comp_all_loss']['comp_aux_loss']

            # if 'likelihoods' in output['comp_out']:
            #     comp_loss += self.crit_comp(output['comp_out'], output['comp_gt'])["loss"]
            #     comp_aux_loss += output['comp_aux_loss']
            # else:
            #     comp_loss += loss
            #     comp_aux_loss += loss
            loss_stats.update({
                'comp_loss': comp_loss,
                # 'comp_aux_loss': comp_aux_loss
            })
            return loss, comp_loss, loss_stats
            # return loss, comp_loss, comp_aux_loss, loss_stats
        return loss, loss, loss_stats
        # return loss, loss, loss_stats


class uavcenterpointloss(nn.Module):
    def __init__(self, opt):
        super(uavcenterpointloss, self).__init__()
        self.loss = CtdetLoss(opt)
        self.loss_dict = {}

    def forward(self, output_dict, target_dict):
        loss, loss, loss_stats = self.loss(output_dict, target_dict)
        self.loss_dict.update(loss_stats)
        return loss

    def logging(self, epoch, batch_id, batch_len, writer=None, suffix=""):
        """
        Print out  the loss function for current iteration.

        Parameters
        ----------
        epoch : int
            Current epoch for training.
        batch_id : int
            The current batch.
        batch_len : int
            Total batch length in one iteration of training,
        writer : SummaryWriter
            Used to visualize on tensorboard
        """
        total_loss = self.loss_dict.get('loss', 0)
        reg_loss = self.loss_dict.get('wh_loss', 0)
        cls_loss = self.loss_dict.get('hm_loss', 0)

        print("[epoch %d][%d/%d]%s, || Loss: %.4f || Conf Loss: %.4f"
              " || Loc Loss: %.4f" % (
                  epoch, batch_id + 1, batch_len, suffix,
                  total_loss, cls_loss, reg_loss))

        if not writer is None:
            writer.add_scalar('Regression_loss', reg_loss,
                              epoch * batch_len + batch_id)
            writer.add_scalar('Confidence_loss', cls_loss,
                              epoch * batch_len + batch_id)

