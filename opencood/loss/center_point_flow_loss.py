import torch
from torch import nn
import numpy as np
from opencood.loss.center_point_loss import CenterPointLoss
from icecream import ic


class CenterPointFlowLoss(CenterPointLoss):
    def __init__(self, args):
        super(CenterPointFlowLoss, self).__init__(args)
        self.flow_loss = nn.SmoothL1Loss(reduction='none')
        self.state_loss = nn.BCEWithLogitsLoss()
        self.flow_weight = args['flow_weight']
        self.future_seq_len = 1
        if 'future_seq_len' in args.keys():
            self.future_seq_len = args['future_seq_len']

        self.detection_fix = False
        if 'detection_fix' in args.keys() and args['detection_fix']:
            self.detection_fix = True

        self.flow_unc_flag = False
        if 'flow_unc_flag' in args.keys() and args['flow_unc_flag']:
            self.flow_unc_flag = True
            self.unc_weight = 0.1
            self.loss_unc = KLLoss()

    def forward(self, output_dict, target_dict, suffix=""):
        """
        Parameters
        ----------
        output_dict : dict
        target_dict : dict
        """
        if self.detection_fix:
            total_loss = 0
        else:
            total_loss = super().forward(output_dict, target_dict, suffix)
        ###### flow loss ######
        if f'flow_preds{suffix}' in output_dict and output_dict[f'flow_preds{suffix}'].shape[0] != 0:
            h, w = output_dict[f'flow_preds{suffix}'].shape[-2], output_dict[f'flow_preds{suffix}'].shape[-1]
            output_dict[f'flow_preds{suffix}'] = output_dict[f'flow_preds{suffix}'].squeeze()
            target_dict[f'flow_gt{suffix}'] = target_dict[f'flow_gt{suffix}'].squeeze()

            flow_loss = self.flow_loss(output_dict[f'flow_preds{suffix}'], target_dict[f'flow_gt{suffix}'])
            # n, c, h, w = target_dict[f'flow_gt{suffix}'].shape
            valid_flow_mask = ((target_dict[f'flow_gt{suffix}'].max(dim=2)[0] != 0) * 1.0).unsqueeze(2)

            flow_loss = (flow_loss * valid_flow_mask).sum() / (valid_flow_mask.sum() + 1e-6)
            flow_loss *= self.flow_weight

            # state_loss = self.state_loss(output_dict[f'state_preds{suffix}'],
            #                              valid_flow_mask)

            # total_loss += flow_loss + state_loss
            total_loss += flow_loss
            self.loss_dict.update({'total_loss': total_loss.item(),
                                   'flow_loss': flow_loss.item(),
                                   # 'state_loss': state_loss.item()
                                   })

            if self.flow_unc_flag:
                unc_preds = output_dict['flow_unc_preds{}'.format(suffix)]
                flow_valid_preds = output_dict[f'flow_preds{suffix}'] * valid_flow_mask
                flow_valid_gt = target_dict[f'flow_gt{suffix}'] * valid_flow_mask
                unc_loss = self.get_unc_layer_loss(flow_valid_preds, flow_valid_gt, unc_preds) / (
                        valid_flow_mask.sum() + 1e-6)

                self.loss_dict.update({'unc_loss': unc_loss.item()})
                total_loss += self.unc_weight * unc_loss

        return total_loss

    def get_unc_layer_loss(self, pred_flow, gt_flow, unc_pred, reg_weights=None):

        b, c, h, w = gt_flow.shape
        gt_flow = gt_flow.permute(0, 2, 3, 1).contiguous().view(b, -1, c)
        pred_flow = pred_flow.permute(0, 2, 3, 1).contiguous().view(b, -1, c)
        unc_pred = unc_pred.permute(0, 2, 3, 1).contiguous().view(b, -1, c)  # [n, h*w, 2 ]

        unc_loss = self.loss_unc(pred_flow,
                                 gt_flow,
                                 unc_pred,
                                 reg_weights)
        return unc_loss

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
        total_loss = self.loss_dict.get('total_loss', 0)
        reg_loss = self.loss_dict.get('reg_loss', 0)
        cls_loss = self.loss_dict.get('cls_loss', 0)
        flow_loss = self.loss_dict.get('flow_loss', 0)
        state_loss = self.loss_dict.get('state_loss', 0)
        unc_loss = self.loss_dict.get('unc_loss', 0)

        print("[epoch %d][%d/%d]%s, || Loss: %.4f || Conf Loss: %.4f"
              " || Loc Loss: %.4f || Flow Loss: %.4f || State Loss: %.4f || Unc Loss: %.4f" % (
                  epoch, batch_id + 1, batch_len, suffix,
                  total_loss, cls_loss, reg_loss, flow_loss, state_loss, unc_loss))

        if not writer is None:
            writer.add_scalar('Regression_loss', reg_loss,
                              epoch * batch_len + batch_id)
            writer.add_scalar('Confidence_loss', cls_loss,
                              epoch * batch_len + batch_id)


class KLLoss(nn.Module):
    def __init__(self):
        super(KLLoss, self).__init__()

        self.xy_loss = self.kl_loss_l2

    @staticmethod
    def kl_loss_l2(diff, s):
        """
        Args:
            diff: [B, 2]
            s:    [B, 2]
        Returns:
            loss: [B, 2]
        """
        loss = 0.5 * (torch.exp(-s) * (diff ** 2) + s)
        loss = torch.sum(loss)

        return loss

    def forward(self, input: torch.Tensor,
                target: torch.Tensor,
                sm: torch.Tensor,
                weights: torch.Tensor = None):
        target = torch.where(torch.isnan(target), input,
                             target)  # ignore nan targets
        xy_diff = input[..., :2] - target[..., :2]
        # print("xy_diff", xy_diff)
        # scatter_diffx=xy_diff[...,0]
        # scatter_pred=sm[...,0]
        # scatter_data=torch.vstack([scatter_diffx,scatter_pred])

        loss = self.xy_loss(xy_diff, sm[..., :2])

        if weights is not None:
            assert weights.shape[0] == loss.shape[0] and weights.shape[1] == loss.shape[1]
            loss = loss * weights

        # return loss,scatter_data
        return loss