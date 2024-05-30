import torch
import torch.nn as nn
from .images import get_affine_transform, transform_preds
from .decode import ctdet_decode
import time
import cv2
import numpy as np


def ctdet_post_process(dets, c, s, h, w, num_classes):
    # dets: batch x max_dets x dim
    # return 1-based class det dict
    dims = dets.shape[-1]
    ret = []
    for i in range(dets.shape[0]):
        top_preds = {}
        dets[i, :, :2] = transform_preds(
            dets[i, :, 0:2], c[i], s[i], (w, h))
        dets[i, :, 2:4] = transform_preds(
            dets[i, :, 2:4], c[i], s[i], (w, h))
        if dims > 6:
            dets[i, :, 4:6] = transform_preds(
                dets[i, :, 4:6], c[i], s[i], (w, h))
            dets[i, :, 6:8] = transform_preds(
                dets[i, :, 6:8], c[i], s[i], (w, h))
        classes = dets[i, :, -1]
        for j in range(num_classes):
            inds = (classes == j)
            top_preds[j + 1] = np.concatenate([
                dets[i, inds, :(dims - 2)].astype(np.float32),
                dets[i, inds, (dims - 2):(dims - 1)].astype(np.float32)], axis=1).tolist()
        ret.append(top_preds)
    return ret


class UAVProcessing():
    def __init__(self, opt, model):
        self.opt = opt
        self.test_scales = '1'
        self.test_scales = [float(i) for i in self.test_scales.split(',')]
        self.scales = self.test_scales
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = model

    def pre_process(self, image, scale, meta=None):
        height, width = image.shape[0:2]
        new_height = int(height * scale)
        new_width = int(width * scale)
        if self.opt['fix_res'] is True:
            inp_height, inp_width = self.opt['input_h'], self.opt['input_w']
            c = np.array([new_width / 2., new_height / 2.], dtype=np.float32)
            s = max(height, width) * 1.0
        else:
            inp_height = (new_height | self.opt['pad']) + 1
            inp_width = (new_width | self.opt['pad']) + 1
            c = np.array([new_width // 2, new_height // 2], dtype=np.float32)
            s = np.array([inp_width, inp_height], dtype=np.float32)
        # print('ori: ', width, height)
        # print('new: ', new_width, new_height)
        # print('inp: ', inp_width, inp_height)
        # print('input: ', self.opt.input_h, self.opt.input_w)
        trans_input = get_affine_transform(c, s, 0, [inp_width, inp_height])
        resized_image = cv2.resize(image, (new_width, new_height))
        inp_image = cv2.warpAffine(
            resized_image, trans_input, (inp_width, inp_height),
            flags=cv2.INTER_LINEAR)
        # inp_image = ((inp_image / 255. - self.mean) / self.std).astype(np.float32)
        inp_image = (inp_image / 255.).astype(np.float32)

        images = inp_image.transpose(2, 0, 1).reshape(1, 3, inp_height, inp_width)
        if self.opt['flip_test'] is True:
            images = np.concatenate((images, images[:, :, :, ::-1]), axis=0)
        images = torch.from_numpy(images)

        meta_i = {'c': c, 's': s,
                  'out_height': inp_height // 4,
                  'out_width': inp_width // 4}

        feat_h, feat_w = self.opt['feat_shape']
        c = np.array([feat_w / (2 * self.opt['map_scale']), feat_h / (2 * self.opt['map_scale'])])
        s = np.array([feat_w / (self.opt['map_scale']), feat_h / (self.opt['map_scale'])])
        meta = {'c': c, 's': s,
                'out_height': feat_h / (self.opt['map_scale']),
                'out_width': feat_w / (self.opt['map_scale'])}

        if self.opt['coord'] == 'Local':
            return images, meta_i
        elif self.opt['coord'] == 'Global':
            return images, meta
        elif self.opt['coord'] == 'Joint':
            return images, [meta, meta_i]

    def process(self, images, trans_mats, shift_mats, return_time=False):
        with torch.no_grad():
            data_dict = {'input': images, 'shift_mats_1': shift_mats[0], 'shift_mats_2': shift_mats[1], \
                         'shift_mats_4': shift_mats[2], 'shift_mats_8': shift_mats[3], \
                         'trans_mats': trans_mats[0]}
            output = self.model(data_dict)[-1]
            hm = output['hm'].sigmoid_()
            wh = output['wh']
            reg = output['reg'] if self.opt['reg_offset'] else None
            angle = output['angle'] if self.opt['polygon'] is True else None  # (sin, cos)
            z = output['z'] if 'z' in output else None
            # import ipdb; ipdb.set_trace()
            if self.opt['coord'] == 'Joint':
                hm_i = output['hm_i'].sigmoid_()
                wh_i = output['wh_i']
                reg_i = output['reg_i'] if self.opt['reg_offset'] is True else None

            if self.opt['flip_test']:
                hm = (hm[0:1] + flip_tensor(hm[1:2])) / 2
                wh = (wh[0:1] + flip_tensor(wh[1:2])) / 2
                reg = reg[0:1] if reg is not None else None
                if self.opt['coord'] == 'Joint':
                    hm_i = (hm_i[0:1] + flip_tensor(hm_i[1:2])) / 2
                    wh_i = (wh_i[0:1] + flip_tensor(wh_i[1:2])) / 2
                    reg_i = reg_i[0:1] if reg_i is not None else None
            torch.cuda.synchronize()
            forward_time = time.time()
            if self.opt['coord'] == 'Local':
                dets = ctdet_decode(hm, wh, map_scale=None, shift_mats=None, reg=reg, angle=None,
                                    cat_spec_wh=self.opt['cat_spec_wh'], K=100)
            elif self.opt['coord'] == 'Global':
                dets = ctdet_decode(hm, wh, map_scale=self.opt['map_scale'], shift_mats=shift_mats[0], reg=reg,
                                    angle=angle, cat_spec_wh=self.opt['cat_spec_wh'], K=100)
            else:
                dets_bev = ctdet_decode(hm, wh, map_scale=self.opt['map_scale'], shift_mats=shift_mats[0], reg=reg,
                                        angle=angle, cat_spec_wh=self.opt['cat_spec_wh'], K=100)
                dets_uav = ctdet_decode(hm_i, wh_i, map_scale=None, shift_mats=None, reg=reg_i, angle=None,
                                        cat_spec_wh=self.opt['cat_spec_wh'], K=100)
                dets = [dets_bev, dets_uav]
        if return_time:
            return output, dets, forward_time
        else:
            return output, dets

    def post_process(self, dets, meta, scale=1):
        dets = dets.detach().cpu().numpy()
        dets = dets.reshape(1, -1, dets.shape[2])
        dets = ctdet_post_process(
            dets.copy(), [meta['c']], [meta['s']],
            meta['out_height'], meta['out_width'], self.opt['num_classes'])
        for j in range(1, self.opt['num_classes'] + 1):
            dets[0][j] = np.array(dets[0][j], dtype=np.float32)
            dets[0][j] = dets[0][j].reshape(-1, dets[0][j].shape[-1])
            dets[0][j][:, :(dets[0][j].shape[-1] - 1)] /= scale
        return dets[0]

    def merge_outputs(self, detections):
        results = {}
        for j in range(1, self.opt['num_classes'] + 1):
            results[j] = np.concatenate(
                [detection[j] for detection in detections], axis=0).astype(np.float32)
            if len(self.scales) > 1 or self.opt['nms'] is True:
                if results[j].shape[-1] > 6:
                    polygon_nms(results[j], 0.5)
                else:
                    soft_nms(results[j], Nt=0.5, method=2)
        scores = np.hstack(
            [results[j][:, -1] for j in range(1, self.opt['num_classes'] + 1)])
        if len(scores) > self.opt['max_per_image']:
            kth = len(scores) - self.opt['max_per_image']
            thresh = np.partition(scores, kth)[kth]
            for j in range(1, self.opt['num_classes'] + 1):
                keep_inds = (results[j][:, -1] >= thresh)
                results[j] = results[j][keep_inds]
        return results

    def run(self, image_or_path_or_tensor, img_idx=None, meta=None):
        load_time, pre_time, net_time, dec_time, post_time = 0, 0, 0, 0, 0
        merge_time, tot_time = 0, 0

        start_time = time.time()
        pre_processed = False
        if isinstance(image_or_path_or_tensor, np.ndarray):
            image = image_or_path_or_tensor
        elif type(image_or_path_or_tensor) == type(''):
            image = cv2.imread(image_or_path_or_tensor)
        else:
            image = image_or_path_or_tensor['image']
            pre_processed_images = image_or_path_or_tensor
            pre_processed = True

        loaded_time = time.time()
        load_time += (loaded_time - start_time)

        detections = []
        results = []
        comm_rates = []
        for scale in self.scales:
            scale_start_time = time.time()
            if not pre_processed:
                images, meta = self.pre_process(image, scale, meta)
            else:
                images = pre_processed_images['images'][scale]
                meta = pre_processed_images['meta'][scale]
                if isinstance(meta, list):
                    updated_meta = []
                    for cur_meta in meta:
                        updated_meta.append({k: v.numpy()[0] for k, v in cur_meta.items()})
                    meta = updated_meta
                else:
                    meta = {k: v.numpy()[0] for k, v in meta.items()}
                trans_mats = [pre_processed_images['trans_mats'], pre_processed_images['trans_mats_n010'], \
                              pre_processed_images['trans_mats_n005'], pre_processed_images['trans_mats_p005'], \
                              pre_processed_images['trans_mats_p007'], pre_processed_images['trans_mats_p010'], \
                              pre_processed_images['trans_mats_p015'], pre_processed_images['trans_mats_p020'], \
                              pre_processed_images['trans_mats_p080'], pre_processed_images['trans_mats_withnoise']]
                shift_mats = [pre_processed_images['shift_mats_1'], pre_processed_images['shift_mats_2'], \
                              pre_processed_images['shift_mats_4'], pre_processed_images['shift_mats_8'], \
                              pre_processed_images['shift_mats_withnoise']]
            images = images.to(self.device)
            trans_mats = [x.to(self.device) for x in trans_mats]
            shift_mats = [x.to(self.device) for x in shift_mats]
            torch.cuda.synchronize()
            pre_process_time = time.time()
            pre_time += pre_process_time - scale_start_time
            output, dets, forward_time = self.process(images, trans_mats, shift_mats, return_time=True)
            output['code_loss'] = 0
            # comm_rates.append(output['comm_rate'].item())
            comm_rates = [0]
            quant_error = output['code_loss']
            # code_len = output['code_len']
            ############## to be fixed!
            # quant_error = 0
            code_len = 0

            torch.cuda.synchronize()
            net_time += forward_time - pre_process_time
            decode_time = time.time()
            dec_time += decode_time - forward_time

            if isinstance(dets, list):
                for cur_dets, cur_meta in zip(dets, meta):
                    cur_detections = []
                    cur_results = []
                    for i in range(len(cur_dets)):
                        cur_detections.append(self.post_process(cur_dets[i:i + 1], cur_meta, scale))
                        cur_results.append(self.merge_outputs([cur_detections[-1]]))
                    detections.append(cur_detections)
                    results.append(cur_results)
            else:
                for i in range(len(dets)):
                    detections.append(self.post_process(dets[i:i + 1], meta, scale))
                    results.append(self.merge_outputs([detections[-1]]))
            torch.cuda.synchronize()
            post_process_time = time.time()
            post_time += post_process_time - decode_time

        torch.cuda.synchronize()
        end_time = time.time()
        merge_time += end_time - post_process_time
        tot_time += end_time - start_time

        return {'results': results, 'tot': tot_time, 'load': load_time,
                'pre': pre_time, 'net': net_time, 'dec': dec_time,
                'post': post_time, 'merge': merge_time, 'comm_rate': sum(comm_rates) / len(comm_rates),
                'quant_error': quant_error, 'code_len': code_len}
