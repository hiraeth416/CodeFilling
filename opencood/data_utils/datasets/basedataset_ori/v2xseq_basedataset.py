import os
from collections import OrderedDict
import cv2
import h5py
import torch
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
import random
import opencood.utils.pcd_utils as pcd_utils
from opencood.data_utils.augmentor.data_augmentor import DataAugmentor
from opencood.hypes_yaml.yaml_utils import load_yaml
from opencood.utils.pcd_utils import downsample_lidar_minimum
from opencood.utils.camera_utils import load_camera_data, load_intrinsic_DAIR_V2X
from opencood.utils.common_utils import read_json
from opencood.utils.transformation_utils import tfm_to_pose, rot_and_trans_to_trasnformation_matrix
from opencood.utils.transformation_utils import veh_side_rot_and_trans_to_trasnformation_matrix
from opencood.utils.transformation_utils import inf_side_rot_and_trans_to_trasnformation_matrix
from opencood.data_utils.pre_processor import build_preprocessor
from opencood.data_utils.post_processor import build_postprocessor


class V2XSEQBaseDataset(Dataset):
    def __init__(self, params, visualize, train=True):
        self.params = params
        self.visualize = visualize
        self.train = train

        self.pre_processor = build_preprocessor(params["preprocess"], train)
        self.post_processor = build_postprocessor(params["postprocess"], train)
        self.post_processor.generate_gt_bbx = self.post_processor.generate_gt_bbx_by_iou
        self.data_augmentor = DataAugmentor(params['data_augment'],
                                            train)

        if 'clip_pc' in params['fusion']['args'] and params['fusion']['args']['clip_pc']:
            self.clip_pc = True
        else:
            self.clip_pc = False

        if 'train_params' not in params or 'max_cav' not in params['train_params']:
            self.max_cav = 2
        else:
            self.max_cav = params['train_params']['max_cav']

        self.load_lidar_file = True if 'lidar' in params['input_source'] or self.visualize else False
        self.load_camera_file = True if 'camera' in params['input_source'] else False
        self.load_depth_file = True if 'depth' in params['input_source'] else False

        assert self.load_depth_file is False

        self.label_type = params['label_type']  # 'lidar' or 'camera'
        self.generate_object_center = self.generate_object_center_lidar if self.label_type == "lidar" \
            else self.generate_object_center_camera

        if self.load_camera_file:
            self.data_aug_conf = params["fusion"]["args"]["data_aug_conf"]

        # split dir存储sample id
        if self.train:
            split_dir = params['root_dir']
        else:
            split_dir = params['validate_dir']

        self.root_dir = params['data_dir']

        self.split_info = read_json(split_dir)

        self.co_datainfo = read_json(os.path.join(self.root_dir, 'cooperative/data_info.json'))
        self.co_data = OrderedDict()
        for frame_info in self.co_datainfo:
            veh_frame_id = frame_info['vehicle_frame']
            self.co_data[veh_frame_id] = frame_info

        if "noise_setting" not in self.params:
            self.params['noise_setting'] = OrderedDict()
            self.params['noise_setting']['add_noise'] = False

        self.len_record = self.get_len_record(self.train)

    def reinitialize(self):
        pass

    def get_len_record(self, train):
        # train, val, test size = 6457, 2152, 2152
        train_idx = [0, 6457]
        val_idx = [6457, 8609]
        test_idx = [8609, 10761]

        if train:
            data_idx = train_idx
        else:
            data_idx = val_idx

        scene_idx = []
        len_record = []
        frame_count = 1
        for i in range(data_idx[0], data_idx[1]):
            if self.co_datainfo[i]['vehicle_sequence'] not in scene_idx:
                if len(scene_idx) != 0:
                    scene_idx.append(self.co_datainfo[i]['vehicle_sequence'])
                    len_record.append(frame_count)
                    frame_count = 1
                else:
                    scene_idx.append(self.co_datainfo[i]['vehicle_sequence'])
                    frame_count = 1
            else:
                frame_count += 1

        return len_record

    def retrieve_base_data(self, idx):
        """
        Given the index, return the corresponding data.
        NOTICE!
        It is different from Intermediate Fusion and Early Fusion
        Label is not cooperative and loaded for both veh side and inf side.
        Parameters
        ----------
        idx : int
            Index given by dataloader.
        Returns
        -------
        data : dict
            The dictionary contains loaded yaml params and lidar data for
            each cav.
        """
        veh_frame_id = self.split_info[idx]
        frame_info = self.co_data[veh_frame_id]
        inf_frame_id = frame_info['infrastructure_frame']
        system_error_offset = frame_info["system_error_offset"]
        #############
        data = OrderedDict()

        data[0] = OrderedDict()
        data[0]['ego'] = True
        data[1] = OrderedDict()
        data[1]['ego'] = False

        data[0]['params'] = OrderedDict()
        data[1]['params'] = OrderedDict()

        # pose of agent 
        lidar_to_novatel = read_json(
            os.path.join(self.root_dir, 'vehicle-side/calib/lidar_to_novatel/' + str(veh_frame_id) + '.json'))
        novatel_to_world = read_json(
            os.path.join(self.root_dir, 'vehicle-side/calib/novatel_to_world/' + str(veh_frame_id) + '.json'))
        transformation_matrix = veh_side_rot_and_trans_to_trasnformation_matrix(lidar_to_novatel, novatel_to_world)
        data[0]['params']['lidar_pose'] = tfm_to_pose(transformation_matrix)

        virtuallidar_to_world = read_json(os.path.join(self.root_dir,
                                                       'infrastructure-side/calib/virtuallidar_to_world/' + str(
                                                           inf_frame_id) + '.json'))
        transformation_matrix = inf_side_rot_and_trans_to_trasnformation_matrix(virtuallidar_to_world,
                                                                                system_error_offset)
        data[1]['params']['lidar_pose'] = tfm_to_pose(transformation_matrix)

        # TODO: cooperative label not divived into front and all
        data[0]['params']['vehicles_front'] = read_json(os.path.join(self.root_dir,
                                                                     'cooperative/label/' + str(
                                                                         veh_frame_id) + '.json'))
        data[0]['params']['vehicles_all'] = data[0]['params']['vehicles_front']

        data[1]['params']['vehicles_front'] = []  # we only load cooperative label in vehicle side
        data[1]['params']['vehicles_all'] = []  # we only load cooperative label in vehicle side
        #################

        if self.load_camera_file:
            data[0]['camera_data'] = load_camera_data(
                [os.path.join(self.root_dir, 'vehicle-side/image/' + str(veh_frame_id) + '.jpg')])
            data[0]['params']['camera0'] = OrderedDict()
            data[0]['params']['camera0']['extrinsic'] = rot_and_trans_to_trasnformation_matrix(
                read_json(
                    os.path.join(self.root_dir, 'vehicle-side/calib/lidar_to_camera/' + str(veh_frame_id) + '.json')))
            data[0]['params']['camera0']['intrinsic'] = load_intrinsic_DAIR_V2X(
                read_json(
                    os.path.join(self.root_dir, 'vehicle-side/calib/camera_intrinsic/' + str(veh_frame_id) + '.json')))

            data[1]['camera_data'] = load_camera_data(
                [os.path.join(self.root_dir, 'infrastructure-side/image/' + str(inf_frame_id) + '.jpg')])
            data[1]['params']['camera0'] = OrderedDict()
            data[1]['params']['camera0']['extrinsic'] = rot_and_trans_to_trasnformation_matrix(
                read_json(os.path.join(self.root_dir, 'infrastructure-side/calib/virtuallidar_to_camera/' + str(
                    inf_frame_id) + '.json')))
            data[1]['params']['camera0']['intrinsic'] = load_intrinsic_DAIR_V2X(
                read_json(os.path.join(self.root_dir,
                                       'infrastructure-side/calib/camera_intrinsic/' + str(inf_frame_id) + '.json')))

        if self.load_lidar_file or self.visualize:
            data[0]['lidar_np'], _ = pcd_utils.read_pcd(
                os.path.join(self.root_dir, 'vehicle-side/velodyne/' + str(veh_frame_id) + '.pcd'))
            data[1]['lidar_np'], _ = pcd_utils.read_pcd(
                os.path.join(self.root_dir, 'infrastructure-side/velodyne/' + str(inf_frame_id) + '.pcd'))

        # Label for single side
        data[0]['params']['vehicles_single_front'] = read_json(os.path.join(self.root_dir,
                                                                            'vehicle-side/label/camera/{}.json'.format(
                                                                                veh_frame_id)))
        data[0]['params']['vehicles_single_all'] = read_json(
            os.path.join(self.root_dir, 'vehicle-side/label/lidar/{}.json'.format(veh_frame_id)))
        data[1]['params']['vehicles_single_front'] = read_json(os.path.join(self.root_dir,
                                                                            'infrastructure-side/label/camera/{}.json'.format(
                                                                                inf_frame_id)))
        data[1]['params']['vehicles_single_all'] = read_json(os.path.join(self.root_dir,
                                                                          'infrastructure-side/label/virtuallidar/{}.json'.format(
                                                                              inf_frame_id)))

        return data

    def __len__(self):
        return len(self.split_info)

    def __getitem__(self, idx):
        pass

    def generate_object_center_lidar(self,
                                     cav_contents,
                                     reference_lidar_pose):
        """
        reference lidar 's coordinate 
        """
        for cav_content in cav_contents:
            cav_content['params']['vehicles'] = cav_content['params']['vehicles_all']
        return self.post_processor.generate_object_center_v2xseq(cav_contents,
                                                                  reference_lidar_pose)

    def generate_object_center_camera(self,
                                      cav_contents,
                                      reference_lidar_pose):
        """
        reference lidar 's coordinate 
        """
        for cav_content in cav_contents:
            cav_content['params']['vehicles'] = cav_content['params']['vehicles_front']
        return self.post_processor.generate_object_center_v2xseq(cav_contents,
                                                                  reference_lidar_pose)

    ### Add new func for single side
    def generate_object_center_single(self,
                                      cav_contents,
                                      reference_lidar_pose,
                                      **kwargs):
        print('Not implemented yet')
        exit()
        """
        veh or inf 's coordinate
        """
        suffix = "_single"
        for cav_content in cav_contents:
            cav_content['params']['vehicles_single'] = \
                cav_content['params']['vehicles_single_front'] if self.label_type == 'camera' else \
                    cav_content['params']['vehicles_single_all']
        return self.post_processor.generate_object_center_dairv2x_single(cav_contents, suffix)

    def get_ext_int(self, params, camera_id):
        lidar_to_camera = params["camera%d" % camera_id]['extrinsic'].astype(np.float32)  # R_cw
        camera_to_lidar = np.linalg.inv(lidar_to_camera)  # R_wc
        camera_intrinsic = params["camera%d" % camera_id]['intrinsic'].astype(np.float32
                                                                              )
        return camera_to_lidar, camera_intrinsic

    def augment(self, lidar_np, object_bbx_center, object_bbx_mask):
        """
        Given the raw point cloud, augment by flipping and rotation.
        Parameters
        ----------
        lidar_np : np.ndarray
            (n, 4) shape
        object_bbx_center : np.ndarray
            (n, 7) shape to represent bbx's x, y, z, h, w, l, yaw
        object_bbx_mask : np.ndarray
            Indicate which elements in object_bbx_center are padded.
        """
        tmp_dict = {'lidar_np': lidar_np,
                    'object_bbx_center': object_bbx_center,
                    'object_bbx_mask': object_bbx_mask}
        tmp_dict = self.data_augmentor.forward(tmp_dict)

        lidar_np = tmp_dict['lidar_np']
        object_bbx_center = tmp_dict['object_bbx_center']
        object_bbx_mask = tmp_dict['object_bbx_mask']

        return lidar_np, object_bbx_center, object_bbx_mask


if __name__ == '__main__':
    path = "/GPFS/public/v2x-seq/V2X-Seq-SPD/V2X-Seq-SPD/cooperative/data_info.json"
    path = "/GPFS/public/v2x-seq/V2X-Seq-SPD/V2X-Seq-SPD/cooperative/label/000009.json"

    data = read_json(path)
    import pdb

    pdb.set_trace()
