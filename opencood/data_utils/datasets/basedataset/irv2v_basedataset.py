# Author: sizhewei
# date: 2023-04-27
# License: MIT
# IRV2V dataset

import os
from collections import OrderedDict
import cv2
import h5py
import torch
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
from scipy import stats
import json
import random
import opencood.utils.pcd_utils as pcd_utils
from opencood.data_utils.augmentor.data_augmentor import DataAugmentor
from opencood.hypes_yaml.yaml_utils import load_yaml
from opencood.utils.camera_utils import load_camera_data
from opencood.utils.transformation_utils import x1_to_x2
from opencood.data_utils.pre_processor import build_preprocessor
from opencood.data_utils.post_processor import build_postprocessor

class OPV2VBaseDataset(Dataset):
    def __init__(self, params, visualize, train=True):
        self.params = params
        self.visualize = visualize
        self.train = train

        self.pre_processor = build_preprocessor(params["preprocess"], train)
        self.post_processor = build_postprocessor(params["postprocess"], train)
        self.data_augmentor = DataAugmentor(params['data_augment'],
                                            train)

        if self.train:
            root_dir = params['root_dir']
        else:
            root_dir = params['validate_dir']
        self.root_dir = root_dir 

        print("Dataset dir:", root_dir)

        if 'train_params' not in params or \
                'max_cav' not in params['train_params']:
            self.max_cav = 5
        else:
            self.max_cav = params['train_params']['max_cav']
        
        if 'binomial_n' in params:
            self.binomial_n = params['binomial_n']
        else:
            self.binomial_n = 0

        if 'binomial_p' in params:
            self.binomial_p = params['binomial_p']
        else:
            self.binomial_p = 1

        if 'time_delay' in params:
            self.time_delay = params['time_delay']
        else:
            self.time_delay = 0

        # 控制在0时刻是否有扰动
        self.is_no_shift = False
        if 'is_no_shift' in params and params['is_no_shift']:
            self.is_no_shift = True

        self.load_lidar_file = True if 'lidar' in params['input_source'] or self.visualize else False
        self.load_camera_file = True if 'camera' in params['input_source'] else False
        self.load_depth_file = True if 'depth' in params['input_source'] else False

        self.label_type = params['label_type'] # 'lidar' or 'camera'
        self.generate_object_center = self.generate_object_center_lidar if self.label_type == "lidar" \
                                            else self.generate_object_center_camera
        self.generate_object_center_single = self.generate_object_center # will it follows 'self.generate_object_center' when 'self.generate_object_center' change?

        if self.load_camera_file:
            self.data_aug_conf = params["fusion"]["args"]["data_aug_conf"]

        # by default, we load lidar, camera and metadata. But users may
        # define additional inputs/tasks
        self.add_data_extension = \
            params['add_data_extension'] if 'add_data_extension' \
                                            in params else []

        if "noise_setting" not in self.params:
            self.params['noise_setting'] = OrderedDict()
            self.params['noise_setting']['add_noise'] = False

        # first load all paths of different scenarios
        scenario_folders = sorted([os.path.join(root_dir, x)
                                   for x in os.listdir(root_dir) if
                                   os.path.isdir(os.path.join(root_dir, x))])
        self.scenario_folders = scenario_folders

        self.reinitialize()


    def reinitialize(self):
        # Structure: {scenario_id : {cav_1 : {timestamp1 : {yaml: path,
        # lidar: path, cameras:list of path}}}}
        self.scenario_database = OrderedDict()
        self.len_record = []

        # loop over all scenarios
        for (i, scenario_folder) in enumerate(self.scenario_folders):
            self.scenario_database.update({i: OrderedDict()})

            # at least 1 cav should show up
            if self.train:
                cav_list = [x for x in os.listdir(scenario_folder)
                            if os.path.isdir(
                        os.path.join(scenario_folder, x))]
                # cav_list = sorted(cav_list)
                random.shuffle(cav_list)
            else:
                cav_list = sorted([x for x in os.listdir(scenario_folder)
                                   if os.path.isdir(
                        os.path.join(scenario_folder, x))], key=lambda y:int(y))
            assert len(cav_list) > 0

            # roadside unit data's id is always negative, so here we want to
            # make sure they will be in the end of the list as they shouldn't
            # be ego vehicle.
            if int(cav_list[0]) < 0:
                cav_list = cav_list[1:] + [cav_list[0]]

            # copy timestamps npy file
            timestamps_file = os.path.join(scenario_folder, 'timestamps.npy')
            time_annotations = np.load(timestamps_file)

            yaml_files = \
                sorted([x
                        for x in os.listdir(os.path.join(scenario_folder, cav_list[0])) if
                        x.endswith('.json') and 'additional' not in x], 
                        key=lambda y:float((y.split('/')[-1]).split('.json')[0]))

            # for IRV2V dataset
            # sizhewei @ 2023/04/20
            start_timestamp = int(float(self.extract_timestamps(yaml_files)[0]))
            while(1):
                time_id_json = ("%.3f" % float(start_timestamp)) + ".json"
                time_id_yaml = ("%.3f" % float(start_timestamp)) + ".yaml"
                if not (time_id_json in yaml_files or time_id_yaml in yaml_files):
                    start_timestamp += 1
                else:
                    break
            end_timestamp = int(float(self.extract_timestamps(yaml_files)[-1]))
            if start_timestamp%2 == 0:
                # even
                end_timestamp = end_timestamp-1 if end_timestamp%2==1 else end_timestamp
            else:
                end_timestamp = end_timestamp-1 if end_timestamp%2==0 else end_timestamp
            num_timestamps = int((end_timestamp - start_timestamp)/2 + 1)
            regular_timestamps = [start_timestamp+2*i for i in range(num_timestamps)]
            #####################################################

            # loop over all CAV data
            for (j, cav_id) in enumerate(cav_list):
                if j > self.max_cav - 1:
                    print('too many cavs reinitialize')
                    break
                self.scenario_database[i][cav_id] = OrderedDict()

                # save all yaml files to the dictionary
                cav_path = os.path.join(scenario_folder, cav_id)

                if j == 0:
                    timestamps = regular_timestamps
                else:
                    timestamps = list(time_annotations[j-1, :])

                for timestamp in timestamps:
                    # for irregular dataset:
                    # sizhewei @ 2023/04/20
                    timestamp = "%.3f" % float(timestamp)
                    ####################################

                    self.scenario_database[i][cav_id][timestamp] = \
                        OrderedDict()
                    yaml_file = os.path.join(cav_path,
                                             timestamp + '.yaml')
                    # for irregular dataset:
                    # sizhewei 
                    yaml_file = yaml_file.replace("OPV2V_irregular_npy", "OPV2V_irregular_npy_updated")
                    ############################

                    lidar_file = os.path.join(cav_path,
                                              timestamp + '.pcd')
                    camera_files = self.find_camera_files(cav_path, 
                                                timestamp)
                    depth_files = self.find_camera_files(cav_path, 
                                                timestamp,sensor="depth")

                    self.scenario_database[i][cav_id][timestamp]['yaml'] = \
                        yaml_file
                    self.scenario_database[i][cav_id][timestamp]['lidar'] = \
                        lidar_file
                    self.scenario_database[i][cav_id][timestamp]['cameras'] = \
                        camera_files
                    self.scenario_database[i][cav_id][timestamp]['depths'] = \
                        depth_files

                   # load extra data
                    for file_extension in self.add_data_extension:
                        file_name = \
                            os.path.join(cav_path,
                                         timestamp + '_' + file_extension)

                        self.scenario_database[i][cav_id][timestamp][
                            file_extension] = file_name                  

                # regular的timestamps 用于做 curr 真实时刻的ground truth
                self.scenario_database[i][cav_id]['regular'] = OrderedDict()
                for timestamp in regular_timestamps:
                    timestamp = "%.3f" % float(timestamp)
                    self.scenario_database[i][cav_id]['regular'][timestamp] = \
                        OrderedDict()

                    yaml_file = os.path.join(cav_path,
                                             timestamp + '.yaml')
                    lidar_file = os.path.join(cav_path,
                                              timestamp + '.pcd')
                    # camera_files = self.load_camera_files(cav_path, timestamp)

                    self.scenario_database[i][cav_id]['regular'][timestamp]['yaml'] = \
                        yaml_file
                    self.scenario_database[i][cav_id]['regular'][timestamp]['lidar'] = \
                        lidar_file
                
                # Assume all cavs will have the same timestamps length. Thus
                # we only need to calculate for the first vehicle in the 
                # scene.
                if j == 0:
                    # we regard the agent with the minimum id as the ego
                    self.scenario_database[i][cav_id]['ego'] = True
                    if not self.len_record:
                        self.len_record.append(len(timestamps))
                    else:
                        prev_last = self.len_record[-1]
                        self.len_record.append(prev_last + len(timestamps))
                else:
                    self.scenario_database[i][cav_id]['ego'] = False


    def retrieve_base_data(self, idx):
        """
        Given the index, return the corresponding data.

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
        # we loop the accumulated length list to see get the scenario index
        scenario_index = 0
        for i, ele in enumerate(self.len_record):
            if idx < ele:
                scenario_index = i
                break
        scenario_database = self.scenario_database[scenario_index]

        # 生成冻结分布函数
        bernoulliDist = stats.bernoulli(self.binomial_p) 

        # check the timestamp index
        curr_timestamp_idx = idx if scenario_index == 0 else \
            idx - self.len_record[scenario_index - 1]
        # retrieve the corresponding timestamp key
        curr_timestamp_key = curr_timestamp_idx + self.binomial_n
        # self.return_timestamp_key(scenario_database, timestamp_index)

        data = OrderedDict()
        # load files for all CAVs
        for cav_id, cav_content in scenario_database.items():
            data[cav_id] = OrderedDict()
            data[cav_id]['ego'] = cav_content['ego']

            # 2. current timestamp, for label
            data[cav_id]['curr'] = {}
            timestamp_key = list(cav_content['regular'].items())[curr_timestamp_idx][0]

            # 2.1 load curr params
            # json is faster than yaml
            json_file = cav_content['regular'][timestamp_key]['yaml'].replace("yaml", "json")
            
            scene_name = json_file.split('/')[-3]
            if os.path.exists(json_file):
                with open(json_file, "r") as f:
                    data[cav_id]['curr']['params'] = json.load(f)
            else:
                data[cav_id]['curr']['params'] = \
                            load_yaml(cav_content['regular'][timestamp_key]['yaml'])

            # 2.2 load curr lidar file
            npy_file = cav_content['regular'][timestamp_key]['lidar'].replace("pcd", "npy")
            data[cav_id]['curr']['lidar_np'] = np.load(npy_file)

            # 3. previous timestamp, for input
            data[cav_id]['prev'] = {}
            latest_sample_stamp_idx = curr_timestamp_idx
            if data[cav_id]['ego']:
                data[cav_id]['prev'] = data[cav_id]['curr']
            else:
                # B(n, p)
                trails = bernoulliDist.rvs(self.binomial_n)
                # sample_interval = sum(trails)
                sample_interval = self.time_delay
                if sample_interval == 0:
                    tmp_time_key = list(cav_content.items())[latest_sample_stamp_idx][0]
                    if self.dist_time(tmp_time_key, data[cav_id]['curr']['timestamp'])>0:
                        sample_interval = 1
            latest_sample_stamp_idx -= sample_interval
            timestamp_key = list(cav_content.items())[latest_sample_stamp_idx][0]
            json_file = cav_content[timestamp_key]['yaml'].replace("yaml", "json")
            with open(json_file, "r") as f:
                data[cav_id]['prev']['params'] = json.load(f)

            # load lidar file: npy is faster than pcd
            npy_file = cav_content[timestamp_key]['lidar'].replace("pcd", "npy")
            data[cav_id]['prev']['lidar_np'] = np.load(npy_file)

            data[cav_id]['prev']['timestamp'] = timestamp_key
            data[cav_id]['prev']['sample_interval'] = sample_interval
            data[cav_id]['prev']['time_diff'] = \
                self.dist_time(timestamp_key, data[cav_id]['curr']['timestamp'])

            '''
            ### original code below:
            # load param file: json is faster than yaml
            json_file = cav_content[timestamp_key]['yaml'].replace("yaml", "json")
            if os.path.exists(json_file):
                with open(json_file, "r") as f:
                    data[cav_id]['params'] = json.load(f)
            else:
                data[cav_id]['params'] = \
                    load_yaml(cav_content[timestamp_key]['yaml'])

            # load camera file: hdf5 is faster than png
            hdf5_file = cav_content[timestamp_key]['cameras'][0].replace("camera0.png", "imgs_hdf5")
            if os.path.exists(hdf5_file):
                with h5py.File(hdf5_file, "r") as f:
                    data[cav_id]['camera_data'] = []
                    data[cav_id]['depth_data'] = []
                    for i in range(4):
                        data[cav_id]['camera_data'].append(Image.fromarray(f[f'camera{i}'][()]))
                        data[cav_id]['depth_data'].append(Image.fromarray(f[f'depth{i}'][()]))
            else:
                if self.load_camera_file:
                    data[cav_id]['camera_data'] = \
                        load_camera_data(cav_content[timestamp_key]['cameras'])
                if self.load_depth_file:
                    data[cav_id]['depth_data'] = \
                        load_camera_data(cav_content[timestamp_key]['depths']) 

            # load lidar file
            if self.load_lidar_file or self.visualize:
                # original file below
                # data[cav_id]['lidar_np'] = \
                #     pcd_utils.pcd_to_np(cav_content[timestamp_key]['lidar'])

                # for irregular dataset
                # sizhewei
                npy_file = cav_content[timestamp_key]['lidar'].replace("pcd", "npy")
                data[cav_id]['lidar_np'] = np.load(npy_file)
                ####################################


            for file_extension in self.add_data_extension:
                # if not find in the current directory
                # go to additional folder
                if not os.path.exists(cav_content[timestamp_key][file_extension]):
                    cav_content[timestamp_key][file_extension] = cav_content[timestamp_key][file_extension].replace("train","additional/train")
                    cav_content[timestamp_key][file_extension] = cav_content[timestamp_key][file_extension].replace("validate","additional/validate")
                    cav_content[timestamp_key][file_extension] = cav_content[timestamp_key][file_extension].replace("test","additional/test")

                if '.yaml' in file_extension:
                    data[cav_id][file_extension] = \
                        load_yaml(cav_content[timestamp_key][file_extension])
                else:
                    data[cav_id][file_extension] = \
                        cv2.imread(cav_content[timestamp_key][file_extension])
            '''

        return data

    def __len__(self):
        return self.len_record[-1]

    def __getitem__(self, idx):
        """
        Abstract method, needs to be define by the children class.
        """
        pass

    @staticmethod
    def extract_timestamps(yaml_files):
        """
        Given the list of the yaml files, extract the mocked timestamps.

        Parameters
        ----------
        yaml_files : list
            The full path of all yaml files of ego vehicle

        Returns
        -------
        timestamps : list
            The list containing timestamps only.
        """
        timestamps = []

        for file in yaml_files:
            res = file.split('/')[-1]
            if res.endswith('.yaml'):
                timestamp = res.replace('.yaml', '')
            elif res.endswith('.json'):
                timestamp = res.replace('.json', '')
            else:
                print("Woops! There is no processing method for file {}".format(res))
                sys.exit(1)
            timestamps.append(timestamp)

        return timestamps

    @staticmethod
    def return_timestamp_key(scenario_database, timestamp_index):
        """
        Given the timestamp index, return the correct timestamp key, e.g.
        2 --> '000078'.

        Parameters
        ----------
        scenario_database : OrderedDict
            The dictionary contains all contents in the current scenario.

        timestamp_index : int
            The index for timestamp.

        Returns
        -------
        timestamp_key : str
            The timestamp key saved in the cav dictionary.
        """
        # get all timestamp keys
        timestamp_keys = list(scenario_database.items())[0][1]
        # retrieve the correct index
        timestamp_key = list(timestamp_keys.items())[timestamp_index][0]

        return timestamp_key

    @staticmethod
    def find_camera_files(cav_path, timestamp, sensor="camera"):
        """
        Retrieve the paths to all camera files.

        Parameters
        ----------
        cav_path : str
            The full file path of current cav.

        timestamp : str
            Current timestamp

        sensor : str
            "camera" or "depth" 

        Returns
        -------
        camera_files : list
            The list containing all camera png file paths.
        """
        camera0_file = os.path.join(cav_path,
                                    timestamp + f'_{sensor}0.png')
        camera1_file = os.path.join(cav_path,
                                    timestamp + f'_{sensor}1.png')
        camera2_file = os.path.join(cav_path,
                                    timestamp + f'_{sensor}2.png')
        camera3_file = os.path.join(cav_path,
                                    timestamp + f'_{sensor}3.png')
        return [camera0_file, camera1_file, camera2_file, camera3_file]


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


    def generate_object_center_lidar(self,
                               cav_contents,
                               reference_lidar_pose):
        """
        Retrieve all objects in a format of (n, 7), where 7 represents
        x, y, z, l, w, h, yaw or x, y, z, h, w, l, yaw.
        The object_bbx_center is in ego coordinate.

        Notice: it is a wrap of postprocessor

        Parameters
        ----------
        cav_contents : list
            List of dictionary, save all cavs' information.
            in fact it is used in get_item_single_car, so the list length is 1

        reference_lidar_pose : list
            The final target lidar pose with length 6.

        Returns
        -------
        object_np : np.ndarray
            Shape is (max_num, 7).
        mask : np.ndarray
            Shape is (max_num,).
        object_ids : list
            Length is number of bbx in current sample.
        """
        return self.post_processor.generate_object_center(cav_contents,
                                                        reference_lidar_pose)

    def generate_object_center_camera(self, 
                                cav_contents, 
                                reference_lidar_pose):
        """
        Retrieve all objects in a format of (n, 7), where 7 represents
        x, y, z, l, w, h, yaw or x, y, z, h, w, l, yaw.
        The object_bbx_center is in ego coordinate.

        Notice: it is a wrap of postprocessor

        Parameters
        ----------
        cav_contents : list
            List of dictionary, save all cavs' information.
            in fact it is used in get_item_single_car, so the list length is 1

        reference_lidar_pose : list
            The final target lidar pose with length 6.

        visibility_map : np.ndarray
            for OPV2V, its 256*256 resolution. 0.39m per pixel. heading up.

        Returns
        -------
        object_np : np.ndarray
            Shape is (max_num, 7).
        mask : np.ndarray
            Shape is (max_num,).
        object_ids : list
            Length is number of bbx in current sample.
        """
        return self.post_processor.generate_visible_object_center(
            cav_contents, reference_lidar_pose
        )

    def get_ext_int(self, params, camera_id):
        camera_coords = np.array(params["camera%d" % camera_id]["cords"]).astype(
            np.float32)
        camera_to_lidar = x1_to_x2(
            camera_coords, params["lidar_pose_clean"]
        ).astype(np.float32)  # T_LiDAR_camera
        camera_to_lidar = camera_to_lidar @ np.array(
            [[0, 0, 1, 0], [1, 0, 0, 0], [0, -1, 0, 0], [0, 0, 0, 1]],
            dtype=np.float32)  # UE4 coord to opencv coord
        camera_intrinsic = np.array(params["camera%d" % camera_id]["intrinsic"]).astype(
            np.float32
        )
        return camera_to_lidar, camera_intrinsic