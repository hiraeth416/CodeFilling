      
# intermediate fusion dataset
import random
import math
from collections import OrderedDict
import numpy as np
import torch
import copy
from icecream import ic
from PIL import Image
import pickle as pkl
from opencood.utils import box_utils as box_utils
from opencood.data_utils.pre_processor import build_preprocessor
from opencood.data_utils.post_processor import build_postprocessor
from opencood.utils.camera_utils import (
    sample_augmentation,
    img_transform,
    normalize_img,
    img_to_tensor,
)
from opencood.utils.heter_utils import AgentSelector
from opencood.utils.common_utils import merge_features_to_dict
from opencood.utils.transformation_utils import x1_to_x2, x_to_world, get_pairwise_transformation
from opencood.utils.pose_utils import add_noise_data_dict
from opencood.utils.pcd_utils import (
    mask_points_by_range,
    mask_ego_points,
    shuffle_points,
    downsample_lidar_minimum,
)
from opencood.utils.common_utils import read_json
from opencood.data_utils.datasets.intermediate_fusion_dataset import getIntermediateFusionDataset
from opencood.utils.flow_utils import generate_flow_map


def getIntermediateFusionMultiSweepDataset(cls):
    """
    cls: the Basedataset.
    """
    cur_cls = getIntermediateFusionDataset(cls)

    class IntermediateFusionMultiSweepDataset(cur_cls):
        def __init__(self, params, visualize, train=True):
            super().__init__(params, visualize, train)

            # number of past-frames
            if 'num_sweep_frames' in params:
                self.k = params['num_sweep_frames']
            else:
                self.k = 1

        def retrieve_data_sequence(self, idx):
            scenario_index = self.retrieve_scene_idx(idx)
            # timestamp_index = idx if scenario_index == 0 else \
            #             idx - self.len_record[scenario_index - 1]
            # curr_timestamp_index = timestamp_index + self.k - 1
            curr_timestamp_index = idx + self.k - 1
            curr_scenario_index = self.retrieve_scene_idx(curr_timestamp_index)
            while curr_scenario_index != scenario_index:
                curr_timestamp_index -= 1
                curr_scenario_index = self.retrieve_scene_idx(curr_timestamp_index)

            data = OrderedDict()
            for i in range(self.k):
                # check the timestamp index
                past_timestamp_index = curr_timestamp_index - i
                past_base_data_dict = self.retrieve_base_data(past_timestamp_index)
                past_base_data_dict = add_noise_data_dict(past_base_data_dict, self.params['noise_setting'])
                data[i] = past_base_data_dict
            return data

        def __getitem__(self, idx):
            base_data_seq_dict = self.retrieve_data_sequence(idx)
            base_data_dict = base_data_seq_dict[0]

            processed_data_dict = OrderedDict()
            processed_data_dict['ego'] = {}

            ego_id = -1
            ego_lidar_pose = []
            ego_cav_base = None

            # first find the ego vehicle's lidar pose
            for cav_id, cav_content in base_data_dict.items():
                if cav_content['ego']:
                    ego_id = cav_id
                    ego_lidar_pose = cav_content['params']['lidar_pose']
                    break

            assert cav_id == list(base_data_dict.keys())[
                0], "The first element in the OrderedDict must be ego"
            assert ego_id != -1
            assert len(ego_lidar_pose) > 0

            agents_image_inputs = []
            processed_features = []
            object_stack = []
            object_id_stack = []
            prev_object_stack = {}
            prev_object_id_stack = {}
            single_label_list = []
            single_object_bbx_center_list = []
            single_object_bbx_mask_list = []
            too_far = []
            lidar_pose_list = []
            lidar_pose_clean_list = []
            cav_id_list = []
            projected_lidar_clean_list = []  # disconet

            if self.visualize or self.kd_flag:
                projected_lidar_stack = []

            # loop over all CAVs to process information
            for cav_id, selected_cav_base in base_data_dict.items():
                # check if the cav is within the communication range with ego
                distance = \
                    math.sqrt((selected_cav_base['params']['lidar_pose'][0] -
                               ego_lidar_pose[0]) ** 2 + (
                                      selected_cav_base['params'][
                                          'lidar_pose'][1] - ego_lidar_pose[
                                          1]) ** 2)

                # if distance is too far, we will just skip this agent
                if distance > self.params['comm_range']:
                    too_far.append(cav_id)
                    continue
                cav_id_list.append(cav_id)

            for cav_id in too_far:
                base_data_dict.pop(cav_id)

            ########## Updated by Yifan Lu 2022.1.26 ############
            # box align to correct pose.
            # stage1_content contains all agent. Even out of comm range.
            if self.box_align and str(idx) in self.stage1_result.keys():
                from opencood.models.sub_modules.box_align_v2 import box_alignment_relative_sample_np
                stage1_content = self.stage1_result[str(idx)]
                if stage1_content is not None:
                    all_agent_id_list = stage1_content['cav_id_list']  # include those out of range
                    all_agent_corners_list = stage1_content['pred_corner3d_np_list']
                    all_agent_uncertainty_list = stage1_content['uncertainty_np_list']

                    cur_agent_id_list = cav_id_list
                    cur_agent_pose = [base_data_dict[cav_id]['params']['lidar_pose'] for cav_id in cav_id_list]
                    cur_agnet_pose = np.array(cur_agent_pose)
                    cur_agent_in_all_agent = [all_agent_id_list.index(cur_agent) for cur_agent in
                                              cur_agent_id_list]  # indexing current agent in `all_agent_id_list`

                    pred_corners_list = [np.array(all_agent_corners_list[cur_in_all_ind], dtype=np.float64)
                                         for cur_in_all_ind in cur_agent_in_all_agent]
                    uncertainty_list = [np.array(all_agent_uncertainty_list[cur_in_all_ind], dtype=np.float64)
                                        for cur_in_all_ind in cur_agent_in_all_agent]

                    if sum([len(pred_corners) for pred_corners in pred_corners_list]) != 0:
                        refined_pose = box_alignment_relative_sample_np(pred_corners_list,
                                                                        cur_agnet_pose,
                                                                        uncertainty_list=uncertainty_list,
                                                                        **self.box_align_args)
                        cur_agnet_pose[:, [0, 1, 4]] = refined_pose

                        for i, cav_id in enumerate(cav_id_list):
                            lidar_pose_list[i] = cur_agnet_pose[i].tolist()
                            base_data_dict[cav_id]['params']['lidar_pose'] = cur_agnet_pose[i].tolist()

            pairwise_t_matrix = \
                get_pairwise_transformation(base_data_dict,
                                            self.max_cav,
                                            self.proj_first)

            # merge preprocessed features from different cavs into the same dict
            cav_num = len(cav_id_list)

            # heterogeneous
            if self.heterogeneous:
                lidar_agent, camera_agent = self.selector.select_agent(idx)
                lidar_agent = lidar_agent[:cav_num]
                processed_data_dict['ego'].update({"lidar_agent": lidar_agent})

            ########################
            cav_all_prev_object_stack = {}
            cav_all_prev_object_id_stack = {}
            ########################
            for _j, ego_cav_id in enumerate(cav_id_list):
                cav_all_prev_object_stack[ego_cav_id] = {}
                cav_all_prev_object_id_stack[ego_cav_id] = {}

                for _i, cav_id in enumerate(cav_id_list):
                    # dynamic object center generator! for heterogeneous input
                    if (not self.visualize) and self.heterogeneous and lidar_agent[_i]:
                        self.generate_object_center = self.generate_object_center_lidar
                    elif (not self.visualize) and self.heterogeneous and (not lidar_agent[_i]):
                        self.generate_object_center = self.generate_object_center_camera

                    for t_i in range(self.k):
                        if cav_id in base_data_seq_dict[t_i]:
                            selected_cav_base = base_data_seq_dict[t_i][cav_id]
                        else:
                            selected_cav_base = base_data_seq_dict[0][cav_id]

                        ego_cav_base = base_data_seq_dict[0][ego_cav_id]
                        selected_cav_processed = self.get_item_single_car(
                            selected_cav_base,
                            ego_cav_base)

                        ################# FOR FLOW ##############
                        # Save gts for each timestamp
                        if t_i in cav_all_prev_object_stack[ego_cav_id]:
                            cav_all_prev_object_stack[ego_cav_id][t_i].append(selected_cav_processed['object_bbx_center'])
                            cav_all_prev_object_id_stack[ego_cav_id][t_i] += selected_cav_processed['object_ids']
                        else:
                            cav_all_prev_object_stack[ego_cav_id][t_i] = [selected_cav_processed['object_bbx_center']]
                            cav_all_prev_object_id_stack[ego_cav_id][t_i] = selected_cav_processed['object_ids']
                        ################# FOR FLOW ##############

            for _i, cav_id in enumerate(cav_id_list):
                # dynamic object center generator! for heterogeneous input
                if (not self.visualize) and self.heterogeneous and lidar_agent[_i]:
                    self.generate_object_center = self.generate_object_center_lidar
                elif (not self.visualize) and self.heterogeneous and (not lidar_agent[_i]):
                    self.generate_object_center = self.generate_object_center_camera

                for t_i in range(self.k):
                    if cav_id in base_data_seq_dict[t_i]:
                        selected_cav_base = base_data_seq_dict[t_i][cav_id]
                    else:
                        selected_cav_base = base_data_seq_dict[0][cav_id]
                    lidar_pose_clean_list.append(selected_cav_base['params']['lidar_pose_clean'])
                    lidar_pose_list.append(selected_cav_base['params']['lidar_pose'])  # 6dof pose

                    ego_cav_base = base_data_seq_dict[0][ego_id]
                    selected_cav_processed = self.get_item_single_car(
                        selected_cav_base,
                        ego_cav_base)

                    if self.load_lidar_file:
                        processed_features.append(
                            selected_cav_processed['processed_features'])
                    if self.load_camera_file:
                        agents_image_inputs.append(
                            selected_cav_processed['image_inputs'])

                    if self.visualize or self.kd_flag:
                        projected_lidar_stack.append(
                            selected_cav_processed['projected_lidar'])

                    # Current time
                    if t_i == 0:
                        object_stack.append(selected_cav_processed['object_bbx_center'])
                        object_id_stack += selected_cav_processed['object_ids']

                        if self.supervise_single:
                            single_label_list.append(selected_cav_processed['single_label_dict'])
                            single_object_bbx_center_list.append(selected_cav_processed['single_object_bbx_center'])
                            single_object_bbx_mask_list.append(selected_cav_processed['single_object_bbx_mask'])
            # lidar pose of all the cavs across all the frame
            lidar_poses = np.array(lidar_pose_list).reshape(-1, 6)  # [N_cav, self.k, 6]
            lidar_poses_clean = np.array(lidar_pose_clean_list).reshape(-1, 6)  # [N_cav, self.k, 6]

            # generate single view GT label
            if self.supervise_single:
                single_label_dicts = self.post_processor.collate_batch(single_label_list)
                single_object_bbx_center = torch.from_numpy(np.array(single_object_bbx_center_list))
                single_object_bbx_mask = torch.from_numpy(np.array(single_object_bbx_mask_list))
                processed_data_dict['ego'].update({
                    "single_label_dict_torch": single_label_dicts,
                    "single_object_bbx_center_torch": single_object_bbx_center,
                    "single_object_bbx_mask_torch": single_object_bbx_mask,
                })

            if self.kd_flag:
                stack_lidar_np = np.vstack(projected_lidar_stack)
                stack_lidar_np = mask_points_by_range(stack_lidar_np,
                                                      self.params['preprocess'][
                                                          'cav_lidar_range'])
                stack_feature_processed = self.pre_processor.preprocess(stack_lidar_np)
                processed_data_dict['ego'].update({'teacher_processed_lidar':
                                                       stack_feature_processed})


            #######################################
            # import pdb
            # pdb.set_trace()
            flow_map = np.zeros((len(cav_id_list), 2, 96, 352))
            b_idx = 0
            for cav_id in cav_id_list:

                prev_object_stack = cav_all_prev_object_stack[cav_id]
                prev_object_id_stack = cav_all_prev_object_id_stack[cav_id]
                for t_i in range(self.k):
                    unique_object_ids = list(set(prev_object_id_stack[t_i]))
                    unique_indices = \
                        [prev_object_id_stack[t_i].index(x) for x in unique_object_ids]
                    prev_object_stack[t_i] = np.vstack(prev_object_stack[t_i])
                    prev_object_stack[t_i] = prev_object_stack[t_i][unique_indices]
                    prev_object_id_stack[t_i] = unique_object_ids

                flow_map_each = generate_flow_map(prev_object_stack,
                                         prev_object_id_stack,
                                         self.params['preprocess']['cav_lidar_range'],
                                         self.params['preprocess']['args']['voxel_size'],
                                         past_k=1)[np.newaxis, :]
                flow_map[b_idx, :, :, :] = flow_map_each
                b_idx += 1

            #######################################

            # exclude all repetitive objects
            unique_indices = \
                [object_id_stack.index(x) for x in set(object_id_stack)]
            object_stack = np.vstack(object_stack)
            object_stack = object_stack[unique_indices]

            # make sure bounding boxes across all frames have the same number
            object_bbx_center = \
                np.zeros((self.params['postprocess']['max_num'], 7))
            mask = np.zeros(self.params['postprocess']['max_num'])
            object_bbx_center[:object_stack.shape[0], :] = object_stack
            mask[:object_stack.shape[0]] = 1

            if self.load_lidar_file:
                merged_feature_dict = merge_features_to_dict(processed_features)
                processed_data_dict['ego'].update({'processed_lidar': merged_feature_dict})
            if self.load_camera_file:
                merged_image_inputs_dict = merge_features_to_dict(agents_image_inputs, merge='stack')
                processed_data_dict['ego'].update({'image_inputs': merged_image_inputs_dict})

            # generate targets label
            label_dict = \
                self.post_processor.generate_label(
                    gt_box_center=object_bbx_center,
                    anchors=self.anchor_box,
                    mask=mask)

            processed_data_dict['ego'].update(
                {'object_bbx_center': object_bbx_center,
                 'object_bbx_mask': mask,
                 'object_ids': [object_id_stack[i] for i in unique_indices],
                 'anchor_box': self.anchor_box,
                 'label_dict': label_dict,
                 'cav_num': cav_num,
                 'pairwise_t_matrix': pairwise_t_matrix,
                 'lidar_poses_clean': lidar_poses_clean,
                 'lidar_poses': lidar_poses})


            if self.visualize:
                processed_data_dict['ego'].update({'origin_lidar':
                    np.vstack(
                        projected_lidar_stack)})

            processed_data_dict['ego'].update({'sample_idx': idx,
                                               'cav_id_list': cav_id_list})

            ################# FOR FLOW ##############
            processed_data_dict['ego'].update({'flow_gt': flow_map,
                                               'prev_object_id_stack': prev_object_id_stack,
                                               'prev_object_stack': prev_object_stack})
            ################# FOR FLOW ##############

            return processed_data_dict

        def collate_batch_train(self, batch):
            output_dict = cur_cls.collate_batch_train(self, batch)
            flow_map_list = []
            for i in range(len(batch)):
                ego_dict = batch[i]['ego']
                flow_map_list.append(ego_dict['flow_gt'])
            flow_map = torch.from_numpy(np.concatenate(flow_map_list))
            output_dict['ego'].update({'flow_gt': flow_map})
            output_dict['ego']['label_dict'].update({'flow_gt': flow_map})
            return output_dict

        def collate_batch_test(self, batch):
            assert len(batch) <= 1, "Batch size 1 is required during testing!"
            output_dict = self.collate_batch_train(batch)
            if output_dict is None:
                return None

            # check if anchor box in the batch
            if batch[0]['ego']['anchor_box'] is not None:
                output_dict['ego'].update({'anchor_box':
                                               self.anchor_box_torch})

            # save the transformation matrix (4, 4) to ego vehicle
            # transformation is only used in post process (no use.)
            # we all predict boxes in ego coord.
            transformation_matrix_torch = \
                torch.from_numpy(np.identity(4)).float()
            transformation_matrix_clean_torch = \
                torch.from_numpy(np.identity(4)).float()

            output_dict['ego'].update({'transformation_matrix':
                                           transformation_matrix_torch,
                                       'transformation_matrix_clean':
                                           transformation_matrix_clean_torch, })

            output_dict['ego'].update({
                "sample_idx": batch[0]['ego']['sample_idx'],
                "cav_id_list": batch[0]['ego']['cav_id_list']
            })

            prev_info_list = []
            ego_dict = batch[0]['ego']
            for k_i in range(self.k):
                prev_info = OrderedDict()
                prev_info['object_bbx_center'] = torch.from_numpy(np.array(ego_dict['prev_object_stack'][k_i]))
                prev_info['object_bbx_mask'] = torch.from_numpy(np.ones(len(ego_dict['prev_object_stack'][k_i])))
                prev_info['object_ids'] = ego_dict['prev_object_id_stack'][k_i]
                prev_info['transformation_matrix_clean'] = transformation_matrix_clean_torch
                prev_info_list.append({'ego': prev_info})

            output_dict['ego'].update({'prev_info': prev_info_list})

            return output_dict

    return IntermediateFusionMultiSweepDataset


    