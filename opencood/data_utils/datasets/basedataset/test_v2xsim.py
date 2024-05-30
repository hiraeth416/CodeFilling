import pickle
import numpy as np
from collections import OrderedDict

root_dir="/remote-home/share/junhaoge/v2xsim_infos_train.pkl"  # not dir, but file
with open(root_dir, 'rb') as f:
    dataset_info = pickle.load(f)
scene_database = dict()
min_timestap = 1e10
scene_flag = False
scenes_info = OrderedDict()
last_scene_idx = -1
last_scene_info_idx = -1
for i, scene_info in enumerate(dataset_info):
    # # print(i)
    # # print(scene_info["token"])
    # timestamp = scene_info["timestamp"] 
    # if timestamp < 20:
    #     scene_flag = True
    # if timestamp < min_timestap:
    #     min_timestap = timestamp
    # scene_database.update({i: OrderedDict()})
    # cav_num = scene_info['agent_num']
    # # print(cav_num)
    # # print(timestamp)
    # # print(type(scene_info))
    # # print(scene_info.keys())
    # # print(scene_info["lidar_path_1"])
    # scene_idx = scene_info['lidar_path_1'].split('/')[-1].split('.')[0].split('_')[1]
    # print(scene_idx)
    cav_num = scene_info['agent_num']
    scene_idx = scene_info['lidar_path_1'].split('/')[-1].split('.')[0].split('_')[1]
    scene_info_idx = int(scene_info['lidar_path_1'].split('/')[-1].split('.')[0].split('_')[2])

    if last_scene_idx != scene_idx:
        scenes_info[scene_idx] = OrderedDict()
        scenes_info[scene_idx]["min_idx"] = scene_info_idx
        if last_scene_idx != -1:
            scenes_info[last_scene_idx]["max_idx"] = last_scene_info_idx
        last_scene_idx = scene_idx
    else:
        pass

    last_scene_info_idx = scene_info_idx
    # break
    # assert cav_num > 0
    # # with open(".txt", 'rb') as f:
    # #     lidar_data = pickle.load(f)
    # cav_ids = list(range(1, cav_num + 1))
    # max_cav = 5

    # for j, cav_id in enumerate(cav_ids):
    #     if j > max_cav - 1:
    #         print('too many cavs reinitialize')
    #         break
        
    #     scene_database[i][cav_id] = OrderedDict()

    #     scene_database[i][cav_id]['ego'] = j==0

    #     scene_database[i][cav_id]['lidar'] = scene_info[f'lidar_path_{cav_id}']
    #     # need to delete this line is running in /GPFS
    #     scene_database[i][cav_id]['lidar'] = \
    #         scene_database[i][cav_id]['lidar'].replace("/GPFS/rhome/yifanlu/workspace/dataset/v2xsim2-complete", "dataset/V2X-Sim-2.0")

    #     scene_database[i][cav_id]['params'] = OrderedDict()
    #     scene_database[i][cav_id]['params'][
    #         'vehicles'] = scene_info[f'labels_{cav_id}'][
    #             'gt_boxes_global']
    #     scene_database[i][cav_id]['params'][
    #         'object_ids'] = scene_info[f'labels_{cav_id}'][
    #             'gt_object_ids'].tolist()
print(scenes_info)