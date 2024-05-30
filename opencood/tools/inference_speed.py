# -*- coding: utf-8 -*-
# Author: Runsheng Xu <rxx3386@ucla.edu>, Hao Xiang <haxiang@g.ucla.edu>,
# License: TDG-Attribution-NonCommercial-NoDistrib


import argparse
import os
import time
from typing import OrderedDict
import importlib
import torch
import open3d as o3d
from torch.utils.data import DataLoader, Subset
import numpy as np
import opencood.hypes_yaml.yaml_utils as yaml_utils
from opencood.tools import train_utils, inference_utils_speed
from opencood.data_utils.datasets import build_dataset
from opencood.utils import eval_utils
from opencood.visualization import vis_utils, my_vis, simple_vis
from opencood.utils.common_utils import update_dict
import matplotlib.pyplot as plt
import time
torch.multiprocessing.set_sharing_strategy('file_system')

def test_parser():
    parser = argparse.ArgumentParser(description="synthetic data generation")
    parser.add_argument('--model_dir', type=str, required=True,
                        help='Continued training path')
    parser.add_argument('--fusion_method', type=str,
                        default='intermediate',
                        help='no, no_w_uncertainty, late, early or intermediate')
    parser.add_argument('--save_vis_interval', type=int, default=40,
                        help='interval of saving visualization')
    parser.add_argument('--save_npy', action='store_true',
                        help='whether to save prediction and gt result'
                             'in npy file')
    parser.add_argument('--range', type=str, default="140.8,40",
                        help="detection range is [-140.8,+140.8m, -40m, +40m]")
    parser.add_argument('--no_score', action='store_true',
                        help="whether print the score of prediction")
    parser.add_argument('--note', default="", type=str, help="any other thing?")
    parser.add_argument('--comm_thre', default=0.0, type=float, help="communication threhold")
    parser.add_argument('--result_name', default="", type=str, help="result txt name")
    parser.add_argument('--modal', type=int, default=0,
                        help='used in heterogeneous setting, 0 lidaronly, 1 camonly, 2 ego_lidar_other_cam, 3 ego_cam_other_lidar')


    
    parser.add_argument('--sample_method', type=str, default="none",
                        help="the method to downsample the point cloud")
    parser.add_argument('--store_boxes', default=True,action='store_true',
                        help= "store detection boxes and gt boxes")
    parser.add_argument('--sampling_rate', type=float, default=1.0)
    parser.add_argument('--sampler_path', type=str, default=None)
    parser.add_argument('--box_ratio',type=float,default=1.0)
    parser.add_argument('--background_ratio',type=float,default=1.0)
    parser.add_argument('--expansion_ratio',type=float,default=20.0)
    parser.add_argument('--w2cthreshold',type=float,default=0.01)
    parser.add_argument('--vis_score',type=str,default='confidence',
                        help='confidence or uncertainty')


    opt = parser.parse_args()
    return opt


def main():
    opt = test_parser()

    assert opt.fusion_method in ['late', 'early', 'intermediate', 'no', 'no_w_uncertainty', 'single'] 

    hypes = yaml_utils.load_yaml(None, opt)
    print("model_dir:", opt.model_dir)

    print("model:", hypes['model']['core_method'])
    hypes.update({'sample_method': opt.sample_method})
    hypes.update({'sampling_rate': opt.sampling_rate})
    hypes.update({'store_boxes': opt.store_boxes})
    hypes.update({'model_dir':opt.model_dir})
    hypes.update({'sampler_path':opt.sampler_path})
    hypes.update({'expansion_ratio':opt.expansion_ratio})
    hypes.update({'box_ratio':opt.box_ratio})
    hypes.update({'background_ratio':opt.background_ratio})
    hypes.update({'late_choose_method':False})
    """
    saved_file = os.path.join(opt.model_dir, 'result_pointnum.txt')
    point_num = np.loadtxt(saved_file)
    plt.hist(point_num, bins=20)
    plt.title("data analyze")
    plt.xlabel("point_num")
    plt.ylabel("percentage")
    plt.savefig(os.path.join(opt.model_dir, 'chart.png'))
    print("chart completed")
    """


    if 'heter' in hypes:
        # hypes['heter']['lidar_channels'] = 16
        # opt.note += "_16ch"
        if "OPV2V" in hypes['test_dir'] or "v2xsim" in hypes['test_dir']:
            if opt.modal == 0:
                hypes['heter']['mapping_dict']['m1'] = 'm1'
                hypes['heter']['mapping_dict']['m2'] = 'm1'
                hypes['heter']['mapping_dict']['m3'] = 'm1'
                hypes['heter']['mapping_dict']['m4'] = 'm1'
                hypes['heter']['ego_modality'] = 'm1'
                hypes['model']['args']['ego_modality'] = 'm1'
                opt.note += '_lidaronly' 

            if opt.modal == 1:
                hypes['heter']['mapping_dict']['m1'] = 'm2'
                hypes['heter']['mapping_dict']['m2'] = 'm2'
                hypes['heter']['mapping_dict']['m3'] = 'm2'
                hypes['heter']['mapping_dict']['m4'] = 'm2'
                hypes['heter']['ego_modality'] = 'm2'
                hypes['model']['args']['ego_modality'] = 'm2'
                opt.note += '_camonly' 

            if opt.modal == 2:
                hypes['heter']['mapping_dict']['m1'] = 'm1'
                hypes['heter']['mapping_dict']['m2'] = 'm2'
                hypes['heter']['mapping_dict']['m3'] = 'm2'
                hypes['heter']['mapping_dict']['m4'] = 'm2'
                hypes['heter']['ego_modality'] = 'm1'
                hypes['model']['args']['ego_modality'] = 'm1'
                opt.note += 'ego_lidar_other_cam'

            if opt.modal == 3:
                hypes['heter']['mapping_dict']['m1'] = 'm2'
                hypes['heter']['mapping_dict']['m2'] = 'm1'
                hypes['heter']['mapping_dict']['m3'] = 'm1'
                hypes['heter']['mapping_dict']['m4'] = 'm1'
                hypes['heter']['ego_modality'] = 'm2'
                hypes['model']['args']['ego_modality'] = 'm2'
                opt.note += '_ego_cam_other_lidar'

            if opt.modal == 4:
                hypes['heter']['mapping_dict']['m1'] = 'm1'
                hypes['heter']['mapping_dict']['m2'] = 'm1'
                hypes['heter']['mapping_dict']['m3'] = 'm2'
                hypes['heter']['mapping_dict']['m4'] = 'm2'
                hypes['heter']['ego_modality'] = 'm1&m2'
                opt.note += 'ego_random_ratio0.5'
        else:
            if opt.modal == 0:
                hypes['heter']['mapping_dict']['m1'] = 'm1'
                hypes['heter']['mapping_dict']['m2'] = 'm1'
                hypes['heter']['ego_modality'] = 'm1'
                opt.note += '_lidaronly' 

            if opt.modal == 1:
                hypes['heter']['mapping_dict']['m1'] = 'm2'
                hypes['heter']['mapping_dict']['m2'] = 'm2'
                hypes['heter']['ego_modality'] = 'm2'
                opt.note += '_camonly' 

            if opt.modal == 2:
                hypes['heter']['mapping_dict']['m1'] = 'm1'
                hypes['heter']['mapping_dict']['m2'] = 'm2'
                hypes['heter']['ego_modality'] = 'm1'
                opt.note += 'ego_lidar_other_cam'

            if opt.modal == 3:
                hypes['heter']['mapping_dict']['m1'] = 'm2'
                hypes['heter']['mapping_dict']['m2'] = 'm1'
                hypes['heter']['ego_modality'] = 'm2'
                opt.note += '_ego_cam_other_lidar'

            if opt.modal == 4:
                hypes['heter']['mapping_dict']['m1'] = 'm1'
                hypes['heter']['mapping_dict']['m2'] = 'm2'
                hypes['heter']['ego_modality'] = 'm1&m2'
                opt.note += 'ego_random_ratio0.5'
        if 'fusion_args' in hypes['model']['args']:
            hypes['model']['args']['fusion_args']['communication']['thre'] = opt.comm_thre
            print("comm_thre:", opt.comm_thre)
        x_min, x_max = -eval(opt.range.split(',')[0]), eval(opt.range.split(',')[0])
        y_min, y_max = -eval(opt.range.split(',')[1]), eval(opt.range.split(',')[1])
        opt.note += f"_{x_max}_{y_max}"

        new_cav_range = [x_min, y_min, hypes['postprocess']['anchor_args']['cav_lidar_range'][2], \
                            x_max, y_max, hypes['postprocess']['anchor_args']['cav_lidar_range'][5]]

        # replace all appearance
        hypes = update_dict(hypes, {
            "cav_lidar_range": new_cav_range,
            "lidar_range": new_cav_range,
            "gt_range": new_cav_range
        })

        # reload anchor
        yaml_utils_lib = importlib.import_module("opencood.hypes_yaml.yaml_utils")
        for name, func in yaml_utils_lib.__dict__.items():
            if name == hypes["yaml_parser"]:
                parser_func = func
        hypes = parser_func(hypes)
        

        
    
    hypes['validate_dir'] = hypes['test_dir']
    if "OPV2V" in hypes['test_dir'] or "v2xsim" in hypes['test_dir']:
        assert "test" in hypes['validate_dir']
    
    # This is used in visualization
    # left hand: OPV2V, V2XSet
    # right hand: V2X-Sim 2.0 and DAIR-V2X
    left_hand = True if ("OPV2V" in hypes['test_dir'] or "V2XSET" in hypes['test_dir']) else False

    print(f"Left hand visualizing: {left_hand}")

    if 'box_align' in hypes.keys():
        hypes['box_align']['val_result'] = hypes['box_align']['test_result']

    print('Creating Model')
    model = train_utils.create_model(hypes)
    # we assume gpu is necessary
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print('Loading Model from checkpoint')
    saved_path = opt.model_dir
    resume_epoch, model = train_utils.load_saved_model(saved_path, model)
    print(f"resume from {resume_epoch} epoch.")
    opt.note += f"_epoch{resume_epoch}"
    
    if torch.cuda.is_available():
        model.cuda()
    model.eval()

    # setting noise
    np.random.seed(303)
    
    # build dataset for each noise setting
    print('Dataset Building')
    opencood_dataset = build_dataset(hypes, visualize=True, train=False)
    # opencood_dataset_subset = Subset(opencood_dataset, range(640,2100))
    # data_loader = DataLoader(opencood_dataset_subset,
    data_loader = DataLoader(opencood_dataset,
                            batch_size=1,
                            num_workers=4,
                            collate_fn=opencood_dataset.collate_batch_test,
                            shuffle=False,
                            pin_memory=False,
                            drop_last=False)
    
    print("Dataset Building Ends")
    # Create the dictionary for evaluation
    result_stat = {0.3: {'tp': [], 'fp': [], 'gt': 0, 'score': []},                
                0.5: {'tp': [], 'fp': [], 'gt': 0, 'score': []},                
                0.7: {'tp': [], 'fp': [], 'gt': 0, 'score': []}}
    """
    result_stat1 = {0.3: {'tp': [], 'fp': [], 'gt': 0, 'score': []},                
                0.5: {'tp': [], 'fp': [], 'gt': 0, 'score': []},                
                0.7: {'tp': [], 'fp': [], 'gt': 0, 'score': []}}
    result_stat2 = {0.3: {'tp': [], 'fp': [], 'gt': 0, 'score': []},                
                0.5: {'tp': [], 'fp': [], 'gt': 0, 'score': []},                
                0.7: {'tp': [], 'fp': [], 'gt': 0, 'score': []}}
    result_stat3 = {0.3: {'tp': [], 'fp': [], 'gt': 0, 'score': []},                
                0.5: {'tp': [], 'fp': [], 'gt': 0, 'score': []},                
                0.7: {'tp': [], 'fp': [], 'gt': 0, 'score': []}}
    """
    
    infer_info = opt.fusion_method + opt.note

    comm_rates = []
    cav_nums = []
    num=0
    average_time = 0
    for i, batch_data in enumerate(data_loader):
        if i>1000 :
            break
        if opt.modal == 2 or opt.modal == 3:
            if 1459>=i>=1217 or 1672<=i<=1775 or 1996<=i<=2169:
                continue
        print('i:', i)
        print("----------------------------------------------------------------------------------")
        #print(f"{infer_info}_{i}")
        if batch_data is None:
            continue
        cav_num = len(batch_data['ego']['cav_id_list'])
        with torch.no_grad():
            batch_data = train_utils.to_device(batch_data, device)
            if opt.fusion_method == 'late':
                infer_result = inference_utils_speed.inference_late_fusion(batch_data,
                                                        model,
                                                        opencood_dataset)
            elif opt.fusion_method == 'early':
                infer_result = inference_utils_speed.inference_early_fusion(batch_data,
                                                        model,
                                                        opencood_dataset)
            elif opt.fusion_method == 'intermediate':
                infer_result = inference_utils_speed.inference_intermediate_fusion(batch_data,
                                                                model,
                                                                opencood_dataset)
            elif opt.fusion_method == 'no':
                infer_result = inference_utils_speed.inference_no_fusion(batch_data,
                                                                model,
                                                                opencood_dataset)
            elif opt.fusion_method == 'no_w_uncertainty':
                infer_result = inference_utils_speed.inference_no_fusion_w_uncertainty(batch_data,
                                                                model,
                                                                opencood_dataset)
            elif opt.fusion_method == 'single':
                infer_result = inference_utils_speed.inference_no_fusion(batch_data,
                                                                model,
                                                                opencood_dataset,
                                                                single_gt=True)
            else:
                raise NotImplementedError('Only single, no, no_w_uncertainty, early, late and intermediate'
                                        'fusion is supported.')
            num =num + 1
            average_time+=infer_result['model_time']
            print("time:",infer_result['model_time'])
            pred_box_tensor = infer_result['pred_box_tensor']
            gt_box_tensor = infer_result['gt_box_tensor']
            pred_score = infer_result['pred_score']
            #point_num = infer_result['point_num']
            '''
            if "comm_rates" in infer_result:
               print('comm_rates', infer_result["comm_rates"])
               comm_rates.append(infer_result["comm_rates"].cpu().numpy())
            '''
            if "comm_rates" in infer_result:
                print('comm_rates', infer_result["comm_rates"])
                assert infer_result["comm_rates"] <= 1.0
                comm_rates.append((cav_num-1)*infer_result["comm_rates"].cpu().numpy())
                cav_nums.append(cav_num-1)

            """
            if point_num<5000:
                result_stat = result_stat1
            elif point_num<10000:
                result_stat = result_stat2
            else:
                result_stat = result_stat3
            """


            eval_utils.caluclate_tp_fp(pred_box_tensor,
                                    pred_score,
                                    gt_box_tensor,
                                    result_stat,
                                    0.3)
            eval_utils.caluclate_tp_fp(pred_box_tensor,
                                    pred_score,
                                    gt_box_tensor,
                                    result_stat,
                                    0.5)
            eval_utils.caluclate_tp_fp(pred_box_tensor,
                                    pred_score,
                                    gt_box_tensor,
                                    result_stat,
                                    0.7)
            '''
            if opt.save_npy:
                npy_save_path = os.path.join(opt.model_dir, 'npy')
                if not os.path.exists(npy_save_path):
                    os.makedirs(npy_save_path)
                inference_utils_speed.save_prediction_gt(pred_box_tensor,
                                                gt_box_tensor,
                                                batch_data['ego'][
                                                    'origin_lidar'][0],
                                                i,
                                                npy_save_path)
            '''

            if not opt.no_score:
                infer_result.update({'score_tensor': pred_score})

            if getattr(opencood_dataset, "heterogeneous", False):
                cav_box_np, agent_modality_list = inference_utils_speed.get_cav_box(batch_data)
                infer_result.update({"cav_box_np": cav_box_np, \
                                     "agent_modality_list": agent_modality_list})

            '''
            if (i % opt.save_vis_interval == 0) and (pred_box_tensor is not None or gt_box_tensor is not None):
                vis_save_path_root = os.path.join(opt.model_dir, f'vis_{infer_info}')
                if not os.path.exists(vis_save_path_root):
                    os.makedirs(vis_save_path_root)

                vis_save_path = os.path.join(vis_save_path_root, '3d_%05d.png' % i)
                simple_vis.visualize(infer_result,
                                    batch_data['ego'][
                                        'origin_lidar'][0],
                                    hypes['postprocess']['gt_range'],
                                    vis_save_path,
                                    method='3d',
                                    left_hand=left_hand)
                 
                vis_save_path = os.path.join(vis_save_path_root, 'bev_%05d.png' % i)
                simple_vis.visualize(infer_result,
                                    batch_data['ego'][
                                        'origin_lidar'][0],
                                    hypes['postprocess']['gt_range'],
                                    vis_save_path,
                                    method='bev',
                                    left_hand=left_hand)
                '''
        
        
        torch.cuda.empty_cache()
        print("----------------------------------------------------------------------------------")
        print(" ")
    average_time=average_time/num
    
    #ap30, ap50, ap70 = eval_utils.eval_final_results(result_stat, opt.model_dir, infer_info)
    #ap30_1, ap50_1, ap70_1 = eval_utils.eval_final_results(result_stat1, opt.model_dir, infer_info)
    #ap30_2, ap50_2, ap70_2 = eval_utils.eval_final_results(result_stat2, opt.model_dir, infer_info)
    #ap30_3, ap50_3, ap70_3 = eval_utils.eval_final_results(result_stat3, opt.model_dir, infer_info)
    #print('comm',comm)

    
    if len(comm_rates) > 0:
        #comm_rates = sum(comm_rates) / len(comm_rates)
        comm_rates = sum(comm_rates) / sum(cav_nums)
    else:
        comm_rates = 0.0
    with open(os.path.join(opt.model_dir, 'result_time_{}.txt'.format(opt.result_name)), 'a+') as f:
        f.write('average_time: {:.02f}\n'.format(average_time))
        #f.write('ap30: {:.04f} ap50: {:.04f} ap70: {:.04f} comm_thre: {:.04f} comm_rate: {:.06f}\n'.format(ap30, ap50, ap70, opt.comm_thre, comm_rates))
        #f.write('ap30_1: {:.04f} ap50_1: {:.04f} ap70_1: {:.04f} comm_thre: {:.04f} comm_rate: {:.06f}\n'.format(ap30_1, ap50_1, ap70_1, opt.comm_thre, comm_rates))
        #f.write('ap30_2: {:.04f} ap50_2: {:.04f} ap70_2: {:.04f} comm_thre: {:.04f} comm_rate: {:.06f}\n'.format(ap30_2, ap50_2, ap70_2, opt.comm_thre, comm_rates))
        #f.write('ap30_3: {:.04f} ap50_3: {:.04f} ap70_3: {:.04f} comm_thre: {:.04f} comm_rate: {:.06f}\n'.format(ap30_3, ap50_3, ap70_3, opt.comm_thre, comm_rates))

if __name__ == '__main__':
    main()
