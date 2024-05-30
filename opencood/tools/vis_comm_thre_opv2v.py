from matplotlib import markers
import matplotlib.pyplot as plt
plt.style.use('ggplot')
import numpy as np
import math
import os
from collections import OrderedDict
from vis_comm_utils import load_data, load_track_data, list2dict
import argparse

det_metrics = ['AP30', 'AP50', 'AP70']
det_metrics_c = ['AP30-c', 'AP50-c', 'AP70-c']
tracking_metrics = 'MOTA,MOTP,HOTA,DetA,AssA,DetRe,DetPr,AssRe,AssPr,LocA'.split(',')


def train_parser():
    parser = argparse.ArgumentParser(description="synthetic data generation")
    parser.add_argument('--save_model_dir', default='',
                        help='Continued training path')
    opt = parser.parse_args()
    return opt

opt = train_parser()
save_model_dir = opt.save_model_dir
#save_model_dir = 'HEAL_opv2v_m1_pointpillar_m2_lsseff_end2end_140.8_40_2023_07_05_21_30_17'
#save_model_dir = "OPV2V_heter_resnet1x1_layer1_dairv2xnofreeze"
#modal = ['lidar_only', 'camera_only', 'egolidar_othercamera', 'egocamera_otherlidar']
modal = ['lidar_only', 'camera_only', 'camera_only_lidar_range', 'egolidar_othercamera', 'egocamera_otherlidar', 'egorandom_ratio0.5']
for j in modal:
    s_det_comm_thre, s_det_comm_rate, s_det_metrics = load_data(save_model_dir, 'result_{}.txt'.format(j))
    s_det_data_dict = list2dict(s_det_comm_thre, s_det_comm_rate, s_det_metrics)
    #s_det_comm_thre_codebook, s_det_comm_rate_codebook, s_det_metrics_codebook = load_data(save_model_dir, 'result_{}_codebook.txt'.format(j))
    #s_det_data_dict_codebook = list2dict(s_det_comm_thre_codebook, s_det_comm_rate_codebook, s_det_metrics_codebook)
    '''s_track_comm_thre, s_track_comm_rate, s_track_metrics = load_track_data(save_model_dir, s_det_data_dict, file_name='result_tracking_skip1.txt')
    
    st_model_dir = 'opv2v_point_pillar_lidar_where2comm_kalman_ms_tune_allo_2023_05_22_15_01_29'
    st_det_comm_thre, st_det_comm_rate, st_det_metrics = load_data(st_model_dir, 'result_comm_skip2.txt')
    st_det_data_dict = list2dict(st_det_comm_thre, st_det_comm_rate, st_det_metrics)
    st_track_comm_thre, st_track_comm_rate, st_track_metrics = load_track_data(st_model_dir, st_det_data_dict, file_name='result_tracking_skip2.txt')
    '''
    fontsize = 20
    label_size = 18
    legend_size = 10
    tick_size = 20
    labelsize = 20
    point_size = 100
    # figsize=(7,6)
    # figsize=(6.7,5.5)
    figsize=(6,5)
    params = {
        # 'legend.fontsize': 'x-large',
            # 'figure.figsize': (9, 7),
             'axes.labelsize': 'x-large',
             'axes.titlesize':'x-large',
             'xtick.labelsize':'x-large',
             'ytick.labelsize':'x-large'}
    plt.rcParams.update(params)
    plt.tick_params(labelsize=labelsize)
    colors = ['mediumpurple', 'red', 'steelblue', 'darkgreen', 'orange']
    
    #######################  Detection Performance #######################
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111)
    for i, metric in enumerate(det_metrics):
        plt.plot(s_det_comm_rate, s_det_metrics[i], label=metric, linewidth=2, c=colors[0])
        #plt.plot(s_det_comm_rate_codebook, s_det_metrics_codebook[i], label=det_metrics_c[i], linewidth=2, c=colors[1])
        #plt.plot(st_det_comm_rate, st_det_metrics[i], label='ST', linewidth=3)
        # plt.plot(comm_rate_ST2_late, APs_ST2_late[i], label='STmodel_ST_box', linewidth=3, linestyle='dashed', c=colors[0])
        plt.title(j)
        ratio = 1.0
        xleft, xright = ax.get_xlim()
        ybottom, ytop = ax.get_ylim()
        ax.set_aspect(abs((xright-xleft)/(ybottom-ytop))*ratio)
        plt.ylabel(metric, size=label_size)
        plt.xlabel('Communication cost', size=label_size)
        plt.legend(prop={'size': legend_size})
        plt.savefig('/GPFS/public/sifeiliu/OpenCOODv2_new/opencood/logs_HEAL/{}/for_{}_heter_{}.png'.format(save_model_dir, metric, j))
####################################################################

#######################  Tracking Performance #######################
'''for i, metric in enumerate(tracking_metrics):
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111)

    plt.plot(s_track_comm_rate, s_track_metrics[i], label='S', linewidth=3)
    plt.plot(st_track_comm_rate, st_track_metrics[i], label='ST', linewidth=3)
    # plt.plot(comm_rate_ST2_late, APs_ST2_late[i], label='STmodel_ST_box', linewidth=3, linestyle='dashed', c=colors[0])
    
    plt.title(metric)
    ratio = 1.0
    xleft, xright = ax.get_xlim()
    ybottom, ytop = ax.get_ylim()
    ax.set_aspect(abs((xright-xleft)/(ybottom-ytop))*ratio)
    plt.ylabel(metric, size=label_size)
    plt.xlabel('Communication cost', size=label_size)
    plt.legend(prop={'size': legend_size})
    plt.savefig('OPV2V/OPV2V_Ablation_CommCost_{}.png'.format(metric))'''
####################################################################