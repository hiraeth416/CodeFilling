from matplotlib import markers
import matplotlib.pyplot as plt
plt.style.use('ggplot')
import numpy as np
import math
import os
from collections import OrderedDict
import sys
import argparse
from vis_utils import load_data, load_track_data, load_baseline, list2dict

det_metrics_name = ['AP30', 'AP50', 'AP70']
tracking_metrics_name = 'MOTA,MOTP,HOTA,DetA,AssA,DetRe,DetPr,AssRe,AssPr,LocA'.split(',')
mode = 'comm'
codebook_config = {'d_size': 256, 'd_seg': 2, 'd_residual': 3}

def train_parser():
    parser = argparse.ArgumentParser(description="synthetic data generation")
    parser.add_argument('--model_dir', default='',
                        help='Continued training path')
    opt = parser.parse_args()
    return opt


opt = train_parser()
model_dir = 'where2comm_MoreAgents_db' if opt.model_dir is None else opt.model_dir
model_dir = os.path.join(os.path.dirname(__file__), '../../logs/{}/'.format(opt.model_dir))


############# Solver Ablation ##########
solver_thre = [0.5, 0.8, 1.0, 2.0, 3.0]
solver_thre = [1.0]
cav_num = [5, 10]
# cav_num = [5]
results_dict = {}
for cav in cav_num:
    baseline_dir = 'result_baseline_3.0_cav{}.txt'.format(cav)
    abs_dir = os.path.join(model_dir, baseline_dir)
    print(abs_dir)
    if os.path.exists(abs_dir):
        # if use codebook: load_data(abs_dir, mode, codebook=codebook_config)
        det_comm_thre, det_comm_rate, det_metrics = load_data(abs_dir, mode)
        results_dict['baseline_cav{}'.format(cav)] = {
            'det_comm_thre': det_comm_thre,
            'det_comm_rate': det_comm_rate,
            'det_metrics': det_metrics
        }
    for thre in solver_thre:
        baseline_dir = 'result_solver_thre{}_cav{}.txt'.format(thre, cav)
        abs_dir = os.path.join(model_dir, baseline_dir)
        print(abs_dir)
        if os.path.exists(abs_dir):
            det_comm_thre, det_comm_rate, det_metrics = load_data(abs_dir, mode)
            results_dict['thre{}_cav{}'.format(thre, cav)] = {
                'det_comm_thre': det_comm_thre,
                'det_comm_rate': det_comm_rate,
                'det_metrics': det_metrics
            }

fontsize = 20
label_size = 18
legend_size = 10
tick_size = 20
labelsize = 20
point_size = 100
# figsize=(7,6)
# figsize=(6.7,5.5)
figsize=(5.5,5)
params = {
    # 'legend.fontsize': 'x-large',
        # 'figure.figsize': (9, 7),
         'axes.labelsize': 'x-large',
         'axes.titlesize':'x-large',
         'xtick.labelsize':'x-large',
         'ytick.labelsize':'x-large'}
plt.rcParams.update(params)
plt.tick_params(labelsize=labelsize)
# colors = ['mediumpurple', 'red', 'steelblue', 'darkgreen', 'orange']
colors = ['red', 'royalblue', 'mediumorchid', 'seagreen', 'darkorange', 'crimson',
          'dodgerblue', 'mediumseagreen', 'darkviolet', 'firebrick', 'goldenrod',
          'mediumblue', 'mediumvioletred', 'forestgreen', 'darkgoldenrod', 'indianred']

#######################  Detection Performance #######################
for i, metric in enumerate(det_metrics_name):
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111)
    for name, result in results_dict.items():
        c = colors[1] if '5' in name else colors[0]
        line = 'dashed' if 'baseline' in name else "solid"
        det_comm_thre, det_comm_rate, det_metrics = result.values()
        det_data_dict = list2dict(det_comm_thre, det_comm_rate, det_metrics)
        plt.plot(det_comm_rate, det_metrics[i], label=name, linewidth=2, linestyle=line, color=c)

    plt.title(metric)
    ratio = 1.0
    xleft, xright = ax.get_xlim()
    ybottom, ytop = ax.get_ylim()
    ax.set_aspect(abs((xright-xleft)/(ybottom-ytop))*ratio)
    plt.ylabel(metric, size=label_size)
    plt.xlabel('Communication cost', size=label_size)
    # handles, labels = plt.gca().get_legend_handles_labels()
    # handles = [handles[i] for i in [1,0,2,3]]
    # labels = [labels[i] for i in [1,0,2,3]]
    # plt.legend(handles, labels, prop={'size': legend_size})
    plt.legend(prop={'size': legend_size})
    plt.savefig('OPV2V_Ablation_Comm_{}.png'.format(metric))
####################################################################
