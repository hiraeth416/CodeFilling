from matplotlib import markers
import matplotlib.pyplot as plt
plt.style.use('ggplot')
import numpy as np
import math
import os
from collections import OrderedDict

feat_size_dict ={
    'opv2v': 100*352*64,
    'dair': 100*252*64,
    'v2xsim2': 80*80*64
}

def load_data(model_dir, file_name='result.txt', dataset='opv2v'):
    feat_size = feat_size_dict[dataset]
    comm_thre = []
    comm_rate = []
    APs = [[],[],[]]
    result_dir = os.path.join('/GPFS/public/sifeiliu/OpenCOODv2_new/opencood/logs_HEAL/', '{}/{}'.format(model_dir, file_name))
    with open(result_dir, 'r') as f:
        data = f.readline().strip()
        while(len(data)>0):
            data = data.split(' ')[1::2]
            data = [float(x) for x in data]
            comm_rate.append(data[-1])
            comm_thre.append(data[-2])
            APs[0].append(data[0])
            APs[1].append(data[1])
            APs[2].append(data[2])
            data = f.readline().strip()
    idx = np.argsort(comm_rate)
    comm_rate = np.array(comm_rate)[idx]
    updated_comm_rate = []
    for x in comm_rate:
        if x != 0:
            updated_comm_rate.append(np.log2(x*feat_size))
        else:
            updated_comm_rate.append(0)
    comm_rate = np.array(updated_comm_rate)
    comm_thre = np.array(comm_thre)[idx]
    APs = np.array(APs)[:,idx]
    return comm_thre, comm_rate, APs

def load_track_data(model_dir, comm_rate_dict, file_name='result_tracking.txt'):
    data_list = []
    result_dir = os.path.join(os.path.dirname(__file__), '/GPFS/public/sifeiliu/OpenCOODv2_new/opencood/logs_HEAL/{}/{}'.format(model_dir, file_name))
    with open(result_dir, 'r') as f:
        data = f.readline().strip()
        while(len(data)>0):
            data = data.split(' ')
            data = [float(x) for x in data]
            data_list.append(np.array(data)[None,])
            data = f.readline().strip()
    data_list = np.concatenate(data_list, axis=0)
    comm_thre = data_list[:,0]
    metrics = data_list[:,1:].T
    idx = np.argsort(comm_thre)
    comm_thre = comm_thre[idx]
    comm_rate = []
    keep_idx = []
    for i, x in enumerate(comm_thre):
        if x in comm_rate_dict:
            comm_rate.append(comm_rate_dict[x]['rate'])
            keep_idx.append(i)
    comm_rate = np.array(comm_rate)
    keep_idx = np.array(keep_idx)
    metrics = metrics[:,idx][:,keep_idx]
    comm_thre = comm_thre[keep_idx]
    return comm_thre, comm_rate, metrics


def list2dict(comm_thre, comm_rate, metrics):
    data_dict = {}
    for i, thre in enumerate(comm_thre):
        data_dict[thre] = {}
        data_dict[thre]['rate'] = comm_rate[i]
        data_dict[thre]['metrics'] = metrics[:,i]
    return data_dict