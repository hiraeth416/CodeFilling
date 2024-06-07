import xlwt
import xlrd
from matplotlib import markers
import matplotlib.pyplot as plt
plt.style.use('ggplot')
import numpy as np
import math
import os
from collections import OrderedDict
from vis_comm_utils import load_data, load_track_data, list2dict

feat_size_dict ={
    'opv2v': 100*352*64,
    'dair': 100*252*64,
    'v2xsim2': 80*80*64
}

def calc_cost(x, dataset='opv2v', codebook=False, r=3, s=256, segnum=2, channel=64):
    feat_size = feat_size_dict[dataset]
    if x != 0:
        if not codebook:
            comm_rate_calc=np.log2(x*feat_size*(channel/64.0)*32/8)
        else:
            comm_rate_calc=np.log2(x*(feat_size/64)*r*segnum*np.log2(s)/8)
    else:
        comm_rate_calc=0
    if comm_rate_calc < 0:
        comm_rate_calc=0
    return comm_rate_calc

def codebook_parameter(model_dir):
    if model_dir == 'codebook_dairv2x_size256':
        return 256, 2, 2, 64, 'dair' 
    if model_dir == 'codebook_dairv2x_size128':
        return 128, 2, 2, 64, 'dair' 
    if model_dir == 'codebook_dairv2x_size64':
        return 64, 2, 2, 64, 'dair' 
    if model_dir == 'codebook_dairv2x_size32':
        return 32, 2, 2, 64, 'dair' 
    if model_dir == 'codebook_dairv2x_size16':
        return 16, 2, 2, 64, 'dair' 
    if model_dir == 'codebook_dairv2x_size4':
        return 4, 2, 2, 64, 'dair' 
    if model_dir == 'codebook_dairv2x_size32_r1':
        return 32, 1, 2, 64, 'dair' 
    if model_dir == 'codebook_dict_size256':
        return 256, 3, 2, 64, 'opv2v' 
    if model_dir == 'codebook_dict_size128':
        return 128, 3, 2, 64, 'opv2v' 
    if model_dir == 'codebook_dict_size64':
        return 64, 3, 2, 64, 'opv2v' 
    if model_dir == 'codebook_dict_size32':
        return 32, 3, 2, 64, 'opv2v' 
    if model_dir == 'codebook_dict_size16':
        return 16, 3, 2, 64, 'opv2v' 
    if model_dir == 'codebook_dict_size4':
        return 4, 1, 1, 64, 'opv2v' 
    if model_dir == 'codebook_dict_size32_r2':
        return 32, 2, 2, 64, 'opv2v' 
    if model_dir == 'codebook_dict_size32_r1':
        return 32, 1, 2, 64, 'opv2v' 

    


wb = xlwt.Workbook()
ws = wb.add_sheet('SOTA-DAIRV2X')

#feat_size = feat_size_dict[dataset]
comm_thre = []
comm_rate = []
APs = [[],[],[]]
#model_dir=['codebook_dairv2x_size256', 'codebook_dairv2x_size128', 'codebook_dairv2x_size64', 'codebook_dairv2x_size32', 'codebook_dairv2x_size16', 'codebook_dairv2x_size4', 'codebook_dairv2x_size32_r1']
#model_dir=['codebook_dict_size256', 'codebook_dict_size128', 'codebook_dict_size64', 'codebook_dict_size32', 'codebook_dict_size16', 'codebook_dict_size4', 'codebook_dict_size32_r2', 'codebook_dict_size32_r1']
#model_dir=['Dairv2x_single', 'Dairv2x_late', 'Dairv2x_HMVIT', 'Dairv2x_DiscoNet', 'Dairv2x_AttFuse', 'Dairv2x_V2VNet', 'Dairv2x_cobevt', 'Dairv2x_V2X-ViT', 'FedHCP_dairv2x_m1_m2_end2end_singlesup_2023_08_12_08_51_04']
#model_dir=['OPV2V_single', 'OPV2V_late', 'OPV2V_HMVIT', 'OPV2V_DiscoNet', 'OPV2V_Attfuse', 'OPV2V_V2VNet', 'OPV2V_CoBEVT', 'OPV2V_V2X-ViT', 'codebook_notopv2v']
model_dir=['Dairv2x_single', 'Dairv2x_late', 'Dairv2x_hmvit_new', 'Dairv2x_DiscoNet', 'Dairv2x_attfuse_new', 'Dairv2x_V2VNet_new', 'Dairv2x_cobevt_new', 'Dairv2x_v2xvit_new', 'Dairv2x_where2comm']
#modal = ['lidar_only', 'camera_only', 'egorandom_ratio0.5']
modal = ['lidar_only', 'camera_only', 'egocamera_otherlidar']
#
cnt_i = 0

for i in model_dir:
    save_dir = os.path.join("/GPFS/rhome/sifeiliu/OpenCOODv2_new/opencood/logs_HEAL", i)
    print(i)
    ws.write(cnt_i, 0, i)
    cnt_j = 1
    add_i = 0
    for j in modal:
        result_dir = os.path.join(os.path.dirname(__file__), '{}/{}'.format(save_dir, 'result_{}.txt'.format(j)))
        with open(result_dir, 'r') as f:
            data = f.readline().strip()
            #print(type(data))
            temp_i = cnt_i
            while(len(data)>0):                
                #print(type(data))
                data = data.split(' ')[1::2]
                data = [float(x) for x in data]
                if(data[-2]!=0.0 and data[-2]!=0.01 and data[-2]!=0.1 and data[-2]!=0.2 and data[-2]!=0.6 and data[-2]!=1.0):
                    data = f.readline().strip()
                    continue
                #s, r, segnum, channel, dataset = codebook_parameter(i)
                #comm_costs = calc_cost(data[-1], dataset=dataset, codebook=True, r=r, s=s, segnum=segnum, channel=channel)
                print("data:", data[-2])
                print("get!!")
                ws.write(temp_i, cnt_j, data[0])
                ws.write(temp_i, cnt_j+1, data[1])
                ws.write(temp_i, cnt_j+2, data[2])
                ws.write(temp_i, cnt_j+3, data[-2])
                #ws.write(temp_i, cnt_j+4, comm_costs)
                temp_i += 1
                add_i += 1
                
                #comm_rate.append(data[-1])
                #comm_thre.append(data[-2])
                #APs[0].append(data[0])
                #APs[1].append(data[1])
                #APs[2].append(data[2])
                data = f.readline().strip()
        '''
        idx = np.argsort(comm_rate)
        comm_rate = np.array(comm_rate)[idx]
        updated_comm_rate = []
        for x in comm_rate:
            if x != 0:
                if not codebook:
                    comm_rate_calc=np.log2(x*feat_size*(channel/64.0)*32/8)
                else:
                    comm_rate_calc=np.log2(x*(feat_size/64)*r*segnum*np.log2(s)/8)
            else:
                comm_rate_calc=0
            if comm_rate_calc < 0:
                comm_rate_calc=0
            updated_comm_rate.append(comm_rate_calc)
        comm_rate = np.array(updated_comm_rate)
        comm_thre = np.array(comm_thre)[idx]
        APs = np.array(APs)[:,idx]
        '''
        cnt_j+=5
    cnt_i +=add_i//3




wb.save('/GPFS/rhome/sifeiliu/OpenCOODv2_new/opencood/logs_HEAL/SOTA-DAIRV2X.xls')

    
