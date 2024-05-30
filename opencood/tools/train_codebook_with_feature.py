# -*- coding: utf-8 -*-
# Author: Runsheng Xu <rxx3386@ucla.edu>
# License: TDG-Attribution-NonCommercial-NoDistrib

import glob
import importlib
import yaml
import re
import argparse
import os
import statistics

import torch
from torch.utils.data import DataLoader, Subset
from tensorboardX import SummaryWriter

import opencood.hypes_yaml.yaml_utils as yaml_utils
from opencood.tools import train_utils
from opencood.data_utils.datasets import build_dataset

from icecream import ic


def train_parser():
    parser = argparse.ArgumentParser(description="synthetic data generation")
    parser.add_argument("--hypes_yaml", "-y", type=str, required=True,
                        help='data generation yaml file needed ')
    parser.add_argument('--model_dir', default='',
                        help='Continued training path')
    parser.add_argument('--fusion_method', '-f', default="intermediate",
                        help='passed to inference.')
    parser.add_argument('--lidar_ratio', default=0.5,
                        help='lidar ratio when using heter dataset')      
    parser.add_argument('--ego_modality', default='random',
                        help='choose the ego modality of dataset')      
    parser.add_argument('--seg_num', default=2,
                        help='the seg_num of codebook')                        
    parser.add_argument('--dict_size', default=64,
                        help='the real dict_size of codebook will be [dict_size, dict_size, dict_size]')            
    opt = parser.parse_args()
    return opt


def main():
    opt = train_parser()
    hypes = yaml_utils.load_yaml(opt.hypes_yaml, opt)
    if 'heter' in hypes:
        print("lidar_ratio:{}".format(opt.lidar_ratio))
        print("ego_modality:{}".format(opt.ego_modality))
        lidar_ratio = float(opt.lidar_ratio)
        hypes['heter']['lidar_ratio'] = lidar_ratio
        hypes['heter']['ego_modality'] = opt.ego_modality

    '''
    if 'codebook' in hypes['model']:
        print("seg_num:{}".format(opt.seg_num))
        print("dict_size:{}".format(opt.dict_size))
        hypes['model']['codebook']['seg_num'] = seg_num
        hypes['model']['codebook']['dict_size'] = opt.dict_size
    '''

    
    print('Creating Model')
    model = train_utils.create_model(hypes)
        # print training parameters
    print('----------- Training Parameters -----------')
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(name, param.data.shape)
    print('----------- Training Parameters -----------')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # record lowest validation loss checkpoint.
    lowest_val_loss = 1e5
    lowest_val_epoch = -1

    # define the loss
    criterion = train_utils.create_loss(hypes)

    # optimizer setup
    optimizer = train_utils.setup_optimizer(hypes, model)
    # lr scheduler setup
    

    # if we want to train from last checkpoint.
    if opt.model_dir:
        saved_path = opt.model_dir
        file_list = glob.glob(os.path.join(saved_path, '*epoch*.pth'))
        if file_list:
            epochs_exist = [0]
            for file_ in file_list:
                if 'codebook' in file_:
                    result = re.findall(".*epoch(.*).pth.*", file_)
                    epochs_exist.append(int(result[0]))
            initial_ep = max(epochs_exist)
        else:
            initial_ep = 0
        if initial_ep > 0:
          print('resuming by loading epoch %d' % initial_ep)
          model.load_state_dict(torch.load(
              os.path.join(saved_path,
                         'codebook_epoch%d.pth' % initial_ep), map_location='cpu'), strict=False)

        scheduler = train_utils.setup_lr_schedular(hypes, optimizer, init_epoch=initial_ep)
        print(f"resume from {initial_ep} epoch.")

    else:
        init_epoch = 0
        # if we train the model from scratch, we need to create a folder
        # to save the model,
        saved_path = train_utils.setup_train(hypes)
        scheduler = train_utils.setup_lr_schedular(hypes, optimizer)

    # we assume gpu is necessary
    if torch.cuda.is_available():
        model.to(device)
        
    # record training
    writer = SummaryWriter(saved_path)

    print('Training start')
    epoches = hypes['train_params']['epoches']
    # used to help schedule learning rate
    
    #optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    epoch = 20
    for ep in range(initial_ep, 40):
        for param_group in optimizer.param_groups:
            param_group["lr"] = param_group["lr"] / 100
            print('learning rate %f' % param_group["lr"])
        for i in range(3186):
            # the model will be evaluation mode during validation
            if i%8 != 0:
                continue
            model.train()
            model.zero_grad()
            optimizer.zero_grad()
            ouput_dict = model(i)
            
            final_loss = ouput_dict['codebook_loss']
            final_loss.requires_grad_(True) 
            print("epoch", ep, "iter", i, "codebook_loss:", final_loss)

            # back-propagation
            final_loss.backward()
            optimizer.step()
        scheduler.step(ep)

        torch.cuda.empty_cache()


        if ep % hypes['train_params']['save_freq'] == 0:
            torch.save(model.state_dict(),
                       os.path.join(saved_path,
                                    'codebook_epoch%d.pth' % (ep)))
            

        #opencood_train_dataset.reinitialize()

    print('Training Finished, checkpoints saved to %s' % saved_path)


if __name__ == '__main__':
    main()
