# -*- coding: utf-8 -*-
# Author: Runsheng Xu <rxx3386@ucla.edu>
# License: TDG-Attribution-NonCommercial-NoDistrib


import argparse
import os
import statistics
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

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
    opt = parser.parse_args()
    return opt


def main():
    opt = train_parser()
    hypes = yaml_utils.load_yaml(opt.hypes_yaml, opt)

    print('Dataset Building')
    opencood_train_dataset = build_dataset(hypes, visualize=False, train=True)
    opencood_validate_dataset = build_dataset(hypes,
                                              visualize=False,
                                              train=False)

    train_loader = DataLoader(opencood_train_dataset,
                              batch_size=hypes['train_params']['batch_size'],
                              num_workers=8,
                              collate_fn=opencood_train_dataset.collate_batch_train,
                              shuffle=True,
                              pin_memory=True,
                              drop_last=True,
                              prefetch_factor=2)

    print('Creating Model')
    get_feature={'get_feature': 'True'}
    hypes['model']['args'].update(get_feature)
    model = train_utils.create_model(hypes)
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

    # we assume gpu is necessary
    if torch.cuda.is_available():
        model.to(device)
        

    print('Training start')
    supervise_single_flag = False if not hasattr(opencood_train_dataset, "supervise_single") else opencood_train_dataset.supervise_single
    # used to help schedule learning rate

    epoch = 0
    '''
    saved_path = "/GPFS/data/sifeiliu/logs/get_feature/"
    saved_path = os.path.join(saved_path, 'segnum%d' % (hypes['model']['args']['codebook']['seg_num']))    
    saved_path = os.path.join(saved_path, 'segnum%d' % (hypes['model']['args']['codebook']['dict_size']))
    '''
    for param_group in optimizer.param_groups:
        print('learning rate %f' % param_group["lr"])
    for i, batch_data in enumerate(train_loader):
        if batch_data is None or batch_data['ego']['object_bbx_mask'].sum()==0:
            continue
        # the model will be evaluation mode during validation
        model.train()
        model.zero_grad()
        optimizer.zero_grad()
        batch_data = train_utils.to_device(batch_data, device)
        batch_data['ego']['epoch'] = epoch
        save_path = "/GPFS/rhome/sifeiliu/OpenCOODv2/opencood/logs/feature_folder/"
        print("batchdata", i)
        np.save(os.path.join(save_path,'batchdata%d.npy' % (i)), batch_data)
        continue #here need to change
        ouput_dict = model(i, batch_data['ego'])
       
        
        # back-propagation
        optimizer.step()
        torch.cuda.empty_cache()
    '''
    if epoch % hypes['train_params']['save_freq'] == 0:
            torch.save(model.state_dict(),
                       os.path.join(saved_path,
                                    'net_epoch%d.pth' % (epoch + 1)))
    '''

        

print('Training Finished')

if __name__ == '__main__':
    main()
