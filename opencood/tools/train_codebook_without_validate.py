# -*- coding: utf-8 -*-
# Author: Runsheng Xu <rxx3386@ucla.edu>
# License: TDG-Attribution-NonCommercial-NoDistrib


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

    print('Dataset Building')
    opencood_train_dataset = build_dataset(hypes, visualize=False, train=True)
    opencood_validate_dataset = build_dataset(hypes, visualize=False, train=False)

    train_loader = DataLoader(opencood_train_dataset,
                              batch_size=hypes['train_params']['batch_size'],
                              num_workers=8,
                              collate_fn=opencood_train_dataset.collate_batch_train,
                              shuffle=True,
                              pin_memory=True,
                              drop_last=True,
                              prefetch_factor=2)
    val_loader = DataLoader(opencood_validate_dataset,
                            batch_size=hypes['train_params']['batch_size'],
                            num_workers=8,
                            collate_fn=opencood_train_dataset.collate_batch_train,
                            shuffle=True,
                            pin_memory=True,
                            drop_last=True,
                            prefetch_factor=2)

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
        init_epoch, model = train_utils.load_saved_model(saved_path, model)
        lowest_val_epoch = init_epoch
        scheduler = train_utils.setup_lr_schedular(hypes, optimizer, init_epoch=init_epoch)
        print(f"resume from {init_epoch} epoch.")

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
    supervise_single_flag = False if not hasattr(opencood_train_dataset, "supervise_single") else opencood_train_dataset.supervise_single
    # used to help schedule learning rate

    for epoch in range(init_epoch, max(epoches, init_epoch)):
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
            ouput_dict = model(i, batch_data['ego'])
            
            final_loss = ouput_dict['codebook_loss']
            final_loss.requires_grad_(True) 
            print("epoch", epoch, " codebook_loss:", final_loss)

            # back-propagation
            final_loss.backward()
            optimizer.step()

            torch.cuda.empty_cache()

        if epoch % hypes['train_params']['eval_freq'] == 0:
            
            # lowest val loss
            torch.save(model.state_dict(),
                    os.path.join(saved_path,
                                'net_epoch_bestval_at%d.pth' % (epoch + 1)))
            if lowest_val_epoch != -1 and os.path.exists(os.path.join(saved_path,
                                'net_epoch_bestval_at%d.pth' % (lowest_val_epoch))):
                os.remove(os.path.join(saved_path,
                                'net_epoch_bestval_at%d.pth' % (lowest_val_epoch)))
            lowest_val_epoch = epoch + 1

        if epoch % hypes['train_params']['save_freq'] == 0:
            torch.save(model.state_dict(),
                       os.path.join(saved_path,
                                    'net_epoch%d.pth' % (epoch + 1)))
        scheduler.step(epoch)

        opencood_train_dataset.reinitialize()

    print('Training Finished, checkpoints saved to %s' % saved_path)

    run_test = True
    if run_test:
        fusion_method = opt.fusion_method
        cmd = f"python opencood/tools/inference.py --model_dir {saved_path} --fusion_method {fusion_method}"
        print(f"Running command: {cmd}")
        os.system(cmd)

if __name__ == '__main__':
    main()
