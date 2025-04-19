# ------------------------------------------------------------------------
# Copyright (c) 2022 megvii-model. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from BasicSR (https://github.com/xinntao/BasicSR)
# Copyright 2018-2020 BasicSR Authors
# ------------------------------------------------------------------------
import logging
import torch
from os import path as osp
import os
from basicsr.data import create_dataloader, create_dataset
from basicsr.models import create_model
from basicsr.train import parse_options
from basicsr.utils import (get_env_info, get_root_logger, get_time_str,
                           make_exp_dirs)
from basicsr.utils.options import dict2str
import basicsr.data.sirs_dataset as datasets
from basicsr.data.image_folder import read_fns

def main():
    # parse options, set distributed setting, set ramdom seed
    opt = parse_options(is_train=False)

    torch.backends.cudnn.benchmark = True
    # torch.backends.cudnn.deterministic = True

    # mkdir and initialize loggers
    make_exp_dirs(opt)
    log_file = osp.join(opt['path']['log'],
                        f"test_{opt['name']}_{get_time_str()}.log")
    logger = get_root_logger(logger_name='basicsr', log_level=logging.INFO, log_file=log_file)
    logger.info(get_env_info())
    logger.info(dict2str(opt))
    
    datadir = '/home/chenxiao/NAFNet/datasets/reflection-dataset'

    # create test dataset and dataloader
    test_loaders = []
    testset_name = []
    # test_set_real20 = datasets.DSRTestDataset(os.path.join(datadir, f'test/real20_420'),
    #                                         fns=read_fns('/home/chenxiao/NAFNet/basicsr/data/real_test.txt'), if_align=True)
    # test_loader_real20 = datasets.DataLoader(test_set_real20, batch_size=1, shuffle=False, num_workers=8, pin_memory=True)
    # test_loaders.append(test_loader_real20)
    # testset_name.append('real20')

    # test_set_solidobject = datasets.DSRTestDataset(os.path.join(datadir, 'test/SIR2/SolidObjectDataset'), if_align=True)
    # test_loader_solidobject = datasets.DataLoader(test_set_solidobject, batch_size=1, shuffle=False, num_workers=8, pin_memory=True)
    # test_loaders.append(test_loader_solidobject)
    # testset_name.append('solidobject')

    test_set_postcard = datasets.DSRTestDataset(os.path.join(datadir, 'test/SIR2/PostcardDataset'), if_align=True, test_flag=True)
    test_loader_postcard = datasets.DataLoader(test_set_postcard, batch_size=1, shuffle=False, num_workers=8, pin_memory=True)
    test_loaders.append(test_loader_postcard)
    testset_name.append('postcard')
    
    # test_automatic_set_postcard = datasets.DSRTestDataset(os.path.join(datadir, 'automatic_test/SIR2/PostcardDataset'), if_align=True)
    # test_loader_postcard = datasets.DataLoader(test_automatic_set_postcard, batch_size=1, shuffle=False, num_workers=8, pin_memory=True)
    # test_loaders.append(test_loader_postcard)
    # testset_name.append('postcard')

    # test_set_wild = datasets.DSRTestDataset(os.path.join(datadir, 'test/SIR2/WildSceneDataset'), if_align=True, test_flag=True)
    # test_loader_wild = datasets.DataLoader(test_set_wild, batch_size=1, shuffle=False, num_workers=8, pin_memory=True)
    # test_loaders.append(test_loader_wild)          
    # testset_name.append('wildscene')

    # create model
    model = create_model(opt)

    for test_loader, test_set_name in zip(test_loaders, testset_name):
        logger.info(f'Testing {test_set_name}...')
        rgb2bgr = opt['val'].get('rgb2bgr', True)
        # wheather use uint8 image to compute metrics
        use_image = opt['val'].get('use_image', True)
        model.validation(
            test_loader,
            current_iter=opt['name'],
            tb_logger=None,
            save_img=opt['val']['save_img'],
            rgb2bgr=rgb2bgr, use_image=use_image)


if __name__ == '__main__':
    main()
# python basicsr/test.py -opt options/test/REDS/NAFNet-width64.yml