import argparse
import os
import time
from pathlib import Path
import random

import numpy as np
import torch
from torch.backends import cudnn
import gc

from Models.Server_DF import Server_DF
from config.cifar100_delay import get_args_parser

from timm.models import create_model
from timm.scheduler import create_scheduler
from timm.optim import create_optimizer
import Models.vision_transformer
from data.homeoffice_subset_spilter import officehome_spliter
from data.i5datasets import i5datasets
from data.iCIFAR100c import iCIFAR100c

torch.set_printoptions(threshold=float('inf'))

from data.cifar100_subset_spliter import cifar100_Data_Spliter
from data.five_datasets_subset_spliter import five_datasets_Data_Spliter
import warnings
warnings.filterwarnings("ignore")

def main(args):
    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    pretrained_cfg = create_model(args.model).default_cfg
    pretrained_cfg['file']='pretrain_model/ViT-B_16.npz'
    print(pretrained_cfg)

    cudnn.benchmark = True
    args.data_name =args.data_name[0]
    if args.data_name=='cifar100':
        client_data,client_mask= cifar100_Data_Spliter(client_num=args.client_num,task_num=args.task_num,private_class_num=args.private_class_num,input_size=args.input_size).random_split()
        surro_data,test_data = cifar100_Data_Spliter(client_num=args.client_num,task_num=args.task_num,
                                                     private_class_num=args.private_class_num,input_size=args.input_size).process_testdata(args.surrogate_num)
        surro_data = iCIFAR100c(subset=surro_data)
        args.nb_classes = 100
    elif args.data_name=='5datasets':
        data_spliter = five_datasets_Data_Spliter(client_num=args.client_num, task_num=args.task_num,
                                                         private_class_num=args.private_class_num,
                                                         input_size=args.input_size)
        client_data, client_mask = data_spliter.random_split()
        surro_data, test_data = data_spliter.process_testdata(args.surrogate_num)
        surro_data = i5datasets(surro_data)
        args.nb_classes = 50
    else:
        data_spliter = officehome_spliter(client_num=args.client_num, task_num=args.task_num,
                                                          private_class_num=args.private_class_num,
                                                          input_size=args.input_size)
        args.nb_classes = 65

    print(f"Creating original model: {args.model}")
    original_model = create_model(
        args.model,
        pretrained=True,
        pretrained_cfg = pretrained_cfg,
        num_classes=args.nb_classes,
        drop_rate=args.drop,
        drop_path_rate=args.drop_path,
        drop_block_rate=None,
        # pretrained_cfg_overlay=dict(file='pretrain_model/pytorch_model.bin')
        # checkpoint_path='pretrain_model/original_model.pth'
        )

    print(f"Creating model: {args.model}")
    model = create_model(
        args.model,
        pretrained=True,
        pretrained_cfg=pretrained_cfg,
        num_classes=args.nb_classes,
        drop_rate=args.drop,
        drop_path_rate=args.drop_path,
        drop_block_rate=None,
        prompt_length=args.length,
        embedding_key=args.embedding_key,
        prompt_init=args.prompt_key_init,
        prompt_pool=args.prompt_pool,
        prompt_key=args.prompt_key,
        pool_size=args.size,
        top_k=args.top_k,
        batchwise_prompt=args.batchwise_prompt,
        prompt_key_init=args.prompt_key_init,
        head_type=args.head_type,
        use_prompt_mask=args.use_prompt_mask,
        e_prompt_layer_idx=args.e_prompt_layer_idx,
        method = args.method,
        # pretrained_cfg_overlay=dict(file='pretrain_model/pytorch_model.bin')
        # checkpoint_path='pretrain_model/model.pth'
       )

    # model.prompt_length = 10
    # model.top_k = 3
    # model.pool_size = 10
    #
    # # model.init_pos_embedding()
    # # model.init_weights('')
    #
    # model.init_global_prompt()
    model.init_local_prompt()
    original_model.to(device)
    model.to(device)

    if args.freeze:
        # all parameters are frozen for original vit model
        for p in original_model.parameters():
            p.requires_grad = False

        # freeze args.freeze[blocks, patch_embed, cls_token] parameters
        for n, p in model.named_parameters():
            if n.startswith(tuple(args.freeze)):
                p.requires_grad = False

    print(args)
    for n, p in model.named_parameters():
        if p.requires_grad == True:
            print(n)
    #id,origin_model,model,client_num,task_num,subset,class_mask,lr,global_epoch,local_epoch,batch_size,device,method,threshold,surrogate_data,test_data):
    if args.data_name=='cifar100' or args.data_name=='5datasets':
        myServer = Server_DF(id='Server',origin_model=original_model,model=model,client_num=args.client_num,task_num=args.task_num,
                         subset=client_data,class_mask=client_mask,lr=args.lr,global_epoch=args.global_epoch,local_epoch=args.local_epoch,
                         batch_size=args.batch_size,device=args.device,method=args.method,threshold=args.threshold,
                         surrogate_data=surro_data,test_data=None,args=args)
    else:
        myServer = Server_DF(id='Server',origin_model=original_model,model=model,client_num=args.client_num,task_num=args.task_num,
                         subset=data_spliter,class_mask=None,lr=args.lr,global_epoch=args.global_epoch,local_epoch=args.local_epoch,
                         batch_size=args.batch_size,device=args.device,method=args.method,threshold=args.threshold,
                         surrogate_data=None,test_data=None,args=args)
    myServer.start()

if __name__ == '__main__':
    parser = argparse.ArgumentParser('FedMGP training and evaluation configs')

    config = 'five_datasets_delay'

    subparser = parser.add_subparsers(dest='subparser_name')

    config_parser=None
    if config == 'cifar100_delay':
        from config.cifar100_delay import get_args_parser
        config_parser = subparser.add_parser('cifar100_delay', help='Split-CIFAR100 Delay configs')
    elif config =='five_datasets_delay':
        from config.five_datasets_delay import get_args_parser
        config_parser = subparser.add_parser('five_datasets_delay', help='5datasets delay configs')
    else:
        from config.five_datasets_delay import get_args_parser
        config_parser = subparser.add_parser('five_datasets_delay', help='5datasets delay configs')

    get_args_parser(config_parser)

    args = parser.parse_args()

    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    main(args)