import numpy as np
import torch
from timm import create_model
import argparse
import time
from pathlib import Path
import random

import numpy as np
import torch
from torch.autograd import Variable
from torch.backends import cudnn
from torch.utils.data import DataLoader
from tqdm import tqdm

from Models.Server_DF import Server_DF
from config.cifar100_delay import get_args_parser

from timm.models import create_model
from timm.scheduler import create_scheduler
from timm.optim import create_optimizer
import Models.vision_transformer__l2p
from data.cifar100_subset_spliter import cifar100_Data_Spliter

torch.set_printoptions(threshold=float('inf'))





if __name__ == '__main__':
    parser = argparse.ArgumentParser('L2P training and evaluation configs')

    config = 'cifar100_delay'

    subparser = parser.add_subparsers(dest='subparser_name')

    config_parser=None
    if config == 'cifar100_delay':
        from config.cifar100_delay import get_args_parser
        config_parser = subparser.add_parser('cifar100_delay', help='Split-CIFAR100 L2P configs')
    else:
        config_parser = None

    get_args_parser(config_parser)

    args = parser.parse_args()

    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    cudnn.benchmark = True
    args.nb_classes = 100
    client_data, client_mask = cifar100_Data_Spliter(client_num=args.client_num, task_num=args.task_num,
                                                     private_class_num=args.private_class_num,
                                                     input_size=args.input_size).random_split()

    print(f"Creating original model: {args.model}")

    pretrained_cfg = create_model(args.model).default_cfg
    print(pretrained_cfg)


    original_model = create_model(
        args.model,
        pretrained=args.pretrained,
        num_classes=args.nb_classes,
        drop_rate=args.drop,
        drop_path_rate=args.drop_path,
        drop_block_rate=None, )

    print(f"Creating model: {args.model}")
    model = create_model(
        args.model,
        pretrained=args.pretrained,
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
    )

    torch.save(original_model.state_dict(), "original_model.pth")
    torch.save(model.state_dict(), "model.pth")
    original_model.to(device)

    if args.freeze:
        # all parameters are frozen for original vit model
        for p in original_model.parameters():
            p.requires_grad = False

        # freeze args.freeze[blocks, patch_embed, cls_token] parameters
        for n, p in model.named_parameters():
            if n.startswith(tuple(args.freeze)):
                p.requires_grad = False


    train_loader = DataLoader(client_data[0][0],batch_size=16, num_workers=4, shuffle=True)
    print(f'训练global_epoch中')
    # 先训练global_prompt

    for n, p in model.named_parameters():
        if p.requires_grad == True:
            print(n)
    model.to(device)
    optimizer =  create_optimizer(args, model)
    print(optimizer.state_dict())
    criterion = torch.nn.CrossEntropyLoss().to('cuda')

    model.train(True)
    original_model.eval()
    for epoch in tqdm(range(5)):
        for iteration, (input, target) in enumerate(train_loader):
            input = input.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)

            # input走一遍获得cls——token
            with torch.no_grad():
                if original_model is not None:
                    output = original_model(input)
                    cls_features = output['pre_logits']
                else:
                    cls_features = None

            output = model(input, task_id=0, cls_features=cls_features, train=True)

            logits = output['logits']

            # 加了class_mask
            mask = client_mask[0][0]
            not_mask = np.setdiff1d(np.arange(args.nb_classes), mask)
            not_mask = torch.tensor(not_mask, dtype=torch.int64).to('cuda')
            logits = logits.index_fill(dim=1, index=not_mask, value=float('-inf'))

            loss = criterion(logits, target)  # base criterion (CrossEntropyLoss)
            if args.pull_constraint and 'reduce_sim' in output:
                loss = loss - args.pull_constraint_coeff * output['reduce_sim']

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0)
            optimizer.step()

            print(f'训练Global_prompt过程中的loss:{loss}')
    print('Global_Prompt训练完成')

