from copy import deepcopy
import random

import numpy as np
import torch
from timm.optim import create_optimizer
from torch import nn
from torch.autograd import Variable
from torch.utils.data import random_split, DataLoader, Dataset
from tqdm import tqdm

from data.cifar100_subset_spliter import CustomedSubset
from data.iCIFAR100c import iCIFAR100c
from utils import accuracy


# 每个客户端拥有的
class Client_DF(object):

    def __init__(self,id,original_model,vit,task_per_global_epoch,subset,local_epoch,batch_size,lr,device,method,class_mask,args):
        self.id = id
        self.vit = vit
        self.original_model = original_model

        self.global_prompt = None
        self.local_prompt = None
        self.global_head= None
        self.local_head = None

        self.task_id = -1
        self.task_per_global_epoch = task_per_global_epoch
        self.test_loader=[]
        # subset应该是一个【】，其中包含了num_task个数据以及类别，以[[(类别)：[数据]]，{}]的形式保存
        self.train_data =subset
        self.local_epoch = local_epoch
        self.batch_size = batch_size
        self.lr = lr
        self.device=device
        self.method =method
        if args.data_name == 'cifar100' or args.data_name=='5datasets':
            self.class_mask = class_mask
        else:
            self.class_mask = []



    def get_global_prompt(self):
        self.global_prompt = self.vit.get_global_prompt()
        self.vit.init_global_prompt()

    def set_vit_global_prompt(self):
        self.vit.set_global_prompt(self.global_prompt)

    def set_vit_local_prompt(self):
        self.vit.set_local_prompt(self.local_prompt)

    def get_local_prompt(self):
        self.local_prompt = self.vit.get_local_prompt()
        self.vit.init_local_prompt()

    def set_local_prompt(self,local_prompt):
        self.local_prompt = deepcopy(local_prompt)

    def set_global_prompt(self,global_prompt):
        self.global_prompt = deepcopy(global_prompt)
        self.global_evaluate(task=self.task_id,test_global=True,nb_classes=self.nb_classes,topk=(1,self.topk))
        self.global_evaluate(task=self.task_id,test_global=False,nb_classes=self.nb_classes,topk=(1,self.topk))

    def get_head(self):
        self.global_head,self.local_head = self.vit.get_head()
        self.vit.init_head()

    def set_global_head(self,global_head):
        self.global_head=deepcopy(global_head)

    def set_vit_head(self):
        self.vit.set_head(self.global_head,self.local_head)

    # def forward(self,x,task_id=-1, cls_features=None, train=False,train_global_prompt=False):
    #     res = self.vit( x, task_id, cls_features, train,train_global_prompt)
    #     return res


    # def get_data(self, task_id):

    def get_data_office_home(self,task_id,data,mask):
        self.train_dataset = data
        self.current_class = mask
        self.class_mask.append(mask)
        # self.train_dataset = self.train_data[task_id]
        # self.current_class = self.class_mask[task_id]
        print(f'{self.id} client，{task_id} task has {len(self.current_class)} classes:{self.current_class}')
        trainset = self.train_dataset
        traindata, testdata = random_split(trainset,
                                           [int(len(trainset) * 0.7), len(trainset) - int(len(trainset) * 0.7)])
        testdata = deepcopy(testdata)
        self.test_loader.append(testdata)

        self.traindata = traindata

    def get_data(self,task_id):
        self.train_dataset = self.train_data[task_id]
        self.current_class = self.class_mask[task_id]
        print(f'{self.id} client，{task_id} task has {len(self.current_class)} classes:{self.current_class}')
        trainset = self.train_dataset
        traindata, testdata = random_split(trainset,
                                           [int(len(trainset) * 0.7), len(trainset) - int(len(trainset) * 0.7)])
        testdata = deepcopy(testdata)
        self.test_loader.append(testdata)

        self.traindata = traindata


    def train(self, round, args):
        self.topk = args.top_k
        self.nb_classes = args.nb_classes
        self.original_model.eval()
        if self.global_prompt !=None:
            self.set_vit_head()
            self.set_vit_global_prompt()
            self.set_vit_local_prompt()
            print('已设置')

        ###train###
        # 判断是否要更新数据
        task = round // self.task_per_global_epoch
        if self.task_id != task:
            if args.data_name == 'cifar100' or args.data_name=='5datasets':
                self.get_data(task)
            self.task_id = task

        self.vit.to(self.device)
        train_loader = DataLoader(self.traindata, batch_size=self.batch_size, num_workers=args.num_workers,
            pin_memory=args.pin_mem, shuffle=True)
        print(f'Client {self.id} on Task {self.task_id} is training global prompts')
        # 先训练global_prompt
        for n, p in self.vit.named_parameters():
            if n.startswith(tuple(['local_prompt', 'head_local'])):
                p.requires_grad = False
            if n.startswith(tuple(['global_prompt', 'head.'])):
                p.requires_grad = True

        # for n, p in self.vit.named_parameters():
        #     if p.requires_grad ==True:
        #         print(n)
        optimizer = torch.optim.Adam(self.vit.parameters(), lr=self.lr,weight_decay=1e-03)
        criterion = torch.nn.CrossEntropyLoss().to(self.device)
        for epoch in tqdm(range(self.local_epoch)):
            for iteration, (input,target) in enumerate(train_loader):
                input, target = Variable(input, requires_grad=True).to(self.device, non_blocking=True), target.to(self.device,non_blocking=True)
                # input走一遍获得cls——token
                with torch.no_grad():
                    if self.original_model is not None:
                        output = self.original_model(input)
                        cls_features = output['pre_logits']
                    else:
                        cls_features = None
                if self.method == 'delay':
                    output = self.vit(input, task_id=self.task_id, cls_features=cls_features, train=True,
                                   train_global_prompt=True)
                else:
                    output = self.vit(input, task_id=self.task_id, cls_features=cls_features, train=True)
                logits = output['logits']

                # 加了class_mask
                mask = self.current_class
                not_mask = np.setdiff1d(np.arange(args.nb_classes), mask)
                not_mask = torch.tensor(not_mask, dtype=torch.int64).to(self.device)
                logits = logits.index_fill(dim=1, index=not_mask, value=float('-inf'))


                loss = criterion(logits, target)  # base criterion (CrossEntropyLoss)
                if args.pull_constraint and 'reduce_sim' in output:
                    loss = loss - args.pull_constraint_coeff * output['reduce_sim']
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.vit.parameters(), args.clip_grad)
                optimizer.step()


        print('Global_Prompt training completes!')

        # 训练local_prompt
        for n, p in self.vit.named_parameters():
            if n.startswith(tuple(['local_prompt', 'head_local'])):
                p.requires_grad = True
            if n.startswith(tuple(['global_prompt', 'head.'])):
                p.requires_grad = False
        # for n, p in self.vit.named_parameters():
        #     if p.requires_grad == True:
        #         print(n)
        optimizer = create_optimizer(args, self.vit)
        criterion = torch.nn.CrossEntropyLoss().to(self.device)
        for epoch in tqdm(range(self.local_epoch)):
            for iteration, (input,target) in enumerate(train_loader):
                input, target = Variable(input.float(), requires_grad=True).to(self.device, non_blocking=True), target.to(self.device,non_blocking=True)

                with torch.no_grad():
                    if self.original_model is not None:
                        output = self.original_model(input)
                        cls_features = output['pre_logits']
                    else:
                        cls_features = None
                if self.method == 'delay':
                    output = self.vit(input, task_id=self.task_id,class_id = target, cls_features=cls_features, train=True,
                                   train_global_prompt=False)
                else:
                    output = self.vit(input, task_id=self.task_id, cls_features=cls_features, train=True)
                logits = output['logits']

                # 加了class_mask
                mask = self.current_class
                not_mask = np.setdiff1d(np.arange(args.nb_classes), mask)
                not_mask = torch.tensor(not_mask, dtype=torch.int64).to(self.device)
                logits = logits.index_fill(dim=1, index=not_mask, value=float('-inf'))

                loss = criterion(logits, target)  # base criterion (CrossEntropyLoss)
                if args.pull_constraint and 'reduce_sim' in output:
                    loss = loss - args.pull_constraint_coeff * output['reduce_sim']
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.vit.parameters(),args.clip_grad)
                optimizer.step()

        print('Local Prompt training completes!')

        ###评估###
        print('using global prompts...')
        self.evaluate(True, 0,args.nb_classes,(1,args.top_k))
        self.evaluate(True, self.task_id, args.nb_classes, (1, args.top_k))
        print('using local prompts...')
        self.evaluate(False, 0, args.nb_classes, (1, args.top_k))
        self.evaluate(False, self.task_id, args.nb_classes, (1, args.top_k))

        ###get_prompt###
        self.get_global_prompt()
        self.get_local_prompt()
        self.get_head()


    def global_evaluate(self,task=0,test_global = True,nb_classes=None,topk=(1,5)):
        self.original_model.eval()
        if self.global_prompt != None:
            self.set_vit_head()
            self.set_vit_global_prompt()
            self.set_vit_local_prompt()
        test_data = self.test_loader[task]
        test_loader = DataLoader(test_data, batch_size=8, shuffle=True, num_workers=4)
        correct = 0
        total = 0
        for iteration, (input, target) in enumerate(test_loader):
            input = input.to(self.device, non_blocking=True)
            target = target.to(self.device, non_blocking=True)

            with torch.no_grad():
                if self.original_model is not None:
                    output = self.original_model(input)
                    cls_features = output['pre_logits']
                else:
                    cls_features = None
                if self.method == 'delay':
                    output = self.vit(input, task_id=self.task_id, cls_features=cls_features, train=False,
                                      train_global_prompt=test_global)
                else:
                    output = self.vit(input, task_id=self.task_id, cls_features=cls_features, train=False)

                logits = output['logits']
            # 加了class_mask
            mask = self.class_mask[task]
            not_mask = np.setdiff1d(np.arange(nb_classes), mask)
            not_mask = torch.tensor(not_mask, dtype=torch.int64).to(self.device)
            logits = logits.index_fill(dim=1, index=not_mask, value=float('-inf'))

            predicts = torch.max(logits, dim=1)[1].cpu()
            # print(predicts)
            # print(target)
            correct += (predicts == target.cpu()).sum()
            total += len(target)

        acc = 100 * correct / total
        print(f'{acc}')


    def evaluate(self,test_global=True,task=0,nb_classes=None,topk=(1,5)):
        test_data = self.test_loader[task]
        test_loader = DataLoader(test_data,batch_size=8,shuffle=True,num_workers=2)
        correct =0
        total = 0
        for iteration, (input, target) in enumerate(test_loader):
            input = input.to(self.device, non_blocking=True)
            target = target.to(self.device, non_blocking=True)

            with torch.no_grad():
                if self.original_model is not None:
                    output = self.original_model(input)
                    cls_features = output['pre_logits']
                else:
                    cls_features = None

                if self.method == 'delay':
                    output = self.vit(input, task_id=self.task_id, cls_features=cls_features, train=False,
                                   train_global_prompt=test_global)
                else:
                    output = self.vit(input, task_id=self.task_id, cls_features=cls_features, train=False)

                logits = output['logits']

            # 加了class_mask
            mask = self.class_mask[task]
            not_mask = np.setdiff1d(np.arange(nb_classes), mask)
            not_mask = torch.tensor(not_mask, dtype=torch.int64).to(self.device)
            logits = logits.index_fill(dim=1, index=not_mask, value=float('-inf'))

            predicts = torch.max(logits, dim=1)[1].cpu()
            # print(predicts)
            # print(target)
            correct += (predicts == target.cpu()).sum()
            total += len(target)

        acc = 100 * correct / total
        if test_global:
            prompt = 'Global Prompt'
        else:
            prompt = 'Local Prompt'
        print(f'Client {self.id} on Task {task} acc is {acc}, using')

