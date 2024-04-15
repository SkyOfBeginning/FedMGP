import time
from copy import deepcopy

import numpy as np
import torch
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader

from Models.Client_DF import Client_DF
from Models.global_prompt import Global_Prompt
from utils import accuracy, global_distillation_loss


class Server_DF(object):

    def __init__(self,id,origin_model,model,client_num,task_num,subset,class_mask,lr,global_epoch,local_epoch,batch_size,device,method,threshold,surrogate_data,test_data,args):
        self.id = id
        self.vit = model
        self.origin_model = origin_model

        # 保存后的prompt
        self.global_prompt = None

        self.client_num=client_num
        self.task_num = task_num
        self.clients =[]

        # 每个客户端对应一个set
        self.client_data = subset
        self.class_mask = class_mask

        # 客户端有的验证数据集
        self.surrogate_data = surrogate_data
        self.lr = lr
        self.batch_size = batch_size
        self.global_epoch = global_epoch
        self.local_epoch = local_epoch
        self.device = device
        self.method = method
        self.threshold = threshold
        self.test_data = test_data
        self.args = args

        self.task_id = -1

    def init_client(self):
        print('Initialize clients')
        for i in range(self.client_num):
            # id,original_model,vit,task_per_global_epoch,subset,local_epoch,batch_size,lr,device,method
            if self.args.data_name == 'cifar100' or self.args.data_name == '5datasets':
                self.clients.append(Client_DF(i,self.origin_model,self.vit,self.global_epoch,self.client_data[i],
                                              self.local_epoch,self.batch_size,self.lr,self.device,self.method,self.class_mask[i],self.args
                                           ))
            else:
                self.clients.append(Client_DF(i, self.origin_model, self.vit, self.global_epoch, None,
                                              self.local_epoch, self.batch_size, self.lr, self.device, self.method,
                                              None,self.args
                                              ))
        print("Initialization completes")

    def fed_avg(self, chosen_clients):
        prompts = []
        for i in chosen_clients:
            prompts.append(self.clients[i].global_prompt)

        # key 和 prompt都要平均

        result_prompt = deepcopy(prompts[0].state_dict())

        for k in result_prompt.keys():
            for i in range(len(prompts)):
                local_model_params = prompts[i].state_dict()
                if i == 0:
                    result_prompt[k] = local_model_params[k]
                else:
                    result_prompt[k] += local_model_params[k]

            result_prompt[k] = result_prompt[k] / len(chosen_clients)

        self.global_prompt = deepcopy(self.clients[0].global_prompt)
        self.global_prompt.load_state_dict(result_prompt)
        return self.global_prompt

    def fed_avg_head(self, chosen_clients):
        heads = []
        for i in chosen_clients:
            heads.append(self.clients[i].global_head)

        # key 和 prompt都要平均
        result_head = deepcopy(heads[0].state_dict())

        for k in result_head.keys():
            for i in range(len(heads)):
                local_model_params = heads[i].state_dict()
                if i == 0:
                    result_head[k] = local_model_params[k]
                else:
                    result_head[k] += local_model_params[k]

            result_head[k] = result_head[k] / len(chosen_clients)

        self.global_head = deepcopy(self.clients[0].global_head)
        self.global_head.load_state_dict(result_head)
        print('聚合head完成')




    def train_clients(self):
        for i in range(self.task_num * self.global_epoch):
            if self.task_id != i//self.global_epoch:
                self.task_id = i//self.global_epoch
                print(self.args.data_name)
                if self.args.data_name=='office_home':
                    datas,mask = self.client_data.random_split(domain=self.task_id)
                    for j in range(self.client_num):
                        self.clients[j].get_data_office_home(self.task_id,datas[j],mask[j])
            print(f"--------round {i},task number {i//self.global_epoch}-----------")
            for j in range(self.client_num):
                self.clients[j].train(round=i,args=self.args)

            if self.args.data_name == 'office_home':
                self.fed_avg([i for i in range(self.client_num)])
                self.fed_avg_head([i for i in range(self.client_num)])
            else:
                self.filter(self.threshold)
            # self.evaluate_global()
            print('Server aggregation Compelte, results are ()：')
            for j in range(self.client_num):
                self.clients[j].set_global_prompt(self.global_prompt)
                print('-------')
        print("训练完成！")


    def filter(self,threshold = 0.7):
        chosen_clients = []
        for i in self.clients:
            classes = i.current_class
            self.surrogate_data.getTrainData(classes)
            test_loader = DataLoader(self.surrogate_data, batch_size=self.batch_size, shuffle=True,num_workers=2,pin_memory='store_true')
            acc1_num = 0
            total_num = 0
            for iteration, (index,x,y) in enumerate(test_loader):
                x = x.to(self.device)
                y = y.to(self.device)
                with torch.no_grad():
                    if self.origin_model is not None:
                        output = self.origin_model(x)
                        cls_features = output['pre_logits']
                    else:
                        cls_features = None

                    if self.method == 'delay':

                        self.vit.set_global_prompt(i.global_prompt)
                        self.vit.set_global_head(i.global_head)
                        output = self.vit.forward_server(x, cls_features=cls_features, train=False,head=i.global_head,
                                                         train_global_prompt=True)

                    logits = output['logits']

                    [acc1, acc5], number = accuracy(logits, y, topk=(1, 1))
                    acc1_num += acc1
                    total_num += number

            # 根据正确率过滤prompt
            # 达不到就略去
            # if acc1_num / total_num > threshold:
            #     print(f'{i.id}准确率为{acc1_num / total_num}满足，加入聚合')
            #     chosen_clients.append(i.id)
            # else:
            #     print((f'{i.id}准确率为{acc1_num / total_num}{i.id}有害，不加入'))
            chosen_clients.append(i.id)
        if self.method =='delay':
            self.kd_fusion_prompt(chosen_clients)
            self.kd_fusion_head(chosen_clients)
        else:
            self.fed_avg(chosen_clients)
            self.fed_avg_head(chosen_clients)

    def evaluate_global(self):
        self.vit.set_global_prompt(self.global_prompt)
        self.vit.set_global_head(self.global_head)

        test_loader = DataLoader(self.test_data, batch_size=self.batch_size, shuffle=True, num_workers=2,pin_memory='store_true')
        acc1_num = 0
        total_num = 0
        acc5_num = 0
        for iteration, (x, y) in enumerate(test_loader):
            x = x.to(self.device)
            y = y.to(self.device)
            with torch.no_grad():
                if self.origin_model is not None:
                    output = self.origin_model(x)
                    cls_features = output['pre_logits']
                else:
                    cls_features = None

                if self.method == 'delay':
                    output = self.vit.forward_server(x, cls_features=cls_features, train=False, head=self.global_head,
                                                     train_global_prompt=True)

                logits = output['logits']

                [acc1, acc5], number = accuracy(logits, y, topk=(1, 5))
                acc1_num += acc1
                total_num += number
                acc5_num +=acc5

        print(
            f'聚合后，服务器在全部数据集上的准确率为acc1={acc1_num / total_num}\t{acc5_num / total_num}')

    def kd_fusion_prompt(self,chosen_clients):
        my_result = None
        if len(chosen_clients)==1:
            my_result= deepcopy(self.clients[chosen_clients[0]].global_prompt)
        else:
            # 做数据集
            classes = set()
            for i in chosen_clients:
                temp = set(self.clients[i].current_class)
                classes = set.union(classes,temp)
            self.surrogate_data.getTrainData(list(classes))

            # 选此作为初始变量
            my_result = deepcopy(self.clients[chosen_clients[0]].global_prompt)
            test_loader = DataLoader(self.surrogate_data, batch_size=self.batch_size, shuffle=True, num_workers=2)
            self.vit.to(self.device)

            optimizer = torch.optim.Adam(my_result.parameters(), lr=self.lr, weight_decay=1e-03)
            for h in range(self.local_epoch):
                for iteration, (index,x,y) in enumerate(test_loader):
                    x = Variable(x, requires_grad=True).to(self.device, non_blocking=True)
                    y = y.to(self.device)
                    my_result.to(self.device)
                    with torch.no_grad():
                        if self.origin_model is not None:
                            output = self.origin_model(x)
                            cls_features = output['pre_logits']
                        else:
                            cls_features = None
                    # 提取的特征
                    self.vit.set_global_prompt(my_result)
                    output = self.vit.forward_features_l2p(x, cls_features=cls_features, train=False)['x']
                    outputs = []
                    with torch.no_grad():
                        for localmodel in chosen_clients[1:]:
                            prompt = self.clients[localmodel].global_prompt
                            self.vit.set_global_prompt(prompt)
                            my = self.vit.forward_features_l2p(x, cls_features=cls_features, train=False)['x']
                            outputs.append(my)

                    loss = global_distillation_loss(output,outputs)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

            self.global_prompt = my_result
            print('prompt fusion 完成')
        return my_result


    def kd_fusion_head(self,chosen_clients):
        my_result = None
        if len(chosen_clients) == 1:
            my_result = deepcopy(self.clients[chosen_clients[0]].global_head)
        else:
            # 做数据集
            classes = set()
            for i in chosen_clients:
                temp = set(self.clients[i].current_class)
                classes = set.union(classes, temp)
            self.surrogate_data.getTrainData(list(classes))

            # 选此作为初始变量
            my_result = deepcopy(self.clients[chosen_clients[0]].global_head)
            test_loader = DataLoader(self.surrogate_data, batch_size=self.batch_size, shuffle=True, num_workers=2)
            self.vit.to(self.device)
            self.vit.set_global_prompt(self.global_prompt)
            optimizer = torch.optim.Adam(my_result.parameters(), lr=self.lr, weight_decay=1e-03)
            criterion = torch.nn.CrossEntropyLoss().to(self.device)
            for h in range(self.local_epoch):
                for iteration, (index, x, y) in enumerate(test_loader):
                    x = Variable(x, requires_grad=True).to(self.device, non_blocking=True)
                    y = y.to(self.device)
                    my_result.to(self.device)
                    with torch.no_grad():
                        if self.origin_model is not None:
                            output = self.origin_model(x)
                            cls_features = output['pre_logits']
                        else:
                            cls_features = None
                    # 提取的特征
                    self.vit.set_global_head(my_result)
                    output = self.vit(x, task_id=self.task_id, cls_features=cls_features, train=True,
                                      train_global_prompt=True)

                    logits = output['logits']
                    loss = criterion(logits, y)  # base criterion (CrossEntropyLoss)
                    optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(my_result.parameters(), 1.0)
                    optimizer.step()


            self.global_head = my_result


    def start(self):
        self.init_client()
        self.train_clients()
        # self.test_surro()



    def test_surro(self):
        self.surrogate_data.getTrainData([1,2,3,4,5])
        test_loader = DataLoader(self.surrogate_data, batch_size=self.batch_size, shuffle=True, num_workers=2,
                                 pin_memory='store_true')
        acc1_num = 0
        total_num = 0
        for iteration, (index, x, y) in enumerate(test_loader):
            x = x.to(self.device)
            y = y.to(self.device)
            with torch.no_grad():
                if self.origin_model is not None:
                    output = self.origin_model(x)
                    cls_features = output['pre_logits']
                else:
                    cls_features = None

                if self.method == 'delay':
                    self.vit.set_global_prompt(i.global_prompt)
                    self.vit.set_global_head(i.global_head)
                    output = self.vit.forward_server(x, cls_features=cls_features, train=False, head=i.global_head,
                                                     train_global_prompt=True)

                logits = output['logits']

                [acc1, acc5], number = accuracy(logits, y, topk=(1, 1))




