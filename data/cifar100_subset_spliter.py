import random
import time
from typing import TypeVar, Sequence

import numpy as np

import torch
import torchvision
from torch.nn import CrossEntropyLoss
from torch.utils.data import Subset, Dataset, DataLoader
from torchvision.datasets import MNIST,CIFAR100
from torchvision.transforms import transforms
from tqdm import tqdm
from PIL import Image

from utils import build_transform

T_co = TypeVar('T_co', covariant=True)
T = TypeVar('T')


class cifar100_Data_Spliter():

    def __init__(self,client_num,task_num,private_class_num,input_size):
        self.client_num = client_num
        self.task_num = task_num
        scale = (0.05, 1.0)
        ratio = (3. / 4., 4. / 3.)
        self.transform = transforms.Compose([
            transforms.RandomResizedCrop(input_size, scale=scale, ratio=ratio),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
        ])
        self.transform1 = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))])
        self.private_class_num = private_class_num
        self.input_size = input_size



    # 分成client_num数目个subset,每个subset里包含了task个subsubset
    def random_split(self):
        trans = build_transform(True,self.input_size)
        self.cifar100_dataset = CIFAR100(root='./local_datasets', train=True, download=True)
        trainset = self.cifar100_dataset

        # 100个类别的数据分给三个客户端使用
        class_counts = torch.zeros(100) #每个类的数量
        class_label = [] # 每个类的index
        for i in range(100):
            class_label.append([])
        j = 0
        for x, label in tqdm(trainset):
            class_counts[label] += 1
            class_label[label].append(j)
            j += 1
        # class_label 里保存了每个类的index

        # 分类
        total_private_class_num = self.client_num*self.private_class_num
        public_class_num = 100-total_private_class_num
        class_public = [i for i in range(100)]
        class_p = random.sample(class_public, total_private_class_num)
        class_public = list(set(class_public) - set(class_p))

        class_private = [class_p[self.private_class_num*i : self.private_class_num*i+self.private_class_num] for i in range(0,self.client_num)]
        for i in range(0,self.client_num):
            class_private[i].extend(class_public)
            random.shuffle(class_private[i])


        # 对每个客户端进行操作
        client_subset = [[] for i in range(0,self.client_num)]
        client_mask = [[] for i in range(0,self.client_num)]

        class_every_task = int((public_class_num+self.private_class_num)/self.task_num)
        dirichlet_perclass = {}
        for i in class_public:
            a = np.random.dirichlet(np.ones(self.client_num), 1)
            # while  (a < 0.1).any():
            #     a = np.random.dirichlet(np.ones(self.client_num), 1)
            dirichlet_perclass[i] = a[0]
        for i in range(0,self.client_num):
            for j in range(0,self.task_num):
                index = []
                class_this_task = class_private[i][j*class_every_task: j*class_every_task+class_every_task]
                client_mask[i].append(class_this_task)
                for k in class_private[i][j*class_every_task:j*class_every_task+class_every_task]:
                    if k in class_public:
                        # 是公共类
                        len = int(int(class_counts[k])*dirichlet_perclass[k][i])
                        unused_indice = set(class_label[k])
                        q = 0
                        while q < len:
                            random_index = random.choice(list(unused_indice))
                            index.append(random_index)
                            unused_indice.remove(random_index)
                            q += 1
                        class_label[k]=unused_indice
                    else: #是私有类
                        index.extend(class_label[k])
                random.shuffle(index)
                client_subset[i].append(CustomedSubset(trainset,index,trans))

        return client_subset,client_mask


    def process_testdata(self,surrogate_num):
        trans = build_transform(False,self.input_size)
        self.cifar100_dataset = CIFAR100(root='./local_datasets', train=False, download=True)
        testset = self.cifar100_dataset
        # 100个类别的数据分给三个客户端使用

        class_counts = torch.zeros(100)  # 每个类的数量
        class_label = []  # 每个类的index
        for i in range(100):
            class_label.append([])
        j = 0
        for x, label in testset:
            class_counts[label] += 1
            class_label[label].append(j)
            j += 1
        # class_label 里保存了每个类的index

        surro_index =[]
        test_index = []
        for i in tqdm(range(100)):
            q = 0
            unused_indice = set(class_label[i])
            while q < surrogate_num:
                random_index = random.choice(list(unused_indice))
                surro_index.append(random_index)
                unused_indice.remove(random_index)
                q += 1
            test_index.extend(list(unused_indice))
        surrodata = CustomedSubset(testset,surro_index,trans)
        testdata = CustomedSubset(testset,test_index,trans)
        return surrodata,testdata

    def random_split_synchron(self):
        trans = build_transform(True,self.input_size)
        self.cifar100_dataset = CIFAR100(root='./local_datasets', train=True, download=True)
        trainset = self.cifar100_dataset

        # 100个类别的数据分给三个客户端使用
        class_counts = torch.zeros(100) #每个类的数量
        class_label = [] # 每个类的index
        for i in range(100):
            class_label.append([])
        j = 0
        for x, label in tqdm(trainset):
            class_counts[label] += 1
            class_label[label].append(j)
            j += 1
        # class_label 里保存了每个类的index

        # 分类
        class_public = [i for i in range(100)]


        # 对每个客户端进行操作
        client_subset = [[] for i in range(0,self.client_num)]
        client_mask = [[] for i in range(0,self.client_num)]

        class_every_task = 10
        dirichlet_perclass = {}
        for i in class_public:
            a = np.random.dirichlet(np.ones(self.client_num), 1)
            # while  (a < 0.1).any():
            #     a = np.random.dirichlet(np.ones(self.client_num), 1)
            dirichlet_perclass[i] = a[0]
        for i in range(0,self.client_num):
            for j in range(0,self.task_num):
                index = []
                class_this_task = class_public[j*class_every_task: j*class_every_task+class_every_task]
                client_mask[i].append(class_this_task)

                # 是公共类
                for k in class_this_task:
                    len = int(int(class_counts[k])*dirichlet_perclass[k][i])
                    unused_indice = set(class_label[k])
                    q = 0
                    # print(unused_indice)
                    while q < len:
                        random_index = random.choice(list(unused_indice))
                        index.append(random_index)
                        unused_indice.remove(random_index)
                        q += 1
                    class_label[k]=unused_indice
                    random.shuffle(index)
                client_subset[i].append(CustomedSubset(trainset,index,trans))

        return client_subset,client_mask




class CustomedSubset(Dataset[T_co]):
    r"""
    Subset of a dataset at specified indices.

    Args:
        dataset (Dataset): The whole Dataset
        indices (sequence): Indices in the whole set selected for subset
    """
    dataset: Dataset[T_co]
    indices: Sequence[int]

    def __init__(self, dataset: Dataset[T_co], indices: Sequence[int],trans) -> None:

        self.indices = indices
        self.data = []
        self.targets = []
        self.dataset = dataset
        self.transform = trans
        for i in self.indices:
            self.data.append(dataset.data[i])
            self.targets.append(dataset.targets[i])
        self.data = np.array(self.data)
        self.targets = np.array(self.targets)
        self.target_transform = None

    def __getitem__(self, idx):
        img, target = self.data[idx], self.targets[idx]
        # img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)
        return img,target

    def __len__(self):
        return len(self.indices)





