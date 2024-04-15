import deeplake
import random
import random
import time
from typing import TypeVar, Sequence
import torch
from torch.utils.data.dataset import Subset, Dataset
from torchvision import datasets, transforms

from timm.data import create_transform
from tqdm import tqdm

from data.continual_datasets import *

import utils


T_co = TypeVar('T_co', covariant=True)
T = TypeVar('T')



def build_transform(is_train, inputsize):
    resize_im = inputsize > 32
    if is_train:
        scale = (0.05, 1.0)
        ratio = (3. / 4., 4. / 3.)
        transform = transforms.Compose([
            transforms.RandomResizedCrop(inputsize, scale=scale, ratio=ratio),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
        ])
        return transform

    t = []
    if resize_im:
        size = int((256 / 224) * inputsize)
        t.append(
            transforms.Resize(size, interpolation=3),  # to maintain same ratio w.r.t. 224 images
        )
        t.append(transforms.CenterCrop(inputsize))
    t.append(transforms.ToTensor())

    return transforms.Compose(t)

# class DataloadertoDataset(Dataset[T_co]):
#     dataset: Dataset[T_co]
#     indices: Sequence[int]
#
#     def __init__(self, dataloader) -> None:
#         self.datas = [[],[],[],[]]
#         self.targetss =[[],[],[],[]]
#         for i in tqdm(dataloader):
#             self.datas[int(i['domain_categories'])].append(i['images'].squeeze(0))
#             self.targetss[int(i['domain_categories'])].append(int(i['domain_objects']))
#         del dataloader
#         for i in range(len(self.datas)):
#             self.datas[i] = np.array(self.datas[i])
#
#
#     def set_domin(self,i):
#         self.data = self.datas[i]
#         self.targets = np.array(self.targetss[i])
#
#     def __getitem__(self, idx):
#         img, target = self.data[idx], self.targets[idx]
#         return img,target
#
#     def __len__(self):
#         return len(self.dataloder)





class officehome_spliter():
    def __init__(self,client_num,task_num,private_class_num,input_size):
        self.client_num = client_num
        self.task_num = task_num
        scale = (0.05, 1.0)
        ratio = (3. / 4., 4. / 3.)
        self.private_class_num = private_class_num
        self.input_size = input_size
        self.transform_train = build_transform(True, 224)
        self.transform_val = build_transform(False, 224)

        # os.environ['DEEPLAKE_DOWNLOAD_PATH'] = './data/local_datasets'
        # ds = deeplake.load('hub://activeloop/office-home-domain-adaptation',access_method='local')
        # dataloader = ds.pytorch(num_workers=4, batch_size=1, shuffle=False)
        #
        # self.dataset = dataloader.dataset
        # del dataloader,ds

    def random_split(self,domain):
        # 对每个客户端进行操作
        self.dataset = Office_Home('./local_datasets', train=True, download=False,domain=domain)

        client_subset = [[] for i in range(0, self.client_num)]
        client_mask = [[] for i in range(0, self.client_num)]

        surro_data=[]

            # 10个类别的数据分给三个客户端使用

        class_counts = torch.zeros(65)  # 每个类的数量
        class_label = []  # 每个类的inde
        for i in range(65):
            class_label.append([])
        j = 0
        for x, label in self.dataset:
            class_counts[label] += 1
            class_label[label].append(j)
            j += 1
        # class_label 里保存了每个类的index
        class_public = [i for i in range(65)]
        # surro_temp =[]
        # for i in class_public:
        #     unused_indice = set(class_label[i])
        #     q = 0
        #     while q < 10:
        #         random_index = random.choice(list(unused_indice))
        #         surro_temp.append(random_index)
        #         q += 1
        #
        # surro_data.append(CustomedSurroSubset(self.dataset,surro_temp,self.transform_train,idx))


        # 分类,每个客户端有1个private_class
        total_private_class_num = self.client_num * 10
        public_class_num = 65 - total_private_class_num
        class_p = random.sample(class_public, total_private_class_num)
        class_public = list(set(class_public) - set(class_p))
        class_private = [class_p[10*i : 10*i+10] for i in range(0,self.client_num)]
        for i in range(0, self.client_num):
            class_private[i].extend(class_public)
            random.shuffle(class_private[i])

        # using Dirichlet to change distribution of Public_Class
        dirichlet_perclass = {}
        for i in class_public:
            a = np.random.dirichlet(np.ones(self.client_num), 1)
            while  (a < 0.1).any():
                a = np.random.dirichlet(np.ones(self.client_num), 1)
            dirichlet_perclass[i] = a[0]

        for i in range(0, self.client_num):
            index = []
            class_this_task = class_private[i]
            client_mask[i]=class_this_task
            # print(client_mask[i])

            for k in class_private[i]:
                if k in class_public:
                    # 是公共类
                    len = int(int(class_counts[k])*0.8)
                    unused_indice = set(class_label[k])
                    q = 0
                    while q < len:
                        random_index = random.choice(list(unused_indice))
                        index.append(random_index)
                        q += 1
                else: #是私有类
                    index.extend(class_label[k])
            random.shuffle(index)
            client_subset[i]=CustomedSubset(self.dataset,index,self.transform_train)


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
        img = self.transform(Image.fromarray(img))

        return img,target

    def __len__(self):
        return len(self.indices)


class CustomedSurroSubset(Dataset[T_co]):
    r"""
    Subset of a dataset at specified indices.

    Args:
        dataset (Dataset): The whole Dataset
        indices (sequence): Indices in the whole set selected for subset
    """
    dataset: Dataset[T_co]
    indices: Sequence[int]

    def __init__(self, dataset: Dataset[T_co], indices: Sequence[int], trans,domain) -> None:
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
        self.domain = domain

    def __getitem__(self, idx):
        img, target = self.data[idx], self.targets[idx]
        img = img.cpu().numpy()
        img = Image.fromarray(img)

        return img, target

    def __len__(self):
        return len(self.indices)


