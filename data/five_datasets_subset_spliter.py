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

import warnings
warnings.filterwarnings("ignore")

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


class five_datasets_Data_Spliter():

    def __init__(self,client_num,task_num,private_class_num,input_size):
        self.client_num = client_num
        self.task_num = task_num
        scale = (0.05, 1.0)
        ratio = (3. / 4., 4. / 3.)
        self.private_class_num = private_class_num
        self.input_size = input_size
        self.transform_train = build_transform(True, 224)
        self.transform_val = build_transform(False, 224)
        self.get_data_ready()

    def get_data_ready(self):
        print('getting data')
        self.nmnist_data_train = NotMNIST('./local_datasets', train=True, download=False)
        self.nmnist_data_test = NotMNIST('./local_datasets', train=False, download=False)
        print('Not-MNIST DONE')
        self.mnist_data_train =MNIST_RGB('./local_datasets', train=True, download=False)
        self.mnist_data_test =MNIST_RGB('./local_datasets', train=False, download=False)
        print('MNIST_rgb DONE')
        self.svhn_data_train =SVHN('./local_datasets', split='train', download=False)
        self.svhn_data_test =SVHN('./local_datasets', split='test', download=False)
        print('svhn DONE')
        self.cifar10_data_train = datasets.CIFAR10('./local_datasets', train=True, download=True)
        self.cifar10_data_test =datasets.CIFAR10('./local_datasets', train=False, download=True)
        print('CIFAR-10 DONE')
        self.Fmnist_data_train =FashionMNIST('./local_datasets', train=True, download=True)
        self.Fmnist_data_test =FashionMNIST('./local_datasets', train=False, download=True)
        print('Fashion-MNIST DONE')

        self.train_sets = [self.cifar10_data_train,self.Fmnist_data_train,self.nmnist_data_train,self.mnist_data_train,self.svhn_data_train,]
        self.test_sets = [self.cifar10_data_test,self.Fmnist_data_test,self.nmnist_data_test,self.mnist_data_test,self.svhn_data_test]

    def random_split(self):
        # 对每个客户端进行操作
        client_subset = [[] for i in range(0, self.client_num)]
        client_mask = [[] for i in range(0, self.client_num)]

        for idx,trainset in tqdm(enumerate(self.train_sets)):
            # 10个类别的数据分给三个客户端使用
            class_counts = torch.zeros(10)  # 每个类的数量
            class_label = []  # 每个类的index
            for i in range(10):
                class_label.append([])
            j = 0
            for x, label in trainset:
                class_counts[label] += 1
                class_label[label].append(j)
                j += 1
            # class_label 里保存了每个类的index

            # 分类,每个客户端有1个private_class
            total_private_class_num = self.client_num * 1
            public_class_num = 100 - total_private_class_num
            class_public = [i for i in range(10)]
            class_p = random.sample(class_public, total_private_class_num)
            class_public = list(set(class_public) - set(class_p))
            class_private = [[class_p[i]] for i in range(0, self.client_num)]
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
                tem_class =[]
                for pc in class_this_task:
                    tem_class.append(pc+idx*10)
                client_mask[i].append(tem_class)
                # print(client_mask[i])

                for k in class_private[i]:
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
                client_subset[i].append(CustomedSubset(trainset,index,self.transform_train,idx))

        return client_subset,client_mask


    def process_testdata(self,surrogate_num):
        trans = build_transform(False,self.input_size)

        surrodata = []
        testdata = []

        for idx,testset in enumerate(self.test_sets):
            surro_index = []
            test_index = []

            # 10个类别的数据分给三个客户端使用
            class_counts = torch.zeros(10)  # 每个类的数量
            class_label = []  # 每个类的index
            for i in range(10):
                class_label.append([])
            j = 0
            for x, label in testset:
                class_counts[label] += 1
                class_label[label].append(j)
                j += 1
            # class_label 里保存了每个类的index

            for i in range(10):
                q = 0
                unused_indice = set(class_label[i])
                while q < surrogate_num:
                    random_index = random.choice(list(unused_indice))
                    surro_index.append(random_index)
                    unused_indice.remove(random_index)
                    q += 1
                test_index.extend(list(unused_indice))
            surrodata.append(CustomedSubset(testset,surro_index,trans,idx))
            testdata.append(CustomedSubset(testset,test_index,trans,idx))
        return surrodata,testdata




class CustomedSubset(Dataset[T_co]):
    r"""
    Subset of a dataset at specified indices.

    Args:
        dataset (Dataset): The whole Dataset
        indices (sequence): Indices in the whole set selected for subset
    """
    dataset: Dataset[T_co]
    indices: Sequence[int]

    def __init__(self, dataset: Dataset[T_co], indices: Sequence[int],trans,set_count) -> None:

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
        self.set_count = set_count

    def __getitem__(self, idx):
        img, target = self.data[idx], self.targets[idx]+10*self.set_count
        # if len(img.shape)==2:
        #     img = img.unsqueeze(0)

        if type(img) == type(torch.Tensor(0)):
            img = img.cpu().numpy()
            img = Image.fromarray(img).convert('RGB')
        elif img.shape== (3, 32, 32):
            img = Image.fromarray(img.transpose(1,2,0))
        else:
            img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img,target

    def __len__(self):
        return len(self.indices)






