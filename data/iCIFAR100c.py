from numpy import ndarray
from torch import tensor
from torchvision.datasets import CIFAR100, CIFAR10
import numpy as np
from PIL import Image
import random
import torch
import torchvision
from torch.utils.data import Subset
from torchvision.datasets import MNIST
from torchvision.transforms import transforms

from utils import build_transform


class iCIFAR100c(object):
    def __init__(self,subset):
        super(iCIFAR100c,self).__init__()
        if subset.dataset:
            subset = subset.dataset

        self.data = subset.data
        self.targets = subset.targets
        self.trans = build_transform(True,224)

        self.TrainData = []
        self.TrainLabels = []
        self.TestData = []
        self.TestLabels = []

    def concatenate(self,datas,labels):
        con_data=datas[0]
        con_label=labels[0]
        for i in range(1,len(datas)):
            # print(type(con_data),type(datas[i]))
            con_data=np.concatenate((con_data,datas[i]),axis=0)
            con_label=np.concatenate((con_label,labels[i]),axis=0)
        return con_data,con_label

    def getTestData(self, classes):
        datas,labels=[],[]
        for label in range(classes[0], classes[1]):
            data = self.data[np.ndarray(self.targets) == label]
            datas.append(data)
            labels.append(np.full((data.shape[0]), label))
        self.TestData, self.TestLabels=self.concatenate(datas,labels)

    def getTrainData(self, classes):
        datas,labels=[],[]
        for label in classes:
            data=self.data[np.array(self.targets)==label]
            datas.append(data)
            labels.append(np.full((data.shape[0]),label))
        #print(f'在gettraindata里，{type(datas),type(labels)}')
        #print(f'在gettraindata里，{type(datas[0]), type(labels[0])}')
        self.TrainData, self.TrainLabels=self.concatenate(datas,labels)


    def getTrainItem(self,index):
        img, target = self.TrainData[index], self.TrainLabels[index]
        img = self.trans(Image.fromarray(img))
        return index,img,target

    def getTestItem(self,index):
        img, target = self.TestData[index], self.TestLabels[index]
        return index, img, target

    def __getitem__(self, index):
        if self.TrainData!=[]:
            return self.getTrainItem(index)
        elif self.TestData!=[]:
            return self.getTestItem(index)


    def __len__(self):
        if self.TrainData!=[]:
            return len(self.TrainData)
        elif self.TestData!=[]:
            return len(self.TestData)

    def get_image_class(self,label):
        return self.data[np.array(self.targets)==label]

