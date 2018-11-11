import torch
import numpy as np
import torch.utils.data

class MyDataset(torch.utils.data.Dataset):
    def __init__(self, imgList, labelList):
        self.imgList = imgList
        self.labelList = labelList

    def __len__(self):
        return len(self.imgList)

    def __getitem__(self, idx):
        image = self.imgList[idx]
        label = self.labelList[idx]
        return (image, label)