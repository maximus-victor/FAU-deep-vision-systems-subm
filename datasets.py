from torch.utils.data import Dataset, DataLoader
from PIL import Image
import pickle


import pandas as pd
import numpy as np
import glob
import random
import joblib
import os

import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

from torchvision import transforms, datasets, models
from torchvision.datasets import ImageFolder
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

def load_image(infilename, infilegrayscale):
    """
    This function loads an image into memory when you give it the path of the image
    """
    if infilegrayscale:
      img = Image.open(infilename).convert('L')
    else:
      img = Image.open(infilename)
    #data = np.asarray(img)
    return img




def get_transforms(*, data):
    if data == 'valid':
        return A.Compose([
            A.Resize(CFG.size, CFG.size),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
            ToTensorV2(),
        ])


#with open('/content/drive/MyDrive/DEEPVIS/label_encoder.pkl', 'rb') as file:
#    lb = pickle.load(file)
    
lb = pickle.load(open('/content/drive/MyDrive/DEEPVIS/label_encoder.pkl', 'rb'))
classes_map= { 'class_name': list(lb.classes_)}
classes= pd.DataFrame(classes_map)


class WaferDataset(Dataset):
    def __init__(self, path, model=None):
        self.X = path
        self.model = model
        if self.model in ["UnNet", "RegNet", "VicNet"]:
          self.infilegrayscale = True
        else:
          self.infilegrayscale = False

        # preprocessing
        if model in ["ResNet18", "VGG16"]: # validation
            self.preprocess = transforms.Compose([
              transforms.Resize(256),
              transforms.CenterCrop(224),
              transforms.ToTensor(),
              transforms.Lambda(lambda x: x.repeat(3, 1, 1)),  # duplicate channel as Resnet18 based on RGB
              transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
              # Data augmentation
              transforms.RandomHorizontalFlip(p=0.5),
              transforms.RandomRotation(30),
              transforms.RandomVerticalFlip(p=0.5)
            ])
        elif model in ["UnNet", "RegNet", "VicNet"]:
          self.preprocess = transforms.Compose([
              transforms.Resize(256),
              transforms.ToTensor(),
              # Data augmentatio
              #transforms.RandomHorizontalFlip(p=0.5),
              #transforms.RandomRotation(30),
              #transforms.RandomVerticalFlip(p=0.5)
            ])
        elif model in ["ObjectDetect"]:
          self.preprocess = transforms.Compose([
              transforms.Resize(256),
              transforms.ToTensor(),
              transforms.Lambda(lambda x: x.repeat(3, 1, 1)), 
            ])
        
    def __len__(self):
        return (len(self.X))
    
    def __getitem__(self, i):
        image = load_image(self.X, self.infilegrayscale)
        if self.model is not None:
          image = self.preprocess(image)
        # print(image.shape)
        return torch.tensor(image, dtype=torch.float)